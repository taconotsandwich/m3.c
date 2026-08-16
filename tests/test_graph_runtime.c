/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3.h"

#include <math.h>
#include <string.h>

typedef struct {
    m3_graph *graph;
    m3_graph_value_id input;
    m3_graph_value_id constant;
    m3_graph_value_id temporary[3];
    m3_graph_value_id output;
} m3_graph_test_fixture;

static bool m3_graph_test_add_value(
    m3_graph *graph, m3_graph_value_role role, m3_graph_value_id *value,
    m3_error *error)
{
    const uint64_t shape[] = {4U};
    m3_graph_value_desc description;

    m3_graph_value_desc_init(&description, role, M3_DTYPE_F32, 1U, shape);
    return m3_graph_add_value(graph, &description, value, error) ==
        M3_STATUS_OK;
}

static bool m3_graph_test_build(m3_graph_test_fixture *fixture,
                                m3_error *error)
{
    static const float constant[] = {0.5F, -1.0F, 2.0F, 3.0F};
    m3_graph_node_desc node;
    bool ready;

    (void)memset(fixture, 0, sizeof(*fixture));
    ready = m3_graph_create(&fixture->graph, error) == M3_STATUS_OK &&
        m3_graph_test_add_value(fixture->graph, M3_GRAPH_INPUT,
                                &fixture->input, error) &&
        m3_graph_test_add_value(fixture->graph, M3_GRAPH_CONSTANT,
                                &fixture->constant, error) &&
        m3_graph_set_constant(fixture->graph, fixture->constant, constant,
                              sizeof(constant), error) == M3_STATUS_OK;
    m3_graph_node_desc_init(&node, M3_OP_ADD);
    node.inputs[0] = fixture->input;
    node.inputs[1] = fixture->constant;
    ready = ready && m3_graph_add_inferred_value(
        fixture->graph, &node, 0U, M3_GRAPH_TEMPORARY, M3_DTYPE_F32,
        &fixture->temporary[0], error) == M3_STATUS_OK;
    node.outputs[0] = fixture->temporary[0];
    ready = ready &&
        m3_graph_add_node(fixture->graph, &node, error) == M3_STATUS_OK;
    m3_graph_node_desc_init(&node, M3_OP_TANH);
    node.inputs[0] = fixture->temporary[0];
    ready = ready && m3_graph_add_inferred_value(
        fixture->graph, &node, 0U, M3_GRAPH_TEMPORARY, M3_DTYPE_F32,
        &fixture->temporary[1], error) == M3_STATUS_OK;
    node.outputs[0] = fixture->temporary[1];
    ready = ready &&
        m3_graph_add_node(fixture->graph, &node, error) == M3_STATUS_OK;
    m3_graph_node_desc_init(&node, M3_OP_MUL);
    node.inputs[0] = fixture->temporary[1];
    node.inputs[1] = fixture->constant;
    ready = ready && m3_graph_add_inferred_value(
        fixture->graph, &node, 0U, M3_GRAPH_TEMPORARY, M3_DTYPE_F32,
        &fixture->temporary[2], error) == M3_STATUS_OK;
    node.outputs[0] = fixture->temporary[2];
    ready = ready &&
        m3_graph_add_node(fixture->graph, &node, error) == M3_STATUS_OK;
    m3_graph_node_desc_init(&node, M3_OP_ADD);
    node.inputs[0] = fixture->temporary[2];
    node.inputs[1] = fixture->input;
    ready = ready && m3_graph_add_inferred_value(
        fixture->graph, &node, 0U, M3_GRAPH_OUTPUT, M3_DTYPE_F32,
        &fixture->output, error) == M3_STATUS_OK;
    node.outputs[0] = fixture->output;
    ready = ready &&
        m3_graph_add_node(fixture->graph, &node, error) == M3_STATUS_OK;
    if (!ready) {
        m3_graph_free(fixture->graph);
        fixture->graph = NULL;
    }
    return ready;
}

void m3_test_graph_cpu_liveness(m3_test_context *test)
{
    static const float input[] = {1.0F, 2.0F, -3.0F, 0.25F};
    static const float constant[] = {0.5F, -1.0F, 2.0F, 3.0F};
    m3_graph_test_fixture fixture;
    m3_session_options options;
    m3_session_plan_info plan;
    m3_session *session = NULL;
    m3_provider_kind provider = M3_PROVIDER_NEURAL_ENGINE;
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    m3_error error;
    size_t index;
    bool ready = m3_graph_test_build(&fixture, &error);

    M3_TEST_EXPECT(test, ready, "build public CPU graph");
    if (!ready) {
        return;
    }
    m3_session_options_init(&options);
    options.allow_neural_engine = false;
    options.allow_metal = false;
    M3_TEST_EXPECT(
        test,
        m3_session_create(&session, fixture.graph, &options, &error) ==
            M3_STATUS_OK,
        "compile CPU graph session");
    m3_graph_free(fixture.graph);
    fixture.graph = NULL;
    if (session == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_session_get_plan_info(session, &plan, &error) == M3_STATUS_OK &&
            plan.value_count == 6U && plan.node_count == 4U &&
            plan.storage_slot_count == 5U &&
            plan.allocated_bytes == 5U * sizeof(input),
        "liveness plan reuses one expired temporary slot");
    for (index = 0U; index < 4U; ++index) {
        bool selected = m3_session_get_node_provider(
            session, index, &provider, &error) == M3_STATUS_OK &&
            provider == M3_PROVIDER_CPU;

        M3_TEST_EXPECT(test, selected, "CPU-only graph selects CPU provider");
    }
    M3_TEST_EXPECT(
        test,
        m3_session_write_input(session, fixture.input, input,
                               sizeof(input), &error) == M3_STATUS_OK &&
            m3_session_run(session, &error) == M3_STATUS_OK &&
            m3_session_read_output(session, fixture.output, output,
                                   sizeof(output), &error) == M3_STATUS_OK,
        "CPU graph executes after source graph disposal");
    for (index = 0U; index < 4U; ++index) {
        float expected = tanhf(input[index] + constant[index]) *
            constant[index] + input[index];

        M3_TEST_EXPECT_F32(test, output[index], expected, 0.0F, 0.0F,
                           "CPU graph preserves operation order");
    }
    M3_TEST_EXPECT(
        test,
        m3_session_write_input(session, fixture.output, input,
                               sizeof(input), NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_session_read_output(session, fixture.input, output,
                                   sizeof(output), NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_session_write_input(session, fixture.input, input,
                                   sizeof(input) - 1U, NULL) ==
                M3_STATUS_INVALID_ARGUMENT,
        "session I/O enforces value roles and exact byte counts");
    m3_session_free(session);
}

void m3_test_graph_validation(m3_test_context *test)
{
    const uint64_t shape[] = {2U};
    m3_graph_value_desc value_description;
    m3_graph_node_desc node;
    m3_graph *graph = NULL;
    m3_session *session = NULL;
    m3_graph_value_id input;
    m3_graph_value_id inferred = M3_GRAPH_VALUE_NONE;
    m3_graph_value_id temporary;
    m3_graph_value_id output;
    m3_error error;
    bool ready;

    m3_graph_value_desc_init(&value_description, M3_GRAPH_INPUT,
                             M3_DTYPE_F32, 1U, shape);
    ready = m3_graph_create(&graph, &error) == M3_STATUS_OK &&
        m3_graph_add_value(graph, &value_description, &input, &error) ==
            M3_STATUS_OK;
    value_description.role = M3_GRAPH_TEMPORARY;
    ready = ready &&
        m3_graph_add_value(graph, &value_description, &temporary, &error) ==
            M3_STATUS_OK;
    value_description.role = M3_GRAPH_OUTPUT;
    ready = ready &&
        m3_graph_add_value(graph, &value_description, &output, &error) ==
            M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "build malformed topology values");
    if (!ready) {
        m3_graph_free(graph);
        return;
    }
    m3_graph_node_desc_init(&node, M3_OP_ADD);
    node.inputs[0] = temporary;
    node.inputs[1] = input;
    node.outputs[0] = output;
    M3_TEST_EXPECT(
        test,
        m3_graph_add_node(graph, &node, &error) == M3_STATUS_OK &&
            m3_session_create(&session, graph, NULL, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            session == NULL,
        "session compilation rejects read-before-produce atomically");
    m3_graph_node_desc_init(&node, M3_OP_ADD);
    node.inputs[1] = input;
    M3_TEST_EXPECT(
        test,
        m3_graph_add_inferred_value(
            graph, &node, 0U, M3_GRAPH_TEMPORARY, M3_DTYPE_F32,
            &inferred, NULL) == M3_STATUS_OUT_OF_RANGE &&
            inferred == M3_GRAPH_VALUE_NONE,
        "shape inference rejects an absent required input atomically");
    node.kind = (m3_op_kind)999;
    node.inputs[0] = input;
    M3_TEST_EXPECT(
        test,
        m3_graph_add_inferred_value(
            graph, &node, 0U, M3_GRAPH_TEMPORARY, M3_DTYPE_F32,
            &inferred, NULL) == M3_STATUS_UNSUPPORTED &&
            inferred == M3_GRAPH_VALUE_NONE,
        "shape inference rejects an unsupported operation atomically");
    m3_graph_free(graph);

    graph = NULL;
    session = NULL;
    m3_graph_value_desc_init(&value_description, M3_GRAPH_CONSTANT,
                             M3_DTYPE_F32, 1U, shape);
    ready = m3_graph_create(&graph, &error) == M3_STATUS_OK &&
        m3_graph_add_value(graph, &value_description, &input, &error) ==
            M3_STATUS_OK;
    value_description.role = M3_GRAPH_OUTPUT;
    ready = ready &&
        m3_graph_add_value(graph, &value_description, &output, &error) ==
            M3_STATUS_OK;
    m3_graph_node_desc_init(&node, M3_OP_COPY);
    node.inputs[0] = input;
    node.outputs[0] = output;
    ready = ready && m3_graph_add_node(graph, &node, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "build unset-constant graph");
    M3_TEST_EXPECT(
        test,
        ready && m3_session_create(&session, graph, NULL, NULL) ==
                     M3_STATUS_INVALID_ARGUMENT &&
            session == NULL,
        "session compilation rejects an unset constant without diagnostics");
    m3_graph_free(graph);
}
