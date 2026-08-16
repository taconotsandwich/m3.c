/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3.h"

void m3_test_graph_metal_provider(m3_test_context *test)
{
    const uint64_t shape[] = {4U};
    const float input[] = {1.0F, -2.0F, 3.0F, -4.0F};
    const float constant[] = {2.0F, 3.0F, -1.0F, 0.5F};
    float output[] = {0.0F, 0.0F, 0.0F, 0.0F};
    m3_graph_value_desc value_description;
    m3_graph_node_desc node;
    m3_graph *graph = NULL;
    m3_session *session = NULL;
    m3_graph_value_id input_value = M3_GRAPH_VALUE_NONE;
    m3_graph_value_id constant_value = M3_GRAPH_VALUE_NONE;
    m3_graph_value_id output_value = M3_GRAPH_VALUE_NONE;
    m3_provider_kind provider = M3_PROVIDER_CPU;
    m3_error error;
    bool ready;
    size_t index;

    m3_graph_value_desc_init(&value_description, M3_GRAPH_INPUT,
                             M3_DTYPE_F32, 1U, shape);
    ready = m3_graph_create(&graph, &error) == M3_STATUS_OK &&
        m3_graph_add_value(graph, &value_description, &input_value,
                           &error) == M3_STATUS_OK;
    value_description.role = M3_GRAPH_CONSTANT;
    ready = ready &&
        m3_graph_add_value(graph, &value_description, &constant_value,
                           &error) == M3_STATUS_OK &&
        m3_graph_set_constant(graph, constant_value, constant,
                              sizeof(constant), &error) == M3_STATUS_OK;
    value_description.role = M3_GRAPH_OUTPUT;
    ready = ready &&
        m3_graph_add_value(graph, &value_description, &output_value,
                           &error) == M3_STATUS_OK;
    m3_graph_node_desc_init(&node, M3_OP_MUL);
    node.inputs[0] = input_value;
    node.inputs[1] = constant_value;
    node.outputs[0] = output_value;
    ready = ready && m3_graph_add_node(graph, &node, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "build public Metal graph");
    if (!ready) {
        m3_graph_free(graph);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_session_create(&session, graph, NULL, &error) == M3_STATUS_OK &&
            m3_session_get_node_provider(session, 0U, &provider, &error) ==
                M3_STATUS_OK &&
            provider == M3_PROVIDER_METAL,
        "default graph session selects Metal before CPU");
    m3_graph_free(graph);
    if (session == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_session_write_input(session, input_value, input, sizeof(input),
                               &error) == M3_STATUS_OK &&
            m3_session_run(session, &error) == M3_STATUS_OK &&
            m3_session_read_output(session, output_value, output,
                                   sizeof(output), &error) == M3_STATUS_OK,
        "Metal graph executes through the heterogeneous scheduler");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT_F32(test, output[index], input[index] * constant[index],
                           0.0F, 0.0F,
                           "Metal graph output matches exact product");
    }
    m3_session_free(session);
}
