/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3.h"
#include "m3_test.h"

static const uint8_t m3_graph_coreml_model[] = {
    0x08U, 0x04U, 0x12U, 0x20U, 0x0aU, 0x0eU, 0x0aU, 0x01U,
    0x78U, 0x1aU, 0x09U, 0x2aU, 0x07U, 0x0aU, 0x01U, 0x04U,
    0x10U, 0xa0U, 0x80U, 0x04U, 0x52U, 0x0eU, 0x0aU, 0x01U,
    0x79U, 0x1aU, 0x09U, 0x2aU, 0x07U, 0x0aU, 0x01U, 0x04U,
    0x10U, 0xa0U, 0x80U, 0x04U, 0xa2U, 0x1fU, 0x1aU, 0x0aU,
    0x14U, 0x0aU, 0x06U, 0x64U, 0x6fU, 0x75U, 0x62U, 0x6cU,
    0x65U, 0x12U, 0x01U, 0x78U, 0x12U, 0x01U, 0x78U, 0x1aU,
    0x01U, 0x79U, 0xb2U, 0x0eU, 0x00U, 0x28U, 0x01U, 0x30U,
    0x01U,
};

static bool m3_graph_coreml_build(m3_graph **graph,
                                  m3_graph_value_id *input,
                                  m3_graph_value_id *output,
                                  const char *path, m3_error *error)
{
    const uint64_t shape[] = {4U};
    const float bias_data[] = {1.0F, 1.0F, 1.0F, 1.0F};
    const char *input_names[] = {"x"};
    const char *output_names[] = {"y"};
    m3_graph_value_desc value;
    m3_coreml_partition_desc partition;
    m3_graph_node_desc add;
    m3_graph_value_id coreml_output = M3_GRAPH_VALUE_NONE;
    m3_graph_value_id bias = M3_GRAPH_VALUE_NONE;
    bool ready;

    m3_graph_value_desc_init(&value, M3_GRAPH_INPUT, M3_DTYPE_F32, 1U,
                             shape);
    ready = m3_graph_create(graph, error) == M3_STATUS_OK &&
        m3_graph_add_value(*graph, &value, input, error) == M3_STATUS_OK;
    value.role = M3_GRAPH_TEMPORARY;
    ready = ready &&
        m3_graph_add_value(*graph, &value, &coreml_output, error) ==
            M3_STATUS_OK;
    value.role = M3_GRAPH_CONSTANT;
    ready = ready &&
        m3_graph_add_value(*graph, &value, &bias, error) == M3_STATUS_OK &&
        m3_graph_set_constant(*graph, bias, bias_data, sizeof(bias_data),
                              error) == M3_STATUS_OK;
    value.role = M3_GRAPH_OUTPUT;
    ready = ready &&
        m3_graph_add_value(*graph, &value, output, error) == M3_STATUS_OK;
    partition.compiled_model_path = path;
    partition.inputs = input;
    partition.input_names = input_names;
    partition.input_count = 1U;
    partition.outputs = &coreml_output;
    partition.output_names = output_names;
    partition.output_count = 1U;
    ready = ready && m3_graph_add_coreml_partition(
        *graph, &partition, error) == M3_STATUS_OK;
    m3_graph_node_desc_init(&add, M3_OP_ADD);
    add.inputs[0] = coreml_output;
    add.inputs[1] = bias;
    add.outputs[0] = *output;
    ready = ready && m3_graph_add_node(*graph, &add, error) == M3_STATUS_OK;
    return ready;
}

void m3_test_graph_coreml_provider(m3_test_context *test)
{
    const m3_test_fixture model = {
        "tiny Core ML double model",
        m3_graph_coreml_model,
        sizeof(m3_graph_coreml_model),
    };
    const float input_data[] = {1.0F, -2.5F, 0.0F, 7.25F};
    float output_data[] = {0.0F, 0.0F, 0.0F, 0.0F};
    m3_test_temp_file file;
    m3_graph *graph = NULL;
    m3_session *session = NULL;
    m3_session_options options;
    m3_graph_value_id input;
    m3_graph_value_id output;
    m3_provider_kind provider = M3_PROVIDER_CPU;
    m3_provider_kind second_provider = M3_PROVIDER_CPU;
    m3_error error;
    bool ready = m3_test_temp_file_create(&file, &model);
    size_t index;

    M3_TEST_EXPECT(test, ready, "create tiny Core ML model fixture");
    if (!ready) {
        return;
    }
    ready = m3_graph_coreml_build(&graph, &input, &output, file.path,
                                  &error);
    M3_TEST_EXPECT(test, ready, "build Core ML graph partition");
    if (!ready) {
        (void)m3_test_temp_file_remove(&file);
        m3_graph_free(graph);
        return;
    }
    m3_session_options_init(&options);
    options.allow_neural_engine = false;
    M3_TEST_EXPECT(
        test,
        m3_session_create(&session, graph, &options, NULL) ==
                M3_STATUS_UNSUPPORTED &&
            session == NULL,
        "Core ML partition requires the Neural Engine provider");
    options.allow_neural_engine = true;
    M3_TEST_EXPECT(
        test,
        m3_session_create(&session, graph, &options, &error) == M3_STATUS_OK &&
            m3_session_get_node_provider(session, 0U, &provider, &error) ==
                M3_STATUS_OK &&
            m3_session_get_node_provider(session, 1U, &second_provider,
                                         &error) == M3_STATUS_OK &&
            provider == M3_PROVIDER_NEURAL_ENGINE &&
            second_provider == M3_PROVIDER_METAL,
        "heterogeneous graph selects Neural Engine before Metal");
    m3_graph_free(graph);
    (void)m3_test_temp_file_remove(&file);
    if (session == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_session_write_input(session, input, input_data,
                               sizeof(input_data), &error) == M3_STATUS_OK &&
            m3_session_run(session, &error) == M3_STATUS_OK &&
            m3_session_read_output(session, output, output_data,
                                   sizeof(output_data), &error) == M3_STATUS_OK,
        "Core ML partition executes after graph and model source disposal");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT_F32(test, output_data[index],
                           2.0F * input_data[index] + 1.0F,
                           0.0F, 0.0F,
                           "Core ML to Metal handoff preserves exact output");
    }
    m3_session_free(session);
}
