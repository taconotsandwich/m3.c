/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_metal_helpers.h"
#include "metal_sampling_test.h"

#include <stdint.h>
#include <string.h>

static void m3_test_sampling_top_k_metal(
    m3_command *command, const m3_tensor_view *logits,
    m3_tensor_view *values, m3_tensor_view *indices, uint64_t k)
{
    command->kind = M3_OP_TOP_K;
    command->descriptor.top_k.logits = logits;
    command->descriptor.top_k.values = values;
    command->descriptor.top_k.indices = indices;
    command->descriptor.top_k.k = k;
}

static bool m3_test_sampling_supported_order(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 3U};
    const uint64_t output_shape[] = {1U, 2U};
    const float logits[] = {1.0F, 3.0F, 2.0F};
    const float value_sentinel[] = {-9.0F, -9.0F};
    const int32_t index_sentinel[] = {-9, -9};
    const float copy_value_sentinel[] = {-8.0F, -8.0F};
    const int32_t copy_index_sentinel[] = {-8, -8};
    const float expected_values[] = {3.0F, 2.0F};
    const int32_t expected_indices[] = {1, 2};
    m3_metal_sampling_view logits_view;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_metal_sampling_view copied_values;
    m3_metal_sampling_view copied_indices;
    m3_command commands[3];
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits_view, M3_DTYPE_F32, 2U, input_shape,
                  logits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 2U, output_shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 2U, output_shape,
                  index_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &copied_values, M3_DTYPE_F32, 2U, output_shape,
                  copy_value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &copied_indices, M3_DTYPE_I32, 2U, output_shape,
                  copy_index_sentinel);
    M3_TEST_EXPECT(test, created,
                   "create ordered Metal top-k consumer tensors");
    if (!created) {
        return false;
    }
    m3_test_sampling_top_k_metal(
        &commands[0], &logits_view.metal, &values.metal, &indices.metal, 2U);
    commands[1].kind = M3_OP_COPY;
    commands[1].descriptor.copy.input = &values.metal;
    commands[1].descriptor.copy.output = &copied_values.metal;
    commands[2].kind = M3_OP_COPY;
    commands[2].descriptor.copy.input = &indices.metal;
    commands[2].descriptor.copy.output = &copied_indices.metal;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture->metal, commands, 3U, NULL, &error) ==
                M3_STATUS_OK &&
            memcmp(m3_op_test_f32(&values.metal), expected_values,
                   sizeof(expected_values)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), expected_indices,
                   sizeof(expected_indices)) == 0 &&
            memcmp(m3_op_test_f32(&copied_values.metal), expected_values,
                   sizeof(expected_values)) == 0 &&
            memcmp(m3_op_test_i32(&copied_indices.metal), expected_indices,
                   sizeof(expected_indices)) == 0,
        "both Metal top-k outputs are visible to later list consumers");
    return true;
}

static bool m3_test_sampling_top_k_prior_writer(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 3U};
    const uint64_t first_output_shape[] = {1U, 3U};
    const uint64_t second_output_shape[] = {1U, 1U};
    const float logits[] = {3.0F, 2.0F, 1.0F};
    const float first_value_sentinel[] = {-7.0F, -7.0F, -7.0F};
    const int32_t first_index_sentinel[] = {-7, -7, -7};
    const float second_value_sentinel[] = {-6.0F};
    const int32_t second_index_sentinel[] = {-6};
    m3_metal_sampling_view logits_view;
    m3_metal_sampling_view first_values;
    m3_metal_sampling_view first_indices;
    m3_metal_sampling_view second_values;
    m3_metal_sampling_view second_indices;
    m3_command commands[2];
    m3_error error;
    m3_status status;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits_view, M3_DTYPE_F32, 2U, input_shape,
                  logits) &&
              m3_metal_sampling_tensor(
                  fixture, &first_values, M3_DTYPE_F32, 2U,
                  first_output_shape, first_value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &first_indices, M3_DTYPE_I32, 2U,
                  first_output_shape, first_index_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &second_values, M3_DTYPE_F32, 2U,
                  second_output_shape, second_value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &second_indices, M3_DTYPE_I32, 2U,
                  second_output_shape, second_index_sentinel);
    M3_TEST_EXPECT(test, created,
                   "create dependent Metal top-k tensors");
    if (!created) {
        return false;
    }
    m3_test_sampling_top_k_metal(
        &commands[0], &logits_view.metal, &first_values.metal,
        &first_indices.metal, 3U);
    m3_test_sampling_top_k_metal(
        &commands[1], &first_values.metal, &second_values.metal,
        &second_indices.metal, 1U);
    M3_TEST_EXPECT(
        test,
        m3_metal_command_writes_storage(
            &commands[0], first_values.metal.storage) &&
            m3_metal_command_writes_storage(
                &commands[0], first_indices.metal.storage) &&
            !m3_metal_command_writes_storage(
                &commands[0], logits_view.metal.storage),
        "Metal writer registry exposes both top-k outputs");
    status = m3_op_test_execute(
        &fixture->metal, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_UNSUPPORTED &&
            memcmp(m3_op_test_f32(&first_values.metal),
                   first_value_sentinel, sizeof(first_value_sentinel)) == 0 &&
            memcmp(m3_op_test_i32(&first_indices.metal),
                   first_index_sentinel, sizeof(first_index_sentinel)) == 0 &&
            m3_op_test_f32(&second_values.metal)[0] ==
                second_value_sentinel[0] &&
            m3_op_test_i32(&second_indices.metal)[0] ==
                second_index_sentinel[0],
        "top-k rejects a prior-produced logits storage before commit");
    return true;
}

static bool m3_test_sampling_probability_prior_writer(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t shape[] = {1U, 3U};
    const uint64_t row_shape[] = {1U};
    const float logits[] = {3.0F, 2.0F, 1.0F};
    const float value_sentinel[] = {-5.0F, -5.0F, -5.0F};
    const int32_t index_sentinel[] = {-5, -5, -5};
    const float uniform[] = {0.0F};
    const int32_t categorical_sentinel[] = {-5};
    m3_metal_sampling_view logits_view;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_tensor_view probabilities;
    m3_command commands[2];
    m3_error error;
    m3_status status;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits_view, M3_DTYPE_F32, 2U, shape, logits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 2U, shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 2U, shape,
                  index_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_F32, 1U, row_shape,
                  uniform) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape,
                  categorical_sentinel);
    m3_tensor_view_init(&probabilities);
    created = created &&
              m3_tensor_view_contiguous(
                  &probabilities, indices.metal.storage, M3_DTYPE_F32, 2U,
                  shape, 0U, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, created,
                   "create prior-produced categorical probabilities");
    if (!created) {
        return false;
    }
    m3_test_sampling_top_k_metal(
        &commands[0], &logits_view.metal, &values.metal, &indices.metal, 3U);
    commands[1].kind = M3_OP_CATEGORICAL;
    commands[1].descriptor.categorical.probabilities = &probabilities;
    commands[1].descriptor.categorical.uniforms = &uniforms.metal;
    commands[1].descriptor.categorical.output = &output.metal;
    status = m3_op_test_execute(
        &fixture->metal, commands, 2U, NULL, NULL);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_UNSUPPORTED &&
            memcmp(m3_op_test_f32(&values.metal), value_sentinel,
                   sizeof(value_sentinel)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), index_sentinel,
                   sizeof(index_sentinel)) == 0 &&
            m3_op_test_i32(&output.metal)[0] == categorical_sentinel[0],
        "categorical rejects prior-produced probabilities before scanning");
    return true;
}

static bool m3_test_sampling_uniform_prior_writer(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t one_shape[] = {1U};
    const uint64_t probability_shape[] = {1U, 2U};
    const float logits[] = {0.5F};
    const float value_sentinel[] = {-4.0F};
    const int32_t index_sentinel[] = {-4};
    const float probabilities[] = {0.5F, 0.5F};
    const int32_t categorical_sentinel[] = {-4};
    m3_metal_sampling_view logits_view;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_metal_sampling_view probability_view;
    m3_metal_sampling_view output;
    m3_command commands[2];
    m3_error error;
    m3_status status;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits_view, M3_DTYPE_F32, 1U, one_shape,
                  logits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 1U, one_shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 1U, one_shape,
                  index_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &probability_view, M3_DTYPE_F32, 2U,
                  probability_shape, probabilities) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, one_shape,
                  categorical_sentinel);
    M3_TEST_EXPECT(test, created,
                   "create prior-produced categorical uniform tensors");
    if (!created) {
        return false;
    }
    m3_test_sampling_top_k_metal(
        &commands[0], &logits_view.metal, &values.metal, &indices.metal, 1U);
    commands[1].kind = M3_OP_CATEGORICAL;
    commands[1].descriptor.categorical.probabilities =
        &probability_view.metal;
    commands[1].descriptor.categorical.uniforms = &values.metal;
    commands[1].descriptor.categorical.output = &output.metal;
    M3_TEST_EXPECT(
        test,
        m3_metal_command_writes_storage(
            &commands[1], output.metal.storage) &&
            !m3_metal_command_writes_storage(
                &commands[1], probability_view.metal.storage),
        "Metal writer registry exposes only categorical output storage");
    status = m3_op_test_execute(
        &fixture->metal, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_UNSUPPORTED &&
            m3_op_test_f32(&values.metal)[0] == value_sentinel[0] &&
            m3_op_test_i32(&indices.metal)[0] == index_sentinel[0] &&
            m3_op_test_i32(&output.metal)[0] == categorical_sentinel[0],
        "categorical rejects a prior-produced uniform before scanning");
    return true;
}

void m3_test_metal_sampling_dependencies(m3_test_context *test)
{
    m3_metal_sampling_fixture fixture;

    if (!m3_metal_sampling_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_sampling_supported_order(test, &fixture) ||
        !m3_test_sampling_top_k_prior_writer(test, &fixture) ||
        !m3_test_sampling_probability_prior_writer(test, &fixture) ||
        !m3_test_sampling_uniform_prior_writer(test, &fixture)) {
        m3_metal_sampling_fixture_dispose(&fixture);
        return;
    }
    m3_metal_sampling_fixture_dispose(&fixture);
}
