/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_sampling_test.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static bool m3_test_sampling_top_k_nan(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 3U};
    const uint64_t output_shape[] = {1U, 2U};
    const float logits[] = {2.0F, NAN, 1.0F};
    const float value_sentinel[] = {-9.0F, -9.0F};
    const int32_t index_sentinel[] = {-9, -9};
    m3_metal_sampling_view logits_view;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_command command;
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
                  index_sentinel);
    M3_TEST_EXPECT(test, created, "create NaN Metal top-k tensors");
    if (!created) {
        return false;
    }
    command.kind = M3_OP_TOP_K;
    command.descriptor.top_k.logits = &logits_view.metal;
    command.descriptor.top_k.values = &values.metal;
    command.descriptor.top_k.indices = &indices.metal;
    command.descriptor.top_k.k = 2U;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture->metal, &command, 1U, NULL, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            memcmp(m3_op_test_f32(&values.metal), value_sentinel,
                   sizeof(value_sentinel)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), index_sentinel,
                   sizeof(index_sentinel)) == 0,
        "shared top-k NaN preflight prevents both Metal writes");
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture->metal, &command, 1U, NULL, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            memcmp(m3_op_test_f32(&values.metal), value_sentinel,
                   sizeof(value_sentinel)) == 0,
        "top-k NaN preflight accepts a null error sink");
    command.descriptor.top_k.k = 0U;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture->metal, &command, 1U, NULL, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            memcmp(m3_op_test_i32(&indices.metal), index_sentinel,
                   sizeof(index_sentinel)) == 0,
        "invalid Metal top-k arguments accept a null error sink");
    return true;
}

static bool m3_test_sampling_categorical_errors(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t probability_shape[] = {1U, 2U};
    const uint64_t row_shape[] = {1U};
    const float valid_probabilities[] = {0.25F, 0.75F};
    const float negative_probabilities[] = {-0.25F, 1.25F};
    const float nan_probabilities[] = {NAN, 0.5F};
    const float zero_probabilities[] = {0.0F, 0.0F};
    const float overflow_probabilities[] = {FLT_MAX, FLT_MAX};
    const float valid_uniform[] = {0.5F};
    const float invalid_uniform[] = {1.0F};
    const float nan_uniform[] = {NAN};
    const int32_t sentinel[] = {-8};
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_command command;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &probabilities, M3_DTYPE_F32, 2U,
                  probability_shape, negative_probabilities) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_F32, 1U, row_shape,
                  valid_uniform) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create invalid Metal categorical tensors");
    if (!created) {
        return false;
    }
    command.kind = M3_OP_CATEGORICAL;
    command.descriptor.categorical.probabilities = &probabilities.metal;
    command.descriptor.categorical.uniforms = &uniforms.metal;
    command.descriptor.categorical.output = &output.metal;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture->metal, &command, 1U, NULL, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_op_test_i32(&output.metal)[0] == sentinel[0],
        "shared categorical preflight rejects a negative probability");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(probabilities.metal.storage, 0U, nan_probabilities,
                         sizeof(nan_probabilities), &error) ==
                M3_STATUS_OK &&
            m3_op_test_execute(&fixture->metal, &command, 1U, NULL, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_op_test_i32(&output.metal)[0] == sentinel[0],
        "shared categorical preflight rejects a NaN probability");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(probabilities.metal.storage, 0U, zero_probabilities,
                         sizeof(zero_probabilities), &error) ==
                M3_STATUS_OK &&
            m3_op_test_execute(&fixture->metal, &command, 1U, NULL, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_op_test_i32(&output.metal)[0] == sentinel[0],
        "shared categorical preflight rejects an all-zero row");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(probabilities.metal.storage, 0U,
                         overflow_probabilities,
                         sizeof(overflow_probabilities), &error) ==
                M3_STATUS_OK &&
            m3_op_test_execute(&fixture->metal, &command, 1U, NULL, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_op_test_i32(&output.metal)[0] == sentinel[0],
        "shared categorical preflight rejects a non-finite row sum");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(probabilities.metal.storage, 0U,
                         valid_probabilities, sizeof(valid_probabilities),
                         &error) == M3_STATUS_OK &&
            m3_storage_write(uniforms.metal.storage, 0U, invalid_uniform,
                             sizeof(invalid_uniform), &error) ==
                M3_STATUS_OK &&
            m3_op_test_execute(&fixture->metal, &command, 1U, NULL, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_op_test_i32(&output.metal)[0] == sentinel[0],
        "shared categorical preflight rejects uniform one");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(uniforms.metal.storage, 0U, nan_uniform,
                         sizeof(nan_uniform), &error) == M3_STATUS_OK &&
            m3_op_test_execute(&fixture->metal, &command, 1U, NULL, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_op_test_i32(&output.metal)[0] == sentinel[0],
        "shared categorical preflight rejects a NaN uniform");
    return true;
}

static bool m3_test_sampling_atomic_order(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t top_input_shape[] = {1U, 3U};
    const uint64_t top_output_shape[] = {1U, 2U};
    const uint64_t probability_shape[] = {1U, 2U};
    const uint64_t row_shape[] = {1U};
    const float logits[] = {3.0F, 2.0F, 1.0F};
    const float value_sentinel[] = {-7.0F, -7.0F};
    const int32_t index_sentinel[] = {-7, -7};
    const float invalid_probabilities[] = {0.5F, -0.5F};
    const float uniform[] = {0.0F};
    const int32_t categorical_sentinel[] = {-7};
    m3_metal_sampling_view logits_view;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_command top_k;
    m3_command categorical;
    m3_command commands[2];
    m3_error error;
    m3_status status;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits_view, M3_DTYPE_F32, 2U, top_input_shape,
                  logits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 2U, top_output_shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 2U, top_output_shape,
                  index_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &probabilities, M3_DTYPE_F32, 2U,
                  probability_shape, invalid_probabilities) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_F32, 1U, row_shape,
                  uniform) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape,
                  categorical_sentinel);
    M3_TEST_EXPECT(test, created,
                   "create atomic Metal sampling list tensors");
    if (!created) {
        return false;
    }
    top_k.kind = M3_OP_TOP_K;
    top_k.descriptor.top_k.logits = &logits_view.metal;
    top_k.descriptor.top_k.values = &values.metal;
    top_k.descriptor.top_k.indices = &indices.metal;
    top_k.descriptor.top_k.k = 2U;
    categorical.kind = M3_OP_CATEGORICAL;
    categorical.descriptor.categorical.probabilities = &probabilities.metal;
    categorical.descriptor.categorical.uniforms = &uniforms.metal;
    categorical.descriptor.categorical.output = &output.metal;
    commands[0] = top_k;
    commands[1] = categorical;
    status = m3_op_test_execute(
        &fixture->metal, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OUT_OF_RANGE &&
            memcmp(m3_op_test_f32(&values.metal), value_sentinel,
                   sizeof(value_sentinel)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), index_sentinel,
                   sizeof(index_sentinel)) == 0 &&
            m3_op_test_i32(&output.metal)[0] == categorical_sentinel[0],
        "later categorical failure prevents an earlier top-k commit");
    commands[0] = categorical;
    commands[1] = top_k;
    status = m3_op_test_execute(
        &fixture->metal, commands, 2U, NULL, NULL);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OUT_OF_RANGE &&
            memcmp(m3_op_test_f32(&values.metal), value_sentinel,
                   sizeof(value_sentinel)) == 0 &&
            m3_op_test_i32(&output.metal)[0] == categorical_sentinel[0],
        "earlier categorical failure prevents later top-k encoding");
    return true;
}

void m3_test_metal_sampling_preflight(m3_test_context *test)
{
    m3_metal_sampling_fixture fixture;

    if (!m3_metal_sampling_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_sampling_top_k_nan(test, &fixture) ||
        !m3_test_sampling_categorical_errors(test, &fixture) ||
        !m3_test_sampling_atomic_order(test, &fixture)) {
        m3_metal_sampling_fixture_dispose(&fixture);
        return;
    }
    m3_metal_sampling_fixture_dispose(&fixture);
}
