/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_metal_helpers.h"
#include "m3_op_test.h"

#include <stdint.h>
#include <string.h>

static bool m3_test_metal_view_tail_is_zero(
    const m3_metal_view_parameters *parameters)
{
    size_t axis;

    for (axis = parameters->rank; axis < M3_TENSOR_MAX_RANK; ++axis) {
        if (parameters->shape[axis] != 0U ||
            parameters->byte_strides[axis] != 0U) {
            return false;
        }
    }
    return true;
}

static bool m3_test_metal_fixture_init(m3_test_context *test,
                                       m3_op_test_fixture *fixture)
{
    m3_error error;
    m3_status status;

    (void)memset(fixture, 0, sizeof(*fixture));
    status = m3_backend_create_metal(&fixture->backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal backend with command pipelines");
    return status == M3_STATUS_OK;
}

static bool m3_test_metal_strided_copy(m3_test_context *test,
                                       m3_op_test_fixture *fixture)
{
    const uint64_t shape[] = {3U};
    const size_t strides[] = {2U * sizeof(float)};
    const float padded[] = {
        90.0F, 1.25F, 91.0F, -2.5F, 92.0F, 3.75F, 93.0F
    };
    const float zeros[] = {0.0F, 0.0F, 0.0F};
    m3_storage *input_storage = NULL;
    m3_metal_view_parameters parameters;
    m3_tensor_view input;
    m3_tensor_view output;
    m3_command command;
    m3_error error;
    float *result;

    m3_tensor_view_init(&input);
    m3_tensor_view_init(&output);
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(fixture, sizeof(padded),
                                      &input_storage) &&
                       m3_storage_write(input_storage, 0U, padded,
                                        sizeof(padded), &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_strided(
                           &input, input_storage, M3_DTYPE_F32, 1U, shape,
                           strides, sizeof(float), &error) == M3_STATUS_OK &&
                       m3_op_test_tensor(fixture, &output, M3_DTYPE_F32, 1U,
                                         shape, zeros),
                   "create strided Metal COPY tensors");
    if (input.storage == NULL || output.storage == NULL) {
        return false;
    }
    m3_metal_view_parameters_init(&parameters, &input);
    M3_TEST_EXPECT(
        test,
        parameters.element_count == 3U &&
            parameters.byte_offset == sizeof(float) &&
            parameters.shape[0] == 3U &&
            parameters.byte_strides[0] == 2U * sizeof(float) &&
            parameters.rank == 1U && parameters.dtype == M3_DTYPE_F32 &&
            m3_test_metal_view_tail_is_zero(&parameters),
        "initialize the shared Metal view parameter ABI");
    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = &input;
    command.descriptor.copy.output = &output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "execute strided Metal COPY");
    result = m3_op_test_f32(&output);
    M3_TEST_EXPECT(test,
                   result[0] == 1.25F && result[1] == -2.5F &&
                       result[2] == 3.75F,
                   "Metal COPY gathers a strided input in logical order");
    command.descriptor.copy.input = &output;
    command.descriptor.copy.output = &output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       result[0] == 1.25F && result[1] == -2.5F &&
                       result[2] == 3.75F,
                   "Metal COPY preserves an exact alias");
    return true;
}

static bool m3_test_metal_multi_command(m3_test_context *test,
                                        m3_op_test_fixture *fixture)
{
    const uint64_t shape[] = {2U};
    const float source_values[] = {1.5F, -2.25F};
    const float zeros32[] = {0.0F, 0.0F};
    const uint16_t zeros16[] = {0U, 0U};
    m3_tensor_view source;
    m3_tensor_view middle;
    m3_tensor_view output;
    m3_command commands[2];
    m3_error error;
    uint16_t *result;

    m3_tensor_view_init(&source);
    m3_tensor_view_init(&middle);
    m3_tensor_view_init(&output);
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(fixture, &source, M3_DTYPE_F32, 1U,
                                     shape, source_values) &&
                       m3_op_test_tensor(fixture, &middle, M3_DTYPE_F32, 1U,
                                         shape, zeros32) &&
                       m3_op_test_tensor(fixture, &output, M3_DTYPE_BF16, 1U,
                                         shape, zeros16),
                   "create dependent Metal command tensors");
    if (source.storage == NULL || middle.storage == NULL ||
        output.storage == NULL) {
        return false;
    }
    commands[0].kind = M3_OP_COPY;
    commands[0].descriptor.copy.input = &source;
    commands[0].descriptor.copy.output = &middle;
    commands[1].kind = M3_OP_CAST;
    commands[1].descriptor.cast.input = &middle;
    commands[1].descriptor.cast.output = &output;
    M3_TEST_EXPECT(
        test,
        !m3_metal_has_prior_writer(commands, 0U, middle.storage) &&
            m3_metal_has_prior_writer(commands, 1U, middle.storage) &&
            !m3_metal_has_prior_writer(commands, 1U, output.storage) &&
            m3_metal_has_prior_writer(commands, 2U, output.storage) &&
            !m3_metal_has_prior_writer(commands, 2U, NULL),
        "scan prior Metal writers through the family registry");
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, commands, 2U, NULL,
                                      &error) == M3_STATUS_OK,
                   "commit a dependent Metal command list once");
    result = m3_op_test_u16(&output);
    M3_TEST_EXPECT(test,
                   result[0] == 0x3fc0U && result[1] == 0xc010U,
                   "Metal commands observe prior encodes in list order");
    return true;
}

static bool m3_test_metal_unsupported_atomic(
    m3_test_context *test, m3_op_test_fixture *fixture)
{
    const uint64_t vector_shape[] = {1U};
    const uint64_t matrix_shape[] = {1U, 1U};
    const float source_value[] = {7.0F};
    const float right_value[] = {3.0F};
    const float copy_sentinel[] = {-41.0F};
    const float dense_sentinel[] = {-42.0F};
    m3_tensor_view source;
    m3_tensor_view copy_output;
    m3_tensor_view left;
    m3_tensor_view right;
    m3_tensor_view dense_output;
    m3_command commands[2];
    m3_error error;
    m3_status status;

    m3_tensor_view_init(&source);
    m3_tensor_view_init(&copy_output);
    m3_tensor_view_init(&left);
    m3_tensor_view_init(&right);
    m3_tensor_view_init(&dense_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(fixture, &source, M3_DTYPE_F32, 1U,
                                     vector_shape, source_value) &&
                       m3_op_test_tensor(fixture, &copy_output, M3_DTYPE_F32,
                                         1U, vector_shape, copy_sentinel) &&
                       m3_op_test_tensor(fixture, &left, M3_DTYPE_F32, 2U,
                                         matrix_shape, source_value) &&
                       m3_op_test_tensor(fixture, &right, M3_DTYPE_F32, 2U,
                                         matrix_shape, right_value) &&
                       m3_op_test_tensor(fixture, &dense_output,
                                         M3_DTYPE_F32, 2U, matrix_shape,
                                         dense_sentinel),
                   "create unsupported-list Metal tensors");
    if (dense_output.storage == NULL) {
        return false;
    }
    commands[0].kind = M3_OP_COPY;
    commands[0].descriptor.copy.input = &source;
    commands[0].descriptor.copy.output = &copy_output;
    commands[1].kind = M3_OP_MATMUL;
    commands[1].descriptor.matmul.left = &left;
    commands[1].descriptor.matmul.right = &right;
    commands[1].descriptor.matmul.output = &dense_output;
    status = m3_backend_execute(fixture->backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       m3_op_test_f32(&copy_output)[0] == copy_sentinel[0] &&
                       m3_op_test_f32(&dense_output)[0] == dense_sentinel[0],
                   "unsupported dense command prevents the encoded COPY commit");
    status = m3_backend_execute(fixture->backend, commands, 2U, NULL, NULL);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       m3_op_test_f32(&copy_output)[0] == copy_sentinel[0],
                   "unsupported Metal list accepts a null error sink");
    return true;
}

static bool m3_test_metal_strided_output_rejected(
    m3_test_context *test, m3_op_test_fixture *fixture)
{
    const uint64_t shape[] = {3U};
    const size_t strides[] = {2U * sizeof(float)};
    const float source_values[] = {1.0F, 2.0F, 3.0F};
    const uint32_t sentinels[] = {
        0xdeadbeefU, 0xdeadbeefU, 0xdeadbeefU,
        0xdeadbeefU, 0xdeadbeefU, 0xdeadbeefU
    };
    m3_storage *output_storage = NULL;
    m3_tensor_view input;
    m3_tensor_view output;
    m3_command command;
    m3_error error;

    m3_tensor_view_init(&input);
    m3_tensor_view_init(&output);
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(fixture, &input, M3_DTYPE_F32, 1U,
                                     shape, source_values) &&
                       m3_op_test_storage(fixture, sizeof(sentinels),
                                          &output_storage) &&
                       m3_storage_write(output_storage, 0U, sentinels,
                                        sizeof(sentinels), &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_strided(
                           &output, output_storage, M3_DTYPE_F32, 1U, shape,
                           strides, 0U, &error) == M3_STATUS_OK,
                   "create a valid strided Metal output view");
    if (output.storage == NULL) {
        return false;
    }
    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = &input;
    command.descriptor.copy.output = &output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_INVALID_ARGUMENT &&
                       memcmp(m3_storage_const_data(output_storage), sentinels,
                              sizeof(sentinels)) == 0,
                   "strided Metal output is rejected without mutation");
    return true;
}

void m3_test_metal_copy_commands(m3_test_context *test)
{
    m3_op_test_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_error error;

    if (!m3_test_metal_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_strided_copy(test, &fixture) ||
        !m3_test_metal_multi_command(test, &fixture) ||
        !m3_test_metal_unsupported_atomic(test, &fixture) ||
        !m3_test_metal_strided_output_rejected(test, &fixture)) {
        m3_op_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &stats, &error) == M3_STATUS_OK &&
                       stats.live_storage_count == fixture.storage_count &&
                       stats.live_storage_count != 0U &&
                       stats.live_allocated_bytes != 0U,
                   "Metal command storage remains live and accounted");
    m3_backend_free(fixture.backend);
    fixture.backend = NULL;
    fixture.storage_count = 0U;
    M3_TEST_EXPECT(test, true,
                   "Metal backend releases live command storage at teardown");
}
