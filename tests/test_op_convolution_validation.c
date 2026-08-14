/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stdint.h>
#include <string.h>

void m3_test_op_convolution_validation(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 2U, 3U};
    const uint64_t weight_shape[] = {2U, 1U, 2U};
    const uint64_t wrong_weight_shape[] = {2U, 2U, 2U};
    const uint64_t bias_shape[] = {2U};
    const uint64_t output_shape[] = {1U, 2U, 2U};
    const uint64_t wrong_output_shape[] = {1U, 2U, 3U};
    const float input_values[] = {1, 2, 3, 4, 5, 6};
    const float weight_values[] = {1, 1, 1, 1};
    const float wrong_weight_values[8] = {0};
    const float bias_values[] = {0, 0};
    const float output_sentinel[] = {41, 42, 43, 44};
    const float wrong_output_zeros[6] = {0};
    const uint64_t alias_shape[] = {1U, 1U, 3U};
    const uint64_t alias_weight_shape[] = {1U, 1U, 1U};
    const float alias_values[] = {1, 2, 3};
    const float alias_weight_value[] = {1};
    const uint64_t empty_input_shape[] = {0U, 2U, 3U};
    const uint64_t empty_output_shape[] = {0U, 2U, 2U};
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view weight;
    m3_tensor_view wrong_weight;
    m3_tensor_view bias;
    m3_tensor_view output;
    m3_tensor_view wrong_output;
    m3_tensor_view alias_input;
    m3_tensor_view alias_weight;
    m3_tensor_view empty_input;
    m3_tensor_view empty_output;
    m3_command command = {0};
    m3_command commands[2];
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create convolution validation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 3U,
                          input_shape, input_values) &&
            m3_op_test_tensor(&fixture, &weight, M3_DTYPE_F32, 3U,
                              weight_shape, weight_values) &&
            m3_op_test_tensor(&fixture, &wrong_weight, M3_DTYPE_F32, 3U,
                              wrong_weight_shape, wrong_weight_values) &&
            m3_op_test_tensor(&fixture, &bias, M3_DTYPE_F32, 1U,
                              bias_shape, bias_values) &&
            m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 3U,
                              output_shape, output_sentinel) &&
            m3_op_test_tensor(&fixture, &wrong_output, M3_DTYPE_F32, 3U,
                              wrong_output_shape, wrong_output_zeros),
        "create canonical Conv1d validation tensors");
    command.kind = M3_OP_CONV1D;
    command.descriptor.conv1d.input = &input;
    command.descriptor.conv1d.weight = &weight;
    command.descriptor.conv1d.bias = &bias;
    command.descriptor.conv1d.output = &output;
    command.descriptor.conv1d.groups = 2U;
    command.descriptor.conv1d.stride = 1U;
    command.descriptor.conv1d.dilation = 1U;
    command.descriptor.conv1d.pad_left = 0U;
    command.descriptor.conv1d.pad_right = 0U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_OK,
                   "accept canonical grouped Conv1d contract");

    command.descriptor.conv1d.groups = 0U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject zero Conv1d groups");
    command.descriptor.conv1d.groups = 3U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject nondivisible Conv1d groups");
    command.descriptor.conv1d.groups = 2U;
    command.descriptor.conv1d.weight = &wrong_weight;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject incompatible Conv1d weight shape");
    command.descriptor.conv1d.weight = &weight;
    command.descriptor.conv1d.dilation = 4U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject Conv1d effective kernel larger than input");
    command.descriptor.conv1d.dilation = UINT64_MAX;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_OVERFLOW,
                   "reject overflowing Conv1d effective kernel");
    command.descriptor.conv1d.dilation = 1U;
    command.descriptor.conv1d.pad_left = UINT64_MAX;
    command.descriptor.conv1d.pad_right = 1U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_OVERFLOW,
                   "reject overflowing Conv1d padded length");
    command.descriptor.conv1d.pad_left = 0U;
    command.descriptor.conv1d.pad_right = 0U;
    command.descriptor.conv1d.output = &wrong_output;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject incorrect Conv1d output shape");
    command.descriptor.conv1d.output = &output;

    commands[0] = command;
    commands[1] = command;
    commands[1].descriptor.conv1d.groups = 0U;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, commands, 2U, NULL, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "prevalidate all convolution commands atomically");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output)[index],
                           output_sentinel[index], 0.0F, 0.0F,
                           "structural failure leaves output unchanged");
    }

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &alias_input, M3_DTYPE_F32, 3U,
                          alias_shape, alias_values) &&
            m3_op_test_tensor(&fixture, &alias_weight, M3_DTYPE_F32, 3U,
                              alias_weight_shape, alias_weight_value),
        "create Conv1d alias tensors");
    command.descriptor.conv1d.input = &alias_input;
    command.descriptor.conv1d.weight = &alias_weight;
    command.descriptor.conv1d.bias = NULL;
    command.descriptor.conv1d.output = &alias_input;
    command.descriptor.conv1d.groups = 1U;
    command.descriptor.conv1d.stride = 1U;
    command.descriptor.conv1d.dilation = 1U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject in-place Conv1d output overlap");

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &empty_input, M3_DTYPE_F32, 3U,
                          empty_input_shape, NULL) &&
            m3_op_test_tensor(&fixture, &empty_output, M3_DTYPE_F32, 3U,
                              empty_output_shape, NULL),
        "create empty-batch Conv1d tensors");
    command.descriptor.conv1d.input = &empty_input;
    command.descriptor.conv1d.weight = &weight;
    command.descriptor.conv1d.bias = &bias;
    command.descriptor.conv1d.output = &empty_output;
    command.descriptor.conv1d.groups = 2U;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "allow empty Conv1d batch with nonzero length");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_conv_transpose_validation(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 2U};
    const uint64_t output_shape[] = {1U, 1U, 5U};
    const uint64_t wrong_output_shape[] = {1U, 1U, 4U};
    const uint64_t empty_input_shape[] = {0U, 1U, 2U};
    const uint64_t empty_output_shape[] = {0U, 1U, 5U};
    const float input_values[] = {1, 2};
    const float weight_values[] = {1, 1};
    const float output_zeros[5] = {0};
    const float wrong_output_zeros[4] = {0};
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view weight;
    m3_tensor_view output;
    m3_tensor_view wrong_output;
    m3_tensor_view empty_input;
    m3_tensor_view empty_output;
    m3_command command = {0};
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create ConvTranspose1d validation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 3U,
                          input_shape, input_values) &&
            m3_op_test_tensor(&fixture, &weight, M3_DTYPE_F32, 3U,
                              weight_shape, weight_values) &&
            m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 3U,
                              output_shape, output_zeros) &&
            m3_op_test_tensor(&fixture, &wrong_output, M3_DTYPE_F32, 3U,
                              wrong_output_shape, wrong_output_zeros),
        "create ConvTranspose1d validation tensors");
    command.kind = M3_OP_CONV_TRANSPOSE1D;
    command.descriptor.conv_transpose1d.input = &input;
    command.descriptor.conv_transpose1d.weight = &weight;
    command.descriptor.conv_transpose1d.bias = NULL;
    command.descriptor.conv_transpose1d.output = &output;
    command.descriptor.conv_transpose1d.groups = 1U;
    command.descriptor.conv_transpose1d.stride = 2U;
    command.descriptor.conv_transpose1d.dilation = 1U;
    command.descriptor.conv_transpose1d.pad_left = 0U;
    command.descriptor.conv_transpose1d.pad_right = 0U;
    command.descriptor.conv_transpose1d.output_padding = 1U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_OK,
                   "accept canonical ConvTranspose1d contract");
    command.descriptor.conv_transpose1d.groups = 2U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject nondivisible ConvTranspose1d groups");
    command.descriptor.conv_transpose1d.groups = 1U;
    command.descriptor.conv_transpose1d.output_padding = 2U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject ConvTranspose1d output padding at stride");
    command.descriptor.conv_transpose1d.output_padding = 1U;
    command.descriptor.conv_transpose1d.output = &wrong_output;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject incorrect ConvTranspose1d output shape");
    command.descriptor.conv_transpose1d.output = &output;
    command.descriptor.conv_transpose1d.stride = UINT64_MAX;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_OVERFLOW,
                   "reject overflowing ConvTranspose1d output length");
    command.descriptor.conv_transpose1d.stride = 2U;

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &empty_input, M3_DTYPE_F32, 3U,
                          empty_input_shape, NULL) &&
            m3_op_test_tensor(&fixture, &empty_output, M3_DTYPE_F32, 3U,
                              empty_output_shape, NULL),
        "create empty-batch ConvTranspose1d tensors");
    command.descriptor.conv_transpose1d.input = &empty_input;
    command.descriptor.conv_transpose1d.output = &empty_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "allow empty ConvTranspose1d batch with nonzero length");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_convolution_metal_unsupported(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 1U};
    const float input_values[] = {1, 2};
    const float weight_value[] = {3};
    const float sentinel[] = {-7, -8};
    m3_backend *backend = NULL;
    m3_storage *input_storage = NULL;
    m3_storage *weight_storage = NULL;
    m3_storage *output_storage = NULL;
    m3_tensor_view input;
    m3_tensor_view weight;
    m3_tensor_view output;
    m3_command command = {0};
    m3_error error;
    m3_status status = m3_backend_create_metal(&backend, &error);

    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, "Metal has no default device");
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal backend for convolution rejection");
    if (backend == NULL) {
        return;
    }
    m3_tensor_view_init(&input);
    m3_tensor_view_init(&weight);
    m3_tensor_view_init(&output);
    M3_TEST_EXPECT(
        test,
        m3_storage_allocate(backend, sizeof(input_values), 16U,
                            &input_storage, &error) == M3_STATUS_OK &&
            m3_storage_allocate(backend, sizeof(weight_value), 16U,
                                &weight_storage, &error) == M3_STATUS_OK &&
            m3_storage_allocate(backend, sizeof(sentinel), 16U,
                                &output_storage, &error) == M3_STATUS_OK &&
            m3_storage_write(input_storage, 0U, input_values,
                             sizeof(input_values), &error) == M3_STATUS_OK &&
            m3_storage_write(weight_storage, 0U, weight_value,
                             sizeof(weight_value), &error) == M3_STATUS_OK &&
            m3_storage_write(output_storage, 0U, sentinel,
                             sizeof(sentinel), &error) == M3_STATUS_OK &&
            m3_tensor_view_contiguous(&input, input_storage, M3_DTYPE_F32,
                                      3U, input_shape, 0U, &error) ==
                M3_STATUS_OK &&
            m3_tensor_view_contiguous(&weight, weight_storage, M3_DTYPE_F32,
                                      3U, weight_shape, 0U, &error) ==
                M3_STATUS_OK &&
            m3_tensor_view_contiguous(&output, output_storage, M3_DTYPE_F32,
                                      3U, input_shape, 0U, &error) ==
                M3_STATUS_OK,
        "create Metal convolution tensors");
    command.kind = M3_OP_CONV1D;
    command.descriptor.conv1d.input = &input;
    command.descriptor.conv1d.weight = &weight;
    command.descriptor.conv1d.bias = NULL;
    command.descriptor.conv1d.output = &output;
    command.descriptor.conv1d.groups = 0U;
    command.descriptor.conv1d.stride = 1U;
    command.descriptor.conv1d.dilation = 1U;
    command.descriptor.conv1d.pad_left = 0U;
    command.descriptor.conv1d.pad_right = 0U;
    M3_TEST_EXPECT(
        test,
        m3_backend_execute(backend, &command, 1U, NULL, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            memcmp(m3_storage_const_data(output_storage), sentinel,
                   sizeof(sentinel)) == 0,
        "Metal prevalidates convolution without mutation");
    command.descriptor.conv1d.groups = 1U;
    M3_TEST_EXPECT(
        test,
        m3_backend_execute(backend, &command, 1U, NULL, &error) ==
                M3_STATUS_UNSUPPORTED &&
            memcmp(m3_storage_const_data(output_storage), sentinel,
                   sizeof(sentinel)) == 0,
        "Metal rejects convolution without CPU fallback or mutation");
    m3_storage_free(output_storage);
    m3_storage_free(weight_storage);
    m3_storage_free(input_storage);
    m3_backend_free(backend);
}
