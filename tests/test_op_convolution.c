/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stddef.h>

static m3_command m3_test_conv1d_command(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    const m3_tensor_view *bias, m3_tensor_view *output, uint64_t groups,
    uint64_t stride, uint64_t dilation, uint64_t pad_left,
    uint64_t pad_right)
{
    m3_command command = {0};

    command.kind = M3_OP_CONV1D;
    command.descriptor.conv1d.input = input;
    command.descriptor.conv1d.weight = weight;
    command.descriptor.conv1d.bias = bias;
    command.descriptor.conv1d.output = output;
    command.descriptor.conv1d.groups = groups;
    command.descriptor.conv1d.stride = stride;
    command.descriptor.conv1d.dilation = dilation;
    command.descriptor.conv1d.pad_left = pad_left;
    command.descriptor.conv1d.pad_right = pad_right;
    return command;
}

static m3_command m3_test_conv_transpose1d_command(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    const m3_tensor_view *bias, m3_tensor_view *output, uint64_t groups,
    uint64_t stride, uint64_t dilation, uint64_t pad_left,
    uint64_t pad_right, uint64_t output_padding)
{
    m3_command command = {0};

    command.kind = M3_OP_CONV_TRANSPOSE1D;
    command.descriptor.conv_transpose1d.input = input;
    command.descriptor.conv_transpose1d.weight = weight;
    command.descriptor.conv_transpose1d.bias = bias;
    command.descriptor.conv_transpose1d.output = output;
    command.descriptor.conv_transpose1d.groups = groups;
    command.descriptor.conv_transpose1d.stride = stride;
    command.descriptor.conv_transpose1d.dilation = dilation;
    command.descriptor.conv_transpose1d.pad_left = pad_left;
    command.descriptor.conv_transpose1d.pad_right = pad_right;
    command.descriptor.conv_transpose1d.output_padding = output_padding;
    return command;
}

void m3_test_op_conv1d(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 2U, 4U};
    const uint64_t weight_shape[] = {2U, 1U, 3U};
    const uint64_t bias_shape[] = {2U};
    const uint64_t output_shape[] = {1U, 2U, 3U};
    const size_t input_strides[] = {32U, 16U, 4U};
    const uint16_t input_backing[] = {
        0x3c00U, 0U, 0x4000U, 0U, 0x4200U, 0U, 0x4400U, 0U,
        0x4500U, 0U, 0x4600U, 0U, 0x4700U, 0U, 0x4800U, 0U
    };
    const uint16_t weight_values[] = {
        0x3f80U, 0U, 0xbf80U, 0x3f00U, 0x3f80U, 0x3f00U
    };
    const float bias_values[] = {10.0F, -1.0F};
    const uint16_t output_zeros[6] = {0};
    const uint16_t output_expected[] = {
        0x4100U, 0x4100U, 0x4160U, 0x40e0U, 0x4150U, 0x4040U
    };
    const uint64_t dilation_input_shape[] = {1U, 1U, 5U};
    const uint64_t dilation_weight_shape[] = {1U, 1U, 2U};
    const uint64_t dilation_output_shape[] = {1U, 1U, 3U};
    const float dilation_input_values[] = {1, 2, 3, 4, 5};
    const float dilation_weight_values[] = {2, -1};
    const float dilation_zeros[3] = {0};
    const float dilation_expected[] = {-1, 0, 1};
    const uint64_t rounding_input_shape[] = {1U, 1U, 2U};
    const uint64_t rounding_weight_shape[] = {1U, 1U, 2U};
    const uint64_t rounding_bias_shape[] = {1U};
    const uint64_t rounding_output_shape[] = {1U, 1U, 1U};
    const float rounding_input_values[] = {1, 1};
    const float rounding_weight_values[] = {-16777216.0F, 1.0F};
    const float rounding_bias_value[] = {16777216.0F};
    const float rounding_zero[] = {0};
    m3_op_test_fixture fixture;
    m3_storage *input_storage = NULL;
    m3_tensor_view input;
    m3_tensor_view weight;
    m3_tensor_view bias;
    m3_tensor_view output;
    m3_tensor_view dilation_input;
    m3_tensor_view dilation_weight;
    m3_tensor_view dilation_output;
    m3_tensor_view rounding_input;
    m3_tensor_view rounding_weight;
    m3_tensor_view rounding_bias;
    m3_tensor_view rounding_output;
    m3_command command;
    m3_error error;
    size_t scratch_bytes = SIZE_MAX;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create Conv1d fixture");
    if (fixture.backend == NULL) {
        return;
    }
    m3_tensor_view_init(&input);
    M3_TEST_EXPECT(
        test,
        m3_op_test_storage(&fixture, sizeof(input_backing),
                           &input_storage) &&
            m3_storage_write(input_storage, 0U, input_backing,
                             sizeof(input_backing), &error) ==
                M3_STATUS_OK &&
            m3_tensor_view_strided(&input, input_storage, M3_DTYPE_F16, 3U,
                                   input_shape, input_strides, 0U, &error) ==
                M3_STATUS_OK &&
            m3_op_test_tensor(&fixture, &weight, M3_DTYPE_BF16, 3U,
                              weight_shape, weight_values) &&
            m3_op_test_tensor(&fixture, &bias, M3_DTYPE_F32, 1U,
                              bias_shape, bias_values) &&
            m3_op_test_tensor(&fixture, &output, M3_DTYPE_BF16, 3U,
                              output_shape, output_zeros),
        "create grouped mixed-dtype Conv1d tensors");
    command = m3_test_conv1d_command(&input, &weight, &bias, &output, 2U,
                                     2U, 1U, 1U, 2U);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U,
                                      &scratch_bytes, &error) ==
                           M3_STATUS_OK &&
                       scratch_bytes == 0U,
                   "execute grouped strided Conv1d without scratch");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT(test, m3_op_test_u16(&output)[index] ==
                                 output_expected[index],
                       "Conv1d hard-coded BF16 result");
    }

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &dilation_input, M3_DTYPE_F32, 3U,
                          dilation_input_shape, dilation_input_values) &&
            m3_op_test_tensor(&fixture, &dilation_weight, M3_DTYPE_F32, 3U,
                              dilation_weight_shape,
                              dilation_weight_values) &&
            m3_op_test_tensor(&fixture, &dilation_output, M3_DTYPE_F32, 3U,
                              dilation_output_shape, dilation_zeros),
        "create dilated Conv1d tensors");
    command = m3_test_conv1d_command(
        &dilation_input, &dilation_weight, NULL, &dilation_output, 1U, 1U,
        2U, 0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "execute dilated Conv1d");
    for (index = 0U; index < 3U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&dilation_output)[index],
                           dilation_expected[index], 0.0F, 0.0F,
                           "dilated Conv1d hard-coded result");
    }

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &rounding_input, M3_DTYPE_F32, 3U,
                          rounding_input_shape, rounding_input_values) &&
            m3_op_test_tensor(&fixture, &rounding_weight, M3_DTYPE_F32, 3U,
                              rounding_weight_shape,
                              rounding_weight_values) &&
            m3_op_test_tensor(&fixture, &rounding_bias, M3_DTYPE_F32, 1U,
                              rounding_bias_shape,
                              rounding_bias_value) &&
            m3_op_test_tensor(&fixture, &rounding_output, M3_DTYPE_F32, 3U,
                              rounding_output_shape, rounding_zero),
        "create rounding-sensitive Conv1d tensors");
    command = m3_test_conv1d_command(
        &rounding_input, &rounding_weight, &rounding_bias, &rounding_output,
        1U, 1U, 1U, 0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       m3_op_test_f32(&rounding_output)[0] == 1.0F,
                   "Conv1d starts accumulation with bias before channels");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_conv_transpose1d(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 2U};
    const uint64_t bias_shape[] = {1U};
    const uint64_t output_shape[] = {1U, 1U, 5U};
    const uint16_t input_values[] = {0x4000U, 0x4040U};
    const uint16_t weight_values[] = {0x3c00U, 0x4900U};
    const uint16_t bias_values[] = {0x3f00U};
    const uint16_t output_zeros[5] = {0};
    const uint16_t output_expected[] = {
        0x3800U, 0x4de0U, 0x3800U, 0x4fa0U, 0x3800U
    };
    const uint64_t grouped_input_shape[] = {1U, 2U, 1U};
    const uint64_t grouped_weight_shape[] = {2U, 1U, 1U};
    const uint64_t grouped_output_shape[] = {1U, 2U, 1U};
    const uint64_t grouped_bias_shape[] = {2U};
    const size_t grouped_input_strides[] = {16U, 8U, 4U};
    const float grouped_input_backing[] = {2, -99, 3, -99};
    const float grouped_weight_values[] = {4, 5};
    const float grouped_bias_values[] = {1, -1};
    const float grouped_output_zeros[2] = {0};
    const float grouped_expected[] = {9, 14};
    m3_op_test_fixture fixture;
    m3_storage *grouped_input_storage = NULL;
    m3_tensor_view input;
    m3_tensor_view weight;
    m3_tensor_view bias;
    m3_tensor_view output;
    m3_tensor_view grouped_input;
    m3_tensor_view grouped_weight;
    m3_tensor_view grouped_bias;
    m3_tensor_view grouped_output;
    m3_command command;
    m3_error error;
    size_t scratch_bytes = SIZE_MAX;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create ConvTranspose1d fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &input, M3_DTYPE_BF16, 3U,
                          input_shape, input_values) &&
            m3_op_test_tensor(&fixture, &weight, M3_DTYPE_F16, 3U,
                              weight_shape, weight_values) &&
            m3_op_test_tensor(&fixture, &bias, M3_DTYPE_BF16, 1U,
                              bias_shape, bias_values) &&
            m3_op_test_tensor(&fixture, &output, M3_DTYPE_F16, 3U,
                              output_shape, output_zeros),
        "create mixed-dtype ConvTranspose1d tensors");
    command = m3_test_conv_transpose1d_command(
        &input, &weight, &bias, &output, 1U, 2U, 2U, 1U, 0U, 1U);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U,
                                      &scratch_bytes, &error) ==
                           M3_STATUS_OK &&
                       scratch_bytes == 0U,
                   "execute ConvTranspose1d with output padding");
    for (index = 0U; index < 5U; ++index) {
        M3_TEST_EXPECT(test, m3_op_test_u16(&output)[index] ==
                                 output_expected[index],
                       "ConvTranspose1d hard-coded F16 result");
    }

    m3_tensor_view_init(&grouped_input);
    M3_TEST_EXPECT(
        test,
        m3_op_test_storage(&fixture, sizeof(grouped_input_backing),
                           &grouped_input_storage) &&
            m3_storage_write(grouped_input_storage, 0U,
                             grouped_input_backing,
                             sizeof(grouped_input_backing), &error) ==
                M3_STATUS_OK &&
            m3_tensor_view_strided(
                &grouped_input, grouped_input_storage, M3_DTYPE_F32, 3U,
                grouped_input_shape, grouped_input_strides, 0U, &error) ==
                M3_STATUS_OK &&
            m3_op_test_tensor(&fixture, &grouped_weight, M3_DTYPE_F32, 3U,
                              grouped_weight_shape,
                              grouped_weight_values) &&
            m3_op_test_tensor(&fixture, &grouped_bias, M3_DTYPE_F32, 1U,
                              grouped_bias_shape, grouped_bias_values) &&
            m3_op_test_tensor(&fixture, &grouped_output, M3_DTYPE_F32, 3U,
                              grouped_output_shape,
                              grouped_output_zeros),
        "create grouped strided ConvTranspose1d tensors");
    command = m3_test_conv_transpose1d_command(
        &grouped_input, &grouped_weight, &grouped_bias, &grouped_output, 2U,
        1U, 1U, 0U, 0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "execute grouped strided ConvTranspose1d");
    for (index = 0U; index < 2U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&grouped_output)[index],
                           grouped_expected[index], 0.0F, 0.0F,
                           "grouped ConvTranspose1d hard-coded result");
    }
    m3_op_test_fixture_dispose(&fixture);
}
