/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_convolution_test.h"

#include <stdint.h>

static m3_command m3_test_metal_conv_transpose_command(
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

static bool m3_test_metal_conv_transpose_mixed(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 2U};
    const uint64_t bias_shape[] = {1U};
    const uint64_t output_shape[] = {1U, 1U, 5U};
    const uint16_t input_values[] = {0x4000U, 0x4040U};
    const uint16_t weight_values[] = {0x3c00U, 0x4900U};
    const uint16_t bias_values[] = {0x3f00U};
    const uint16_t output_zeros[5] = {0};
    const uint16_t expected[] = {
        0x3800U, 0x4de0U, 0x3800U, 0x4fa0U, 0x3800U
    };
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight = {0};
    m3_metal_convolution_view bias;
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(
        test,
        m3_metal_convolution_tensor(
            fixture, &input, M3_DTYPE_BF16, 3U, input_shape,
            input_values) &&
            m3_metal_convolution_tensor(
                fixture, &weight, M3_DTYPE_F16, 3U, weight_shape,
                weight_values) &&
            m3_metal_convolution_tensor(
                fixture, &bias, M3_DTYPE_BF16, 1U, bias_shape,
                bias_values) &&
            m3_metal_convolution_tensor(
                fixture, &output, M3_DTYPE_F16, 3U, output_shape,
                output_zeros),
        "create mixed-dtype Metal ConvTranspose1d");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv_transpose_command(
        &input.host, &weight.host, &bias.host, &output.host, 1U, 2U, 2U,
        1U, 0U, 1U);
    metal = m3_test_metal_conv_transpose_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal, 1U, 2U,
        2U, 1U, 0U, 1U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal padded dilated transpose matches host exactly");
    for (index = 0U; index < 5U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_u16(&output.metal)[index] ==
                           expected[index],
                       "Metal transpose has expected F16 bits");
    }
    return true;
}

static bool m3_test_metal_conv_transpose_grouped_strided(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 2U, 2U};
    const uint64_t weight_shape[] = {2U, 1U, 2U};
    const uint64_t bias_shape[] = {2U};
    const uint64_t output_shape[] = {1U, 2U, 3U};
    const size_t input_strides[] = {32U, 16U, 8U};
    const size_t weight_strides[] = {24U, 16U, 8U};
    const size_t bias_strides[] = {8U};
    const float input_backing[] = {-9, 2, -8, 3, -7, 4, -6, 5};
    const float weight_backing[] = {
        -9, 1, -8, 10, -7, -6, -5, -1, -4, -10
    };
    const float bias_backing[] = {-3, 0.5F, -2, -0.5F};
    const float output_sentinels[] = {-1, -1, -1, -1, -1, -1};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight;
    m3_metal_convolution_view bias;
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(
        test,
        m3_metal_convolution_strided(
            fixture, &input, M3_DTYPE_F32, 3U, input_shape,
            input_strides, sizeof(float), sizeof(input_backing),
            input_backing) &&
            m3_metal_convolution_strided(
                fixture, &weight, M3_DTYPE_F32, 3U, weight_shape,
                weight_strides, sizeof(float), sizeof(weight_backing),
                weight_backing) &&
            m3_metal_convolution_strided(
                fixture, &bias, M3_DTYPE_F32, 1U, bias_shape,
                bias_strides, sizeof(float), sizeof(bias_backing),
                bias_backing) &&
            m3_metal_convolution_tensor(
                fixture, &output, M3_DTYPE_F32, 3U, output_shape,
                output_sentinels),
        "create grouped strided Metal ConvTranspose1d");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv_transpose_command(
        &input.host, &weight.host, &bias.host, &output.host, 2U, 2U, 1U,
        1U, 1U, 1U);
    metal = m3_test_metal_conv_transpose_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal, 2U, 2U,
        1U, 1U, 1U, 1U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal grouped transpose gathers strided operands");
    return true;
}

static bool m3_test_metal_conv_transpose_overlap(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 2U, 3U};
    const uint64_t weight_shape[] = {2U, 1U, 3U};
    const uint64_t output_shape[] = {1U, 1U, 5U};
    const float input_values[] = {1, 2, 3, 4, 5, 6};
    const float weight_values[] = {1, 10, 100, -1, -10, -100};
    const float sentinels[] = {-9, -9, -9, -9, -9};
    const float expected[] = {-3, -33, -333, -330, -300};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight = {0};
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, M3_DTYPE_F32, 3U, input_shape,
                       input_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &weight, M3_DTYPE_F32, 3U,
                           weight_shape, weight_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, M3_DTYPE_F32, 3U,
                           output_shape, sentinels),
                   "create overlapping-sum Metal ConvTranspose1d");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv_transpose_command(
        &input.host, &weight.host, NULL, &output.host, 1U, 1U, 1U, 0U,
        0U, 0U);
    metal = m3_test_metal_conv_transpose_command(
        &input.metal, &weight.metal, NULL, &output.metal, 1U, 1U, 1U, 0U,
        0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal transpose preserves overlapping sum order");
    for (index = 0U; index < 5U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected[index],
                       "Metal transpose overlap has expected value");
    }
    return true;
}

static bool m3_test_metal_conv_transpose_low_precision(
    m3_test_context *test, m3_metal_convolution_fixture *fixture,
    m3_dtype dtype, const uint16_t *input_values,
    const uint16_t *weight_values, const char *message)
{
    const uint64_t input_shape[] = {1U, 1U, 1U};
    const uint64_t weight_shape[] = {1U, 1U, 1U};
    const uint64_t output_shape[] = {1U, 1U, 1U};
    const uint16_t zero[] = {0U};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight;
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, dtype, 3U, input_shape,
                       input_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &weight, dtype, 3U, weight_shape,
                           weight_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, dtype, 3U, output_shape,
                           zero),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv_transpose_command(
        &input.host, &weight.host, NULL, &output.host, 1U, 1U, 1U, 0U,
        0U, 0U);
    metal = m3_test_metal_conv_transpose_command(
        &input.metal, &weight.metal, NULL, &output.metal, 1U, 1U, 1U, 0U,
        0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   message);
    return true;
}

static bool m3_test_metal_conv_transpose_empty(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {0U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 2U};
    const uint64_t output_shape[] = {0U, 1U, 3U};
    const float weight_values[] = {1, 2};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight = {0};
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, M3_DTYPE_F32, 3U, input_shape,
                       NULL) &&
                       m3_metal_convolution_tensor(
                           fixture, &weight, M3_DTYPE_F32, 3U,
                           weight_shape, weight_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, M3_DTYPE_F32, 3U,
                           output_shape, NULL),
                   "create empty-batch Metal ConvTranspose1d");
    if (weight.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv_transpose_command(
        &input.host, &weight.host, NULL, &output.host, 1U, 1U, 1U, 0U,
        0U, 0U);
    metal = m3_test_metal_conv_transpose_command(
        &input.metal, &weight.metal, NULL, &output.metal, 1U, 1U, 1U, 0U,
        0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "empty Metal transpose performs no buffer dispatch");
    return true;
}

void m3_test_metal_conv_transpose1d(m3_test_context *test)
{
    const uint16_t f16_input[] = {0x4200U};
    const uint16_t f16_weight[] = {0x3800U};
    const uint16_t bf16_input[] = {0x4040U};
    const uint16_t bf16_weight[] = {0x3f00U};
    m3_metal_convolution_fixture fixture;

    if (!m3_metal_convolution_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_conv_transpose_mixed(test, &fixture) ||
        !m3_test_metal_conv_transpose_grouped_strided(test, &fixture) ||
        !m3_test_metal_conv_transpose_overlap(test, &fixture) ||
        !m3_test_metal_conv_transpose_low_precision(
            test, &fixture, M3_DTYPE_F16, f16_input, f16_weight,
            "F16 Metal ConvTranspose1d matches host exactly") ||
        !m3_test_metal_conv_transpose_low_precision(
            test, &fixture, M3_DTYPE_BF16, bf16_input, bf16_weight,
            "BF16 Metal ConvTranspose1d matches host exactly") ||
        !m3_test_metal_conv_transpose_empty(test, &fixture)) {
        m3_metal_convolution_fixture_dispose(&fixture);
        return;
    }
    m3_metal_convolution_fixture_dispose(&fixture);
}
