/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_convolution_test.h"

#include <stdint.h>
#include <string.h>

static m3_command m3_test_metal_conv1d_command(
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

static bool m3_test_metal_conv1d_mixed_grouped(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
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
    const uint16_t expected[] = {
        0x4100U, 0x4100U, 0x4160U, 0x40e0U, 0x4150U, 0x4040U
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
        m3_metal_convolution_strided(
            fixture, &input, M3_DTYPE_F16, 3U, input_shape,
            input_strides, 0U, sizeof(input_backing), input_backing) &&
            m3_metal_convolution_tensor(
                fixture, &weight, M3_DTYPE_BF16, 3U, weight_shape,
                weight_values) &&
            m3_metal_convolution_tensor(
                fixture, &bias, M3_DTYPE_F32, 1U, bias_shape,
                bias_values) &&
            m3_metal_convolution_tensor(
                fixture, &output, M3_DTYPE_BF16, 3U, output_shape,
                output_zeros),
        "create grouped mixed-dtype Metal Conv1d");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv1d_command(
        &input.host, &weight.host, &bias.host, &output.host, 2U, 2U, 1U,
        1U, 2U);
    metal = m3_test_metal_conv1d_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal, 2U, 2U,
        1U, 1U, 2U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal grouped mixed Conv1d matches host exactly");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_u16(&output.metal)[index] ==
                           expected[index],
                       "Metal grouped Conv1d has expected BF16 bits");
    }
    return true;
}

static bool m3_test_metal_conv1d_all_strided(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 2U, 4U};
    const uint64_t weight_shape[] = {2U, 2U, 2U};
    const uint64_t bias_shape[] = {2U};
    const uint64_t output_shape[] = {1U, 2U, 3U};
    const size_t input_strides[] = {64U, 32U, 8U};
    const size_t weight_strides[] = {40U, 16U, 8U};
    const size_t bias_strides[] = {8U};
    const float input_backing[] = {
        -90, 1, -91, 2, -92, 3, -93, 4,
        -94, 5, -95, 6, -96, 7, -97, 8
    };
    const float weight_backing[] = {
        -80, 1, -81, 2, -82, -1, -83, 0.5F, -84,
        -85, -86, 3, -87, -2, -88, 0.25F, -89, 4
    };
    const float bias_backing[] = {-70, 0.5F, -71, -1.0F};
    const float output_sentinels[] = {-9, -9, -9, -9, -9, -9};
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
        "create fully strided Metal Conv1d inputs");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv1d_command(
        &input.host, &weight.host, &bias.host, &output.host, 1U, 2U, 2U,
        2U, 1U);
    metal = m3_test_metal_conv1d_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal, 1U, 2U,
        2U, 2U, 1U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal Conv1d gathers every strided operand");
    return true;
}

static bool m3_test_metal_conv1d_low_precision(
    m3_test_context *test, m3_metal_convolution_fixture *fixture,
    m3_dtype dtype, const uint16_t *input_values,
    const uint16_t *weight_values, const uint16_t *bias_values,
    const char *message)
{
    const uint64_t input_shape[] = {1U, 1U, 3U};
    const uint64_t weight_shape[] = {1U, 1U, 2U};
    const uint64_t bias_shape[] = {1U};
    const uint64_t output_shape[] = {1U, 1U, 2U};
    const uint16_t zeros[2] = {0};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight;
    m3_metal_convolution_view bias;
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
                           fixture, &bias, dtype, 1U, bias_shape,
                           bias_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, dtype, 3U, output_shape,
                           zeros),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv1d_command(
        &input.host, &weight.host, &bias.host, &output.host, 1U, 1U, 1U,
        0U, 0U);
    metal = m3_test_metal_conv1d_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal, 1U, 1U,
        1U, 0U, 0U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   message);
    return true;
}

static bool m3_test_metal_conv1d_rounding(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 2U};
    const uint64_t output_shape[] = {1U, 1U, 1U};
    const float input_values[] = {-1.0F, 0x1.000002p+0F};
    const float weight_values[] = {1.0F, 0x1.fffffcp-1F};
    const float sentinel[] = {-9.0F};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight = {0};
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    uint32_t bits;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, M3_DTYPE_F32, 3U, input_shape,
                       input_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &weight, M3_DTYPE_F32, 3U,
                           weight_shape, weight_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, M3_DTYPE_F32, 3U,
                           output_shape, sentinel),
                   "create FMA-sensitive Metal Conv1d");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv1d_command(
        &input.host, &weight.host, NULL, &output.host, 1U, 1U, 1U, 0U,
        0U);
    metal = m3_test_metal_conv1d_command(
        &input.metal, &weight.metal, NULL, &output.metal, 1U, 1U, 1U, 0U,
        0U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal Conv1d keeps separate product and sum rounding");
    (void)memcpy(&bits, m3_op_test_f32(&output.metal), sizeof(bits));
    M3_TEST_EXPECT(test, bits == UINT32_C(0x00000000),
                   "FMA-sensitive Metal Conv1d produces positive zero");
    return true;
}

static bool m3_test_metal_conv1d_empty(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {0U, 2U, 3U};
    const uint64_t weight_shape[] = {2U, 2U, 1U};
    const uint64_t output_shape[] = {0U, 2U, 3U};
    const float weight_values[] = {1, 0, 0, 1};
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
                   "create empty-batch Metal Conv1d");
    if (weight.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_conv1d_command(
        &input.host, &weight.host, NULL, &output.host, 1U, 1U, 1U, 0U,
        0U);
    metal = m3_test_metal_conv1d_command(
        &input.metal, &weight.metal, NULL, &output.metal, 1U, 1U, 1U, 0U,
        0U);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "empty Metal Conv1d performs no buffer dispatch");
    return true;
}

void m3_test_metal_conv1d(m3_test_context *test)
{
    const uint16_t f16_input[] = {0x3c00U, 0x4000U, 0x4200U};
    const uint16_t f16_weight[] = {0x3800U, 0xbc00U};
    const uint16_t f16_bias[] = {0x3400U};
    const uint16_t bf16_input[] = {0x3f80U, 0x4000U, 0x4040U};
    const uint16_t bf16_weight[] = {0x3f00U, 0xbf80U};
    const uint16_t bf16_bias[] = {0x3e80U};
    m3_metal_convolution_fixture fixture;

    if (!m3_metal_convolution_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_conv1d_mixed_grouped(test, &fixture) ||
        !m3_test_metal_conv1d_all_strided(test, &fixture) ||
        !m3_test_metal_conv1d_low_precision(
            test, &fixture, M3_DTYPE_F16, f16_input, f16_weight, f16_bias,
            "F16 Metal Conv1d matches host exactly") ||
        !m3_test_metal_conv1d_low_precision(
            test, &fixture, M3_DTYPE_BF16, bf16_input, bf16_weight,
            bf16_bias, "BF16 Metal Conv1d matches host exactly") ||
        !m3_test_metal_conv1d_rounding(test, &fixture) ||
        !m3_test_metal_conv1d_empty(test, &fixture)) {
        m3_metal_convolution_fixture_dispose(&fixture);
        return;
    }
    m3_metal_convolution_fixture_dispose(&fixture);
}
