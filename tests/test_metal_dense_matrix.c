/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_dense_test.h"

#include <stdint.h>
#include <string.h>

static m3_command m3_test_metal_matmul_command(
    const m3_tensor_view *left, const m3_tensor_view *right,
    m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_MATMUL;
    command.descriptor.matmul.left = left;
    command.descriptor.matmul.right = right;
    command.descriptor.matmul.output = output;
    return command;
}

static m3_command m3_test_metal_linear_command(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    const m3_tensor_view *bias, m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = input;
    command.descriptor.linear.weight = weight;
    command.descriptor.linear.bias = bias;
    command.descriptor.linear.output = output;
    return command;
}

static bool m3_test_metal_small_matmul(
    m3_test_context *test, m3_metal_dense_fixture *fixture,
    m3_dtype dtype, const void *left_values, const void *right_values,
    const void *zeros, const char *message)
{
    const uint64_t left_shape[] = {1U, 2U};
    const uint64_t right_shape[] = {2U, 1U};
    const uint64_t output_shape[] = {1U, 1U};
    m3_metal_dense_view left;
    m3_metal_dense_view right;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(fixture, &left, dtype, 2U,
                                         left_shape, left_values) &&
                       m3_metal_dense_tensor(fixture, &right, dtype, 2U,
                                             right_shape, right_values) &&
                       m3_metal_dense_tensor(fixture, &output, dtype, 2U,
                                             output_shape, zeros),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_matmul_command(
        &left.host, &right.host, &output.host);
    metal = m3_test_metal_matmul_command(
        &left.metal, &right.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error),
                   message);
    M3_TEST_EXPECT(test, m3_metal_dense_equal(&output), message);
    return true;
}

static bool m3_test_metal_matmul_cases(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t left_shape[] = {2U, 3U};
    const uint64_t right_shape[] = {3U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const size_t left_strides[] = {7U * sizeof(float),
                                   2U * sizeof(float)};
    const size_t right_strides[] = {3U * sizeof(float),
                                    2U * sizeof(float)};
    const float left_backing[] = {
        1, 91, 2, 92, 3, 93, 94, 4, 95, 5, 96, 6
    };
    const float right_backing[] = {1, 91, 2, 3, 92, 4, 5, 93, 6};
    const float zeros4[] = {0, 0, 0, 0};
    const float expected[] = {22, 28, 49, 64};
    const uint16_t f16_left[] = {0x3c00U, 0x4000U};
    const uint16_t f16_right[] = {0x4200U, 0x4400U};
    const uint16_t bf16_left[] = {0x3f80U, 0x4000U};
    const uint16_t bf16_right[] = {0x4040U, 0x4080U};
    const uint16_t zero16[] = {0U};
    m3_metal_dense_view left;
    m3_metal_dense_view right;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &left, M3_DTYPE_F32, 2U, left_shape,
                       left_strides, sizeof(left_backing), left_backing) &&
                       m3_metal_dense_strided(
                           fixture, &right, M3_DTYPE_F32, 2U, right_shape,
                           right_strides, sizeof(right_backing),
                           right_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, zeros4),
                   "create reduction-strided F32 Metal matmul");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_matmul_command(
        &left.host, &right.host, &output.host);
    metal = m3_test_metal_matmul_command(
        &left.metal, &right.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "Metal matmul matches host for non-unit reduction "
                   "strides");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected[index],
                       "Metal matmul has the expected sequential result");
    }
    if (!m3_test_metal_small_matmul(
            test, fixture, M3_DTYPE_F16, f16_left, f16_right, zero16,
            "Metal F16 matmul matches host") ||
        !m3_test_metal_small_matmul(
            test, fixture, M3_DTYPE_BF16, bf16_left, bf16_right, zero16,
            "Metal BF16 matmul matches host")) {
        return false;
    }
    return true;
}

static bool m3_test_metal_matmul_rounding(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t left_shape[] = {1U, 2U};
    const uint64_t right_shape[] = {2U, 1U};
    const uint64_t output_shape[] = {1U, 1U};
    const float left_values[] = {-1.0F, 0x1.000002p+0F};
    const float right_values[] = {1.0F, 0x1.fffffcp-1F};
    const float sentinel[] = {-9.0F};
    m3_metal_dense_view left;
    m3_metal_dense_view right;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    uint32_t bits;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &left, M3_DTYPE_F32, 2U, left_shape,
                       left_values) &&
                       m3_metal_dense_tensor(
                           fixture, &right, M3_DTYPE_F32, 2U, right_shape,
                           right_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, sentinel),
                   "create FMA-sensitive Metal matmul");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_matmul_command(
        &left.host, &right.host, &output.host);
    metal = m3_test_metal_matmul_command(
        &left.metal, &right.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "Metal matmul keeps separate multiply and add rounding");
    (void)memcpy(&bits, m3_op_test_f32(&output.metal), sizeof(bits));
    M3_TEST_EXPECT(test, bits == UINT32_C(0x00000000),
                   "FMA-sensitive matmul produces positive zero");
    return true;
}

static bool m3_test_metal_matmul_zero_inner(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t left_shape[] = {2U, 0U};
    const uint64_t right_shape[] = {0U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const float sentinels[] = {-1, -2, -3, -4};
    m3_metal_dense_view left;
    m3_metal_dense_view right;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &left, M3_DTYPE_F32, 2U, left_shape,
                       NULL) &&
                       m3_metal_dense_tensor(
                           fixture, &right, M3_DTYPE_F32, 2U, right_shape,
                           NULL) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, sentinels),
                   "create K=0 Metal matmul without input payloads");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_matmul_command(
        &left.host, &right.host, &output.host);
    metal = m3_test_metal_matmul_command(
        &left.metal, &right.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "K=0 Metal matmul dispatches without input buffers");
    for (index = 0U; index < 4U; ++index) {
        uint32_t bits;

        (void)memcpy(&bits, &m3_op_test_f32(&output.metal)[index],
                     sizeof(bits));
        M3_TEST_EXPECT(test, bits == UINT32_C(0x00000000),
                       "K=0 Metal matmul writes positive zero");
    }
    return true;
}

static bool m3_test_metal_linear_strided(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 2U, 3U};
    const uint64_t weight_shape[] = {2U, 3U};
    const uint64_t bias_shape[] = {2U};
    const uint64_t output_shape[] = {1U, 2U, 2U};
    const size_t input_strides[] = {14U * sizeof(float),
                                    7U * sizeof(float),
                                    2U * sizeof(float)};
    const size_t weight_strides[] = {7U * sizeof(float),
                                     2U * sizeof(float)};
    const size_t bias_strides[] = {2U * sizeof(float)};
    const float input_backing[] = {
        1, 91, 2, 92, 3, 93, 94, 4, 95, 5, 96, 6
    };
    const float weight_backing[] = {
        1, 91, 0, 92, -1, 93, 94, 0.5F, 95, 0.5F, 96, 0.5F
    };
    const float bias_backing[] = {1, 93, -1};
    const float zeros[] = {0, 0, 0, 0};
    const float expected_bias[] = {-1, 2, -1, 6.5F};
    const float expected_no_bias[] = {-2, 3, -2, 7.5F};
    m3_metal_dense_view input;
    m3_metal_dense_view weight;
    m3_metal_dense_view bias;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &input, M3_DTYPE_F32, 3U, input_shape,
                       input_strides, sizeof(input_backing),
                       input_backing) &&
                       m3_metal_dense_strided(
                           fixture, &weight, M3_DTYPE_F32, 2U,
                           weight_shape, weight_strides,
                           sizeof(weight_backing), weight_backing) &&
                       m3_metal_dense_strided(
                           fixture, &bias, M3_DTYPE_F32, 1U, bias_shape,
                           bias_strides, sizeof(bias_backing),
                           bias_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 3U,
                           output_shape, zeros),
                   "create feature-strided rank-three Metal linear");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_linear_command(
        &input.host, &weight.host, &bias.host, &output.host);
    metal = m3_test_metal_linear_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "Metal biased linear matches host for non-unit feature "
                   "strides");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected_bias[index],
                       "Metal biased linear has expected values");
    }
    host.descriptor.linear.bias = NULL;
    metal.descriptor.linear.bias = NULL;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "Metal bias-free linear matches host");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected_no_bias[index],
                       "Metal bias-free linear has expected values");
    }
    return true;
}

static bool m3_test_metal_small_linear(
    m3_test_context *test, m3_metal_dense_fixture *fixture,
    m3_dtype dtype, const void *input_values, const void *weight_values,
    const void *bias_values, const void *zeros, const char *message)
{
    const uint64_t input_shape[] = {1U, 2U};
    const uint64_t weight_shape[] = {1U, 2U};
    const uint64_t vector_shape[] = {1U};
    const uint64_t output_shape[] = {1U, 1U};
    m3_metal_dense_view input;
    m3_metal_dense_view weight;
    m3_metal_dense_view bias;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &input, dtype, 2U, input_shape,
                       input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &weight, dtype, 2U, weight_shape,
                           weight_values) &&
                       m3_metal_dense_tensor(
                           fixture, &bias, dtype, 1U, vector_shape,
                           bias_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, dtype, 2U, output_shape,
                           zeros),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_linear_command(
        &input.host, &weight.host, &bias.host, &output.host);
    metal = m3_test_metal_linear_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error),
                   message);
    M3_TEST_EXPECT(test, m3_metal_dense_equal(&output), message);
    return true;
}

static bool m3_test_metal_linear_precision(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint16_t f16_input[] = {0x3c00U, 0x4000U};
    const uint16_t f16_weight[] = {0x4200U, 0x4400U};
    const uint16_t f16_bias[] = {0x3c00U};
    const uint16_t bf16_input[] = {0x3f80U, 0x4000U};
    const uint16_t bf16_weight[] = {0x4040U, 0x4080U};
    const uint16_t bf16_bias[] = {0x3f80U};
    const uint16_t zero16[] = {0U};
    const uint64_t input_shape[] = {2U};
    const uint64_t weight_shape[] = {1U, 2U};
    const uint64_t output_shape[] = {1U};
    const float input_values[] = {1.0F, 1.0F};
    const float weight_values[] = {-16777216.0F, 1.0F};
    const float bias_value[] = {16777216.0F};
    const float zero[] = {0.0F};
    m3_metal_dense_view input;
    m3_metal_dense_view weight;
    m3_metal_dense_view bias;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    if (!m3_test_metal_small_linear(
            test, fixture, M3_DTYPE_F16, f16_input, f16_weight, f16_bias,
            zero16, "Metal F16 linear matches host") ||
        !m3_test_metal_small_linear(
            test, fixture, M3_DTYPE_BF16, bf16_input, bf16_weight,
            bf16_bias, zero16, "Metal BF16 linear matches host")) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &input, M3_DTYPE_F32, 1U, input_shape,
                       input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &weight, M3_DTYPE_F32, 2U,
                           weight_shape, weight_values) &&
                       m3_metal_dense_tensor(
                           fixture, &bias, M3_DTYPE_F32, 1U, output_shape,
                           bias_value) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 1U,
                           output_shape, zero),
                   "create bias-order-sensitive Metal linear");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_linear_command(
        &input.host, &weight.host, &bias.host, &output.host);
    metal = m3_test_metal_linear_command(
        &input.metal, &weight.metal, &bias.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output) &&
                       m3_op_test_f32(&output.metal)[0] == 1.0F,
                   "Metal linear initializes accumulation with bias");
    return true;
}

void m3_test_metal_dense_matrix(m3_test_context *test)
{
    m3_metal_dense_fixture fixture;

    if (!m3_metal_dense_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_matmul_cases(test, &fixture) ||
        !m3_test_metal_matmul_rounding(test, &fixture) ||
        !m3_test_metal_matmul_zero_inner(test, &fixture) ||
        !m3_test_metal_linear_strided(test, &fixture) ||
        !m3_test_metal_linear_precision(test, &fixture)) {
        m3_metal_dense_fixture_dispose(&fixture);
        return;
    }
    m3_metal_dense_fixture_dispose(&fixture);
}
