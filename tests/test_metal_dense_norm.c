/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_dense_test.h"

#include <stddef.h>
#include <stdint.h>

static m3_command m3_test_metal_rms_command(
    const m3_tensor_view *input, const m3_tensor_view *scale,
    m3_tensor_view *output, float epsilon)
{
    m3_command command = {0};

    command.kind = M3_OP_RMS_NORM;
    command.descriptor.rms_norm.input = input;
    command.descriptor.rms_norm.scale = scale;
    command.descriptor.rms_norm.output = output;
    command.descriptor.rms_norm.epsilon = epsilon;
    return command;
}

static m3_command m3_test_metal_layer_command(
    const m3_tensor_view *input, const m3_tensor_view *scale,
    const m3_tensor_view *bias, m3_tensor_view *output, float epsilon)
{
    m3_command command = {0};

    command.kind = M3_OP_LAYER_NORM;
    command.descriptor.layer_norm.input = input;
    command.descriptor.layer_norm.scale = scale;
    command.descriptor.layer_norm.bias = bias;
    command.descriptor.layer_norm.output = output;
    command.descriptor.layer_norm.epsilon = epsilon;
    return command;
}

static m3_command m3_test_metal_gated_command(
    const m3_tensor_view *gate, const m3_tensor_view *up,
    m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_GATED_SILU;
    command.descriptor.gated_silu.left = gate;
    command.descriptor.gated_silu.right = up;
    command.descriptor.gated_silu.output = output;
    return command;
}

static bool m3_test_metal_rms_strided(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t shape[] = {1U, 2U};
    const uint64_t vector_shape[] = {2U};
    const size_t input_strides[] = {3U * sizeof(float),
                                    2U * sizeof(float)};
    const size_t scale_strides[] = {2U * sizeof(float)};
    const float input_backing[] = {3.0F, 91.0F, 4.0F};
    const float scale_backing[] = {1.0F, 92.0F, 2.0F};
    const float zeros[] = {0.0F, 0.0F};
    m3_metal_dense_view input;
    m3_metal_dense_view scale;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &input, M3_DTYPE_F32, 2U, shape,
                       input_strides, sizeof(input_backing),
                       input_backing) &&
                       m3_metal_dense_strided(
                           fixture, &scale, M3_DTYPE_F32, 1U,
                           vector_shape, scale_strides,
                           sizeof(scale_backing), scale_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U, shape,
                           zeros),
                   "create strided F32 Metal RMSNorm");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_rms_command(
        &input.host, &scale.host, &output.host, 3.5F);
    metal = m3_test_metal_rms_command(
        &input.metal, &scale.metal, &output.metal, 3.5F);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "Metal RMSNorm matches host for strided F32 inputs");
    M3_TEST_EXPECT(test,
                   m3_op_test_f32(&output.metal)[0] == 0.75F &&
                       m3_op_test_f32(&output.metal)[1] == 2.0F,
                   "Metal RMSNorm follows sequential host arithmetic");
    return true;
}

static bool m3_test_metal_rms_precision(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t shape[] = {1U, 2U};
    const uint64_t vector_shape[] = {2U};
    const uint16_t bf16_input_values[] = {0x3d00U, 0x3f00U};
    const uint16_t bf16_scale_values[] = {0x3f40U, 0x3f40U};
    const uint16_t f16_input_values[] = {0x2800U, 0x3800U};
    const uint16_t f16_scale_values[] = {0x3a00U, 0x3a00U};
    const uint16_t zeros[] = {0U, 0U};
    m3_metal_dense_view bf16_input;
    m3_metal_dense_view bf16_scale;
    m3_metal_dense_view f16_input;
    m3_metal_dense_view f16_scale;
    m3_metal_dense_view f16_output;
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &bf16_input, M3_DTYPE_BF16, 2U, shape,
                       bf16_input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &bf16_scale, M3_DTYPE_BF16, 1U,
                           vector_shape, bf16_scale_values),
                   "create in-place BF16 Metal RMSNorm oracle");
    if (bf16_input.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_rms_command(
        &bf16_input.host, &bf16_scale.host, &bf16_input.host, 1.0e-6F);
    metal = m3_test_metal_rms_command(
        &bf16_input.metal, &bf16_scale.metal, &bf16_input.metal,
        1.0e-6F);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&bf16_input),
                   "in-place BF16 Metal RMSNorm matches host");
    M3_TEST_EXPECT(test,
                   m3_op_test_u16(&bf16_input.metal)[0] == 0x3d88U &&
                       m3_op_test_u16(&bf16_input.metal)[1] == 0x3f88U,
                   "Metal RMSNorm rounds normalized BF16 before scale");
    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &f16_input, M3_DTYPE_F16, 2U, shape,
                       f16_input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &f16_scale, M3_DTYPE_F16, 1U,
                           vector_shape, f16_scale_values) &&
                       m3_metal_dense_tensor(
                           fixture, &f16_output, M3_DTYPE_F16, 2U, shape,
                           zeros),
                   "create F16 Metal RMSNorm");
    host = m3_test_metal_rms_command(
        &f16_input.host, &f16_scale.host, &f16_output.host, 1.0e-6F);
    metal = m3_test_metal_rms_command(
        &f16_input.metal, &f16_scale.metal, &f16_output.metal, 1.0e-6F);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&f16_output),
                   "F16 Metal RMSNorm matches host exactly");
    return true;
}

static bool m3_test_metal_layer_norm(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t shape[] = {1U, 3U};
    const uint64_t vector_shape[] = {3U};
    const size_t strides[] = {5U * sizeof(float), 2U * sizeof(float)};
    const size_t vector_strides[] = {2U * sizeof(float)};
    const float input_backing[] = {1, 91, 2, 92, 3};
    const float scale_backing[] = {1, 93, 2, 94, 1};
    const float bias_backing[] = {0, 95, 1, 96, -1};
    const float zeros[] = {0, 0, 0};
    const float expected_bias[] = {-1, 1, 0};
    const float expected_no_bias[] = {-1, 0, 1};
    m3_metal_dense_view input;
    m3_metal_dense_view scale;
    m3_metal_dense_view bias;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &input, M3_DTYPE_F32, 2U, shape, strides,
                       sizeof(input_backing), input_backing) &&
                       m3_metal_dense_strided(
                           fixture, &scale, M3_DTYPE_F32, 1U,
                           vector_shape, vector_strides,
                           sizeof(scale_backing), scale_backing) &&
                       m3_metal_dense_strided(
                           fixture, &bias, M3_DTYPE_F32, 1U,
                           vector_shape, vector_strides,
                           sizeof(bias_backing), bias_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U, shape,
                           zeros),
                   "create strided F32 Metal LayerNorm");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_layer_command(
        &input.host, &scale.host, &bias.host, &output.host, 1.0F / 3.0F);
    metal = m3_test_metal_layer_command(
        &input.metal, &scale.metal, &bias.metal, &output.metal,
        1.0F / 3.0F);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "biased Metal LayerNorm matches host exactly");
    for (index = 0U; index < 3U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected_bias[index],
                       "biased Metal LayerNorm has expected values");
    }
    host.descriptor.layer_norm.bias = NULL;
    metal.descriptor.layer_norm.bias = NULL;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "bias-free Metal LayerNorm matches host exactly");
    for (index = 0U; index < 3U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected_no_bias[index],
                       "bias-free Metal LayerNorm has expected values");
    }
    return true;
}

static bool m3_test_metal_layer_low_precision(
    m3_test_context *test, m3_metal_dense_fixture *fixture,
    m3_dtype dtype, const char *message)
{
    const uint64_t shape[] = {1U, 3U};
    const uint64_t vector_shape[] = {3U};
    const uint16_t f16_input[] = {0x3c00U, 0x4000U, 0x4200U};
    const uint16_t f16_scale[] = {0x3c00U, 0x4000U, 0x3c00U};
    const uint16_t f16_bias[] = {0x0000U, 0x3c00U, 0xbc00U};
    const uint16_t bf16_input[] = {0x3f80U, 0x4000U, 0x4040U};
    const uint16_t bf16_scale[] = {0x3f80U, 0x4000U, 0x3f80U};
    const uint16_t bf16_bias[] = {0x0000U, 0x3f80U, 0xbf80U};
    const uint16_t zeros[] = {0U, 0U, 0U};
    const uint16_t *input_values =
        dtype == M3_DTYPE_F16 ? f16_input : bf16_input;
    const uint16_t *scale_values =
        dtype == M3_DTYPE_F16 ? f16_scale : bf16_scale;
    const uint16_t *bias_values =
        dtype == M3_DTYPE_F16 ? f16_bias : bf16_bias;
    m3_metal_dense_view input;
    m3_metal_dense_view scale;
    m3_metal_dense_view bias;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &input, dtype, 2U, shape, input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &scale, dtype, 1U, vector_shape,
                           scale_values) &&
                       m3_metal_dense_tensor(
                           fixture, &bias, dtype, 1U, vector_shape,
                           bias_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, dtype, 2U, shape, zeros),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_layer_command(
        &input.host, &scale.host, &bias.host, &output.host, 1.0F / 3.0F);
    metal = m3_test_metal_layer_command(
        &input.metal, &scale.metal, &bias.metal, &output.metal,
        1.0F / 3.0F);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   message);
    host.descriptor.layer_norm.output = &input.host;
    metal.descriptor.layer_norm.output = &input.metal;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&input),
                   "Metal LayerNorm permits an exact input alias");
    return true;
}

static bool m3_test_metal_gated_silu(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t shape[] = {2U};
    const size_t strides[] = {2U * sizeof(float)};
    const float gate_backing[] = {0.0F, 91.0F, 1.0F};
    const float up_backing[] = {2.0F, 92.0F, 2.0F};
    const float zeros[] = {0.0F, 0.0F};
    const uint64_t scalar_shape[] = {1U};
    const uint16_t bf16_gate_value[] = {0x3f00U};
    const uint16_t bf16_up_value[] = {0x3f40U};
    const uint16_t f16_gate_value[] = {0x3800U};
    const uint16_t f16_up_value[] = {0x3a00U};
    const uint16_t zero16[] = {0U};
    m3_metal_dense_view gate;
    m3_metal_dense_view up;
    m3_metal_dense_view output = {0};
    m3_metal_dense_view bf16_gate;
    m3_metal_dense_view bf16_up;
    m3_metal_dense_view bf16_output;
    m3_metal_dense_view f16_gate;
    m3_metal_dense_view f16_up;
    m3_metal_dense_view f16_output;
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &gate, M3_DTYPE_F32, 1U, shape, strides,
                       sizeof(gate_backing), gate_backing) &&
                       m3_metal_dense_strided(
                           fixture, &up, M3_DTYPE_F32, 1U, shape, strides,
                           sizeof(up_backing), up_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 1U, shape,
                           zeros),
                   "create strided F32 Metal gated SiLU");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_gated_command(
        &gate.host, &up.host, &output.host);
    metal = m3_test_metal_gated_command(
        &gate.metal, &up.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error),
                   "execute strided F32 Metal gated SiLU");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output.metal)[0],
                       m3_op_test_f32(&output.host)[0], 1.0e-7F, 1.0e-7F,
                       "Metal gated SiLU zero matches host");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output.metal)[1],
                       m3_op_test_f32(&output.host)[1], 1.0e-7F, 1.0e-7F,
                       "Metal gated SiLU positive value matches host");
    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &bf16_gate, M3_DTYPE_BF16, 1U,
                       scalar_shape, bf16_gate_value) &&
                       m3_metal_dense_tensor(
                           fixture, &bf16_up, M3_DTYPE_BF16, 1U,
                           scalar_shape, bf16_up_value) &&
                       m3_metal_dense_tensor(
                           fixture, &bf16_output, M3_DTYPE_BF16, 1U,
                           scalar_shape, zero16),
                   "create BF16 Metal gated SiLU oracle");
    host = m3_test_metal_gated_command(
        &bf16_gate.host, &bf16_up.host, &bf16_output.host);
    metal = m3_test_metal_gated_command(
        &bf16_gate.metal, &bf16_up.metal, &bf16_output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&bf16_output) &&
                       m3_op_test_u16(&bf16_output.metal)[0] == 0x3e6eU,
                   "Metal gated SiLU rounds BF16 activation before multiply");
    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &f16_gate, M3_DTYPE_F16, 1U,
                       scalar_shape, f16_gate_value) &&
                       m3_metal_dense_tensor(
                           fixture, &f16_up, M3_DTYPE_F16, 1U,
                           scalar_shape, f16_up_value) &&
                       m3_metal_dense_tensor(
                           fixture, &f16_output, M3_DTYPE_F16, 1U,
                           scalar_shape, zero16),
                   "create F16 Metal gated SiLU");
    host = m3_test_metal_gated_command(
        &f16_gate.host, &f16_up.host, &f16_output.host);
    metal = m3_test_metal_gated_command(
        &f16_gate.metal, &f16_up.metal, &f16_output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&f16_output),
                   "F16 Metal gated SiLU matches host exactly");
    host.descriptor.gated_silu.output = &bf16_gate.host;
    metal.descriptor.gated_silu.output = &bf16_gate.metal;
    host.descriptor.gated_silu.left = &bf16_gate.host;
    metal.descriptor.gated_silu.left = &bf16_gate.metal;
    host.descriptor.gated_silu.right = &bf16_up.host;
    metal.descriptor.gated_silu.right = &bf16_up.metal;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&bf16_gate),
                   "Metal gated SiLU permits an exact gate alias");
    host = m3_test_metal_gated_command(
        &f16_gate.host, &f16_up.host, &f16_up.host);
    metal = m3_test_metal_gated_command(
        &f16_gate.metal, &f16_up.metal, &f16_up.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&f16_up),
                   "Metal gated SiLU permits an exact up alias");
    return true;
}

void m3_test_metal_dense_normalization(m3_test_context *test)
{
    m3_metal_dense_fixture fixture;

    if (!m3_metal_dense_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_rms_strided(test, &fixture) ||
        !m3_test_metal_rms_precision(test, &fixture) ||
        !m3_test_metal_layer_norm(test, &fixture) ||
        !m3_test_metal_layer_low_precision(
            test, &fixture, M3_DTYPE_F16,
            "F16 Metal LayerNorm matches host exactly") ||
        !m3_test_metal_layer_low_precision(
            test, &fixture, M3_DTYPE_BF16,
            "BF16 Metal LayerNorm matches host exactly") ||
        !m3_test_metal_gated_silu(test, &fixture)) {
        m3_metal_dense_fixture_dispose(&fixture);
        return;
    }
    m3_metal_dense_fixture_dispose(&fixture);
}
