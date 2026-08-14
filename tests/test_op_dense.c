/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stddef.h>

void m3_test_op_dense_norms(m3_test_context *test)
{
    const uint64_t left_shape[] = {2U, 3U};
    const uint64_t right_shape[] = {3U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const size_t transposed_strides[] = {4U, 12U};
    const float left_values[] = {1, 2, 3, 4, 5, 6};
    const float right_backing[] = {1, 3, 5, 2, 4, 6};
    const float zeros4[4] = {0};
    const float matmul_expected[] = {22, 28, 49, 64};
    const uint64_t rank3_shape[] = {2U, 2U, 3U};
    const uint64_t weight_shape[] = {2U, 3U};
    const uint64_t bias_shape[] = {2U};
    const uint64_t linear_shape[] = {2U, 2U, 2U};
    const float linear_input_values[] = {
        1, 2, 3, 4, 5, 6, -1, 0, 1, 2, 0, -2
    };
    const float weight_values[] = {1, 0, -1, 0.5F, 0.5F, 0.5F};
    const float bias_values[] = {1, -1};
    const float zeros8[8] = {0};
    const float linear_expected[] = {-1, 2, -1, 6.5F, -1, -1, 5, -1};
    const uint64_t norm_shape[] = {1U, 2U};
    const float rms_values[] = {3, 4};
    const float rms_scale_values[] = {1, 2};
    const uint64_t layer_shape[] = {1U, 3U};
    const uint64_t three[] = {3U};
    const float layer_values[] = {1, 2, 3};
    const float layer_scale_values[] = {1, 2, 1};
    const float layer_bias_values[] = {0, 1, -1};
    const float zeros3[3] = {0};
    const float gate_values[] = {0, 1};
    const float up_values[] = {2, 2};
    const float zeros2[2] = {0};
    const uint64_t rounding_input_shape[] = {2U};
    const uint64_t rounding_weight_shape[] = {1U, 2U};
    const uint64_t rounding_output_shape[] = {1U};
    const float rounding_input_values[] = {1, 1};
    const float rounding_weight_values[] = {-16777216.0F, 1.0F};
    const float rounding_bias_value[] = {16777216.0F};
    const float rounding_zero[] = {0};
    const uint64_t zero_input_shape[] = {1U, 0U};
    const uint64_t zero_input_weight_shape[] = {1U, 0U};
    const uint64_t one_by_one_shape[] = {1U, 1U};
    const uint64_t zero_output_weight_shape[] = {0U, 1U};
    const uint64_t zero_output_shape[] = {1U, 0U};
    m3_op_test_fixture fixture;
    m3_storage *right_storage = NULL;
    m3_tensor_view left;
    m3_tensor_view right;
    m3_tensor_view matmul_output;
    m3_tensor_view linear_input;
    m3_tensor_view weight;
    m3_tensor_view bias;
    m3_tensor_view linear_output;
    m3_tensor_view rms;
    m3_tensor_view rms_scale;
    m3_tensor_view layer;
    m3_tensor_view layer_scale;
    m3_tensor_view layer_bias;
    m3_tensor_view layer_output;
    m3_tensor_view gate;
    m3_tensor_view up;
    m3_tensor_view gated_output;
    m3_tensor_view rounding_input;
    m3_tensor_view rounding_weight;
    m3_tensor_view rounding_bias;
    m3_tensor_view rounding_output;
    m3_tensor_view zero_input;
    m3_tensor_view zero_input_weight;
    m3_tensor_view zero_input_output;
    m3_tensor_view zero_output_input;
    m3_tensor_view zero_output_weight;
    m3_tensor_view zero_output;
    m3_command command;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create dense fixture");
    if (fixture.backend == NULL) {
        return;
    }
    m3_tensor_view_init(&right);
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &left, M3_DTYPE_F32, 2U,
                                     left_shape, left_values) &&
                       m3_op_test_storage(&fixture, sizeof(right_backing),
                                          &right_storage) &&
                       m3_storage_write(right_storage, 0U, right_backing,
                                        sizeof(right_backing), &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_strided(
                           &right, right_storage, M3_DTYPE_F32, 2U,
                           right_shape, transposed_strides, 0U, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_tensor(&fixture, &matmul_output,
                                         M3_DTYPE_F32, 2U, output_shape,
                                         zeros4),
                   "create transposed-stride matmul tensors");
    command.kind = M3_OP_MATMUL;
    command.descriptor.matmul.left = &left;
    command.descriptor.matmul.right = &right;
    command.descriptor.matmul.output = &matmul_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute 2x3 by 3x2 matmul");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&matmul_output)[index],
                           matmul_expected[index], 0.0F, 0.0F,
                           "matmul hard-coded result");
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &linear_input, M3_DTYPE_F32,
                                     3U, rank3_shape,
                                     linear_input_values) &&
                       m3_op_test_tensor(&fixture, &weight, M3_DTYPE_F32, 2U,
                                         weight_shape, weight_values) &&
                       m3_op_test_tensor(&fixture, &bias, M3_DTYPE_F32, 1U,
                                         bias_shape, bias_values) &&
                       m3_op_test_tensor(&fixture, &linear_output,
                                         M3_DTYPE_F32, 3U, linear_shape,
                                         zeros8),
                   "create rank-three linear tensors");
    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = &linear_input;
    command.descriptor.linear.weight = &weight;
    command.descriptor.linear.bias = &bias;
    command.descriptor.linear.output = &linear_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute rank-three linear with bias");
    for (index = 0U; index < 8U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&linear_output)[index],
                           linear_expected[index], 0.0F, 0.0F,
                           "linear hard-coded result");
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &rms, M3_DTYPE_F32, 2U,
                                     norm_shape, rms_values) &&
                       m3_op_test_tensor(&fixture, &rms_scale, M3_DTYPE_F32,
                                         1U, bias_shape, rms_scale_values),
                   "create in-place RMSNorm tensors");
    command.kind = M3_OP_RMS_NORM;
    command.descriptor.rms_norm.input = &rms;
    command.descriptor.rms_norm.scale = &rms_scale;
    command.descriptor.rms_norm.output = &rms;
    command.descriptor.rms_norm.epsilon = 3.5F;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute exact-alias RMSNorm");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&rms)[0], 0.75F,
                       1.0e-6F, 1.0e-6F, "RMSNorm first channel");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&rms)[1], 2.0F,
                       1.0e-6F, 1.0e-6F, "RMSNorm scaled channel");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &layer, M3_DTYPE_F32, 2U,
                                     layer_shape, layer_values) &&
                       m3_op_test_tensor(&fixture, &layer_scale,
                                         M3_DTYPE_F32, 1U, three,
                                         layer_scale_values) &&
                       m3_op_test_tensor(&fixture, &layer_bias, M3_DTYPE_F32,
                                         1U, three, layer_bias_values) &&
                       m3_op_test_tensor(&fixture, &layer_output,
                                         M3_DTYPE_F32, 2U, layer_shape,
                                         zeros3),
                   "create LayerNorm tensors");
    command.kind = M3_OP_LAYER_NORM;
    command.descriptor.layer_norm.input = &layer;
    command.descriptor.layer_norm.scale = &layer_scale;
    command.descriptor.layer_norm.bias = &layer_bias;
    command.descriptor.layer_norm.output = &layer_output;
    command.descriptor.layer_norm.epsilon = 1.0F / 3.0F;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute two-pass LayerNorm");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&layer_output)[0], -1.0F,
                       1.0e-6F, 1.0e-6F, "LayerNorm first channel");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&layer_output)[1], 1.0F,
                       1.0e-6F, 1.0e-6F, "LayerNorm biased channel");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&layer_output)[2], 0.0F,
                       1.0e-6F, 1.0e-6F, "LayerNorm final channel");
    command.descriptor.layer_norm.epsilon = 0.0F;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "norm validation rejects zero epsilon");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &gate, M3_DTYPE_F32, 1U,
                                     bias_shape, gate_values) &&
                       m3_op_test_tensor(&fixture, &up, M3_DTYPE_F32, 1U,
                                         bias_shape, up_values) &&
                       m3_op_test_tensor(&fixture, &gated_output,
                                         M3_DTYPE_F32, 1U, bias_shape,
                                         zeros2),
                   "create gated SiLU tensors");
    command.kind = M3_OP_GATED_SILU;
    command.descriptor.gated_silu.left = &gate;
    command.descriptor.gated_silu.right = &up;
    command.descriptor.gated_silu.output = &gated_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute gated SiLU");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&gated_output)[0], 0.0F, 0.0F,
                       0.0F, "gated SiLU zero gate");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&gated_output)[1], 1.46211720F,
                       1.0e-6F, 1.0e-6F, "gated SiLU positive gate");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &rounding_input,
                                     M3_DTYPE_F32, 1U,
                                     rounding_input_shape,
                                     rounding_input_values) &&
                       m3_op_test_tensor(&fixture, &rounding_weight,
                                         M3_DTYPE_F32, 2U,
                                         rounding_weight_shape,
                                         rounding_weight_values) &&
                       m3_op_test_tensor(&fixture, &rounding_bias,
                                         M3_DTYPE_F32, 1U,
                                         rounding_output_shape,
                                         rounding_bias_value) &&
                       m3_op_test_tensor(&fixture, &rounding_output,
                                         M3_DTYPE_F32, 1U,
                                         rounding_output_shape,
                                         rounding_zero),
                   "create rounding-sensitive linear tensors");
    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = &rounding_input;
    command.descriptor.linear.weight = &rounding_weight;
    command.descriptor.linear.bias = &rounding_bias;
    command.descriptor.linear.output = &rounding_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_f32(&rounding_output)[0] == 1.0F,
                   "linear starts accumulation with bias before features");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &zero_input, M3_DTYPE_F32, 2U,
                                     zero_input_shape, NULL) &&
                       m3_op_test_tensor(&fixture, &zero_input_weight,
                                         M3_DTYPE_F32, 2U,
                                         zero_input_weight_shape, NULL) &&
                       m3_op_test_tensor(&fixture, &zero_input_output,
                                         M3_DTYPE_F32, 2U, one_by_one_shape,
                                         rounding_zero),
                   "create zero-input-feature linear views");
    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = &zero_input;
    command.descriptor.linear.weight = &zero_input_weight;
    command.descriptor.linear.bias = NULL;
    command.descriptor.linear.output = &zero_input_output;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "linear rejects a zero input feature dimension");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &zero_output_input,
                                     M3_DTYPE_F32, 2U, one_by_one_shape,
                                     rounding_zero) &&
                       m3_op_test_tensor(&fixture, &zero_output_weight,
                                         M3_DTYPE_F32, 2U,
                                         zero_output_weight_shape, NULL) &&
                       m3_op_test_tensor(&fixture, &zero_output,
                                         M3_DTYPE_F32, 2U,
                                         zero_output_shape, NULL),
                   "create zero-output-feature linear views");
    command.descriptor.linear.input = &zero_output_input;
    command.descriptor.linear.weight = &zero_output_weight;
    command.descriptor.linear.output = &zero_output;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "linear rejects a zero output feature dimension");
    m3_op_test_fixture_dispose(&fixture);
}

static bool m3_test_rope_execute(m3_op_test_fixture *fixture,
                                 m3_tensor_view *input,
                                 m3_tensor_view *cosines,
                                 m3_tensor_view *sines,
                                 m3_tensor_view *output,
                                 uint32_t rotary_dimension,
                                 m3_rope_mode mode, m3_error *error)
{
    m3_command command;

    command.kind = M3_OP_ROPE;
    command.descriptor.rope.input = input;
    command.descriptor.rope.cosines = cosines;
    command.descriptor.rope.sines = sines;
    command.descriptor.rope.output = output;
    command.descriptor.rope.position_offset = 0U;
    command.descriptor.rope.rotary_dimension = rotary_dimension;
    command.descriptor.rope.mode = mode;
    return m3_op_test_execute(fixture, &command, 1U, NULL, error) ==
           M3_STATUS_OK;
}

void m3_test_op_rope(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 1U, 1U, 6U};
    const uint64_t partial_table_shape[] = {1U, 2U};
    const uint64_t full_table_shape[] = {1U, 3U};
    const float input_values[] = {1, 2, 3, 4, 5, 6};
    const float partial_cosines[] = {1, 0};
    const float partial_sines[] = {0, 1};
    const float full_cosines[] = {1, 0, -1};
    const float full_sines[] = {0, 1, 0};
    const float zeros6[6] = {0};
    const float half_expected[] = {1, -4, 3, 2, 5, 6};
    const float interleaved_expected[] = {1, 2, -4, 3, 5, 6};
    const float full_expected[] = {1, 2, -4, 3, -5, -6};
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view partial_cos;
    m3_tensor_view partial_sin;
    m3_tensor_view full_cos;
    m3_tensor_view full_sin;
    m3_tensor_view output;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create RoPE fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 4U,
                                     input_shape, input_values) &&
                       m3_op_test_tensor(&fixture, &partial_cos,
                                         M3_DTYPE_F32, 2U,
                                         partial_table_shape,
                                         partial_cosines) &&
                       m3_op_test_tensor(&fixture, &partial_sin,
                                         M3_DTYPE_F32, 2U,
                                         partial_table_shape,
                                         partial_sines) &&
                       m3_op_test_tensor(&fixture, &full_cos, M3_DTYPE_F32,
                                         2U, full_table_shape,
                                         full_cosines) &&
                       m3_op_test_tensor(&fixture, &full_sin, M3_DTYPE_F32,
                                         2U, full_table_shape, full_sines) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 4U,
                                         input_shape, zeros6),
                   "create RoPE input and position-zero tables");
    M3_TEST_EXPECT(test,
                   m3_test_rope_execute(&fixture, &input, &partial_cos,
                                        &partial_sin, &output, 4U,
                                        M3_ROPE_HALF_SPLIT, &error),
                   "execute partial half-split RoPE");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output)[index],
                           half_expected[index], 0.0F, 0.0F,
                           "half-split RoPE value");
    }
    M3_TEST_EXPECT(test,
                   m3_test_rope_execute(&fixture, &input, &partial_cos,
                                        &partial_sin, &output, 4U,
                                        M3_ROPE_INTERLEAVED, &error),
                   "execute partial interleaved RoPE");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output)[index],
                           interleaved_expected[index], 0.0F, 0.0F,
                           "interleaved RoPE value");
    }
    M3_TEST_EXPECT(test,
                   m3_test_rope_execute(&fixture, &input, &full_cos,
                                        &full_sin, &output, 6U,
                                        M3_ROPE_INTERLEAVED, &error),
                   "execute full position-zero RoPE");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output)[index],
                           full_expected[index], 0.0F, 0.0F,
                           "full RoPE value");
    }
    m3_op_test_fixture_dispose(&fixture);
}
