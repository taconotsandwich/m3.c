/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stdint.h>
#include <string.h>

static bool m3_test_transformer_attention_run(
    m3_op_test_fixture *fixture, m3_tensor_view *query,
    m3_tensor_view *key, m3_tensor_view *value, m3_tensor_view *output,
    m3_error *error)
{
    m3_command command;

    command.kind = M3_OP_ATTENTION;
    command.descriptor.attention.query = query;
    command.descriptor.attention.key = key;
    command.descriptor.attention.value = value;
    command.descriptor.attention.mask = NULL;
    command.descriptor.attention.output = output;
    command.descriptor.attention.scale = 1.0F;
    command.descriptor.attention.causal_offset = 0;
    command.descriptor.attention.causal = false;
    return m3_op_test_execute(fixture, &command, 1U, NULL, error) ==
           M3_STATUS_OK;
}

static bool m3_test_transformer_softmax_run(
    m3_op_test_fixture *fixture, m3_tensor_view *input,
    m3_tensor_view *output, m3_error *error)
{
    m3_command command;

    command.kind = M3_OP_SOFTMAX;
    command.descriptor.softmax.input = input;
    command.descriptor.softmax.output = output;
    return m3_op_test_execute(fixture, &command, 1U, NULL, error) ==
           M3_STATUS_OK;
}

static uint32_t m3_test_transformer_f32_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void m3_test_transformer_attention_precision(m3_test_context *test)
{
    const uint64_t score_shape[] = {2U};
    const uint64_t query_shape[] = {1U, 1U, 1U, 1U};
    const uint64_t key_shape[] = {1U, 1U, 2U, 1U};
    const float score_values[] = {0.0F, 1.0F};
    const uint16_t zero_probabilities[] = {0U, 0U};
    const uint16_t bf16_query_values[] = {0x3f80U};
    const uint16_t bf16_key_values[] = {0x0000U, 0x3f80U};
    const uint16_t bf16_value_values[] = {0xc280U, 0xc180U};
    const uint16_t bf16_zero[] = {0U};
    const uint16_t f16_query_values[] = {0x3c00U};
    const uint16_t f16_key_values[] = {0x0000U, 0x3c00U};
    const uint16_t f16_value_values[] = {0xd400U, 0xd240U};
    const uint16_t f16_zero[] = {0U};
    const float f32_query_values[] = {1.0F};
    const float f32_key_values[] = {0.0F, 1.0F};
    const float f32_value_values[] = {-64.0F, -16.0F};
    const float f32_zero[] = {0.0F};
    m3_op_test_fixture fixture;
    m3_tensor_view scores;
    m3_tensor_view bf16_probabilities;
    m3_tensor_view bf16_query;
    m3_tensor_view bf16_key;
    m3_tensor_view bf16_value;
    m3_tensor_view bf16_output;
    m3_tensor_view f16_probabilities;
    m3_tensor_view f16_query;
    m3_tensor_view f16_key;
    m3_tensor_view f16_value;
    m3_tensor_view f16_output;
    m3_tensor_view f32_query;
    m3_tensor_view f32_key;
    m3_tensor_view f32_value;
    m3_tensor_view f32_output;
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create transformer precision fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &scores, M3_DTYPE_F32, 1U,
                          score_shape, score_values) &&
            m3_op_test_tensor(&fixture, &bf16_probabilities,
                              M3_DTYPE_BF16, 1U, score_shape,
                              zero_probabilities) &&
            m3_test_transformer_softmax_run(
                &fixture, &scores, &bf16_probabilities, &error),
        "round F32 softmax probabilities to BF16");
    M3_TEST_EXPECT(
        test,
        m3_op_test_u16(&bf16_probabilities)[0] == 0x3e8aU &&
            m3_op_test_u16(&bf16_probabilities)[1] == 0x3f3bU,
        "BF16 probabilities are exactly 0.26953125 and 0.73046875");
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &bf16_query, M3_DTYPE_BF16, 4U,
                          query_shape, bf16_query_values) &&
            m3_op_test_tensor(&fixture, &bf16_key, M3_DTYPE_BF16, 4U,
                              key_shape, bf16_key_values) &&
            m3_op_test_tensor(&fixture, &bf16_value, M3_DTYPE_BF16, 4U,
                              key_shape, bf16_value_values) &&
            m3_op_test_tensor(&fixture, &bf16_output, M3_DTYPE_BF16, 4U,
                              query_shape, bf16_zero) &&
            m3_test_transformer_attention_run(
                &fixture, &bf16_query, &bf16_key, &bf16_value,
                &bf16_output, &error),
        "execute BF16 eager attention fixture");
    M3_TEST_EXPECT(test, m3_op_test_u16(&bf16_output)[0] == 0xc1e8U,
                   "BF16 attention rounds -28.9375 to -29");
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &f16_probabilities, M3_DTYPE_F16, 1U,
                          score_shape, zero_probabilities) &&
            m3_test_transformer_softmax_run(
                &fixture, &scores, &f16_probabilities, &error),
        "round F32 softmax probabilities to F16");
    M3_TEST_EXPECT(
        test,
        m3_op_test_u16(&f16_probabilities)[0] == 0x344eU &&
            m3_op_test_u16(&f16_probabilities)[1] == 0x39d9U,
        "F16 probabilities have the exact eager bits");
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &f16_query, M3_DTYPE_F16, 4U,
                          query_shape, f16_query_values) &&
            m3_op_test_tensor(&fixture, &f16_key, M3_DTYPE_F16, 4U,
                              key_shape, f16_key_values) &&
            m3_op_test_tensor(&fixture, &f16_value, M3_DTYPE_F16, 4U,
                              key_shape, f16_value_values) &&
            m3_op_test_tensor(&fixture, &f16_output, M3_DTYPE_F16, 4U,
                              query_shape, f16_zero) &&
            m3_test_transformer_attention_run(
                &fixture, &f16_query, &f16_key, &f16_value, &f16_output,
                &error),
        "execute F16 eager attention fixture");
    M3_TEST_EXPECT(test, m3_op_test_u16(&f16_output)[0] == 0xd2b9U,
                   "F16 attention rounds -53.7666015625 exactly");
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &f32_query, M3_DTYPE_F32, 4U,
                          query_shape, f32_query_values) &&
            m3_op_test_tensor(&fixture, &f32_key, M3_DTYPE_F32, 4U,
                              key_shape, f32_key_values) &&
            m3_op_test_tensor(&fixture, &f32_value, M3_DTYPE_F32, 4U,
                              key_shape, f32_value_values) &&
            m3_op_test_tensor(&fixture, &f32_output, M3_DTYPE_F32, 4U,
                              query_shape, f32_zero) &&
            m3_test_transformer_attention_run(
                &fixture, &f32_query, &f32_key, &f32_value, &f32_output,
                &error),
        "execute F32 control attention fixture");
    M3_TEST_EXPECT(
        test,
        m3_test_transformer_f32_bits(m3_op_test_f32(&f32_output)[0]) ==
            UINT32_C(0xc1e74605),
        "F32 attention keeps its existing result bits");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_transformer_rms_norm_precision(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 2U};
    const uint64_t scale_shape[] = {2U};
    const uint16_t bf16_input_values[] = {0x3d00U, 0x3f00U};
    const uint16_t bf16_scale_values[] = {0x3f40U, 0x3f40U};
    const uint16_t bf16_zero[] = {0U, 0U};
    const float f32_input_values[] = {0.03125F, 0.5F};
    const float f32_scale_values[] = {0.75F, 0.75F};
    const float f32_zero[] = {0.0F, 0.0F};
    m3_op_test_fixture fixture;
    m3_tensor_view bf16_input;
    m3_tensor_view bf16_scale;
    m3_tensor_view bf16_output;
    m3_tensor_view f32_input;
    m3_tensor_view f32_scale;
    m3_tensor_view f32_output;
    m3_command command;
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create RMSNorm precision fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &bf16_input, M3_DTYPE_BF16, 2U,
                          input_shape, bf16_input_values) &&
            m3_op_test_tensor(&fixture, &bf16_scale, M3_DTYPE_BF16, 1U,
                              scale_shape, bf16_scale_values) &&
            m3_op_test_tensor(&fixture, &bf16_output, M3_DTYPE_BF16, 2U,
                              input_shape, bf16_zero),
        "create BF16 RMSNorm tensors");
    command.kind = M3_OP_RMS_NORM;
    command.descriptor.rms_norm.input = &bf16_input;
    command.descriptor.rms_norm.scale = &bf16_scale;
    command.descriptor.rms_norm.output = &bf16_output;
    command.descriptor.rms_norm.epsilon = 1.0e-6F;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                M3_STATUS_OK &&
            m3_op_test_u16(&bf16_output)[0] == 0x3d88U &&
            m3_op_test_u16(&bf16_output)[1] == 0x3f88U,
        "BF16 RMSNorm rounds normalized values before scaling");
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &f32_input, M3_DTYPE_F32, 2U,
                          input_shape, f32_input_values) &&
            m3_op_test_tensor(&fixture, &f32_scale, M3_DTYPE_F32, 1U,
                              scale_shape, f32_scale_values) &&
            m3_op_test_tensor(&fixture, &f32_output, M3_DTYPE_F32, 2U,
                              input_shape, f32_zero),
        "create F32 RMSNorm control tensors");
    command.descriptor.rms_norm.input = &f32_input;
    command.descriptor.rms_norm.scale = &f32_scale;
    command.descriptor.rms_norm.output = &f32_output;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
            M3_STATUS_OK,
        "execute F32 RMSNorm control");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&f32_output)[0],
                       0.0661619008F, 1.0e-7F, 1.0e-7F,
                       "F32 RMSNorm first channel is unchanged");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&f32_output)[1],
                       1.05859041F, 1.0e-7F, 1.0e-7F,
                       "F32 RMSNorm second channel is unchanged");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_transformer_gated_silu_precision(m3_test_context *test)
{
    const uint64_t shape[] = {1U};
    const uint16_t bf16_gate_value[] = {0x3f00U};
    const uint16_t bf16_up_value[] = {0x3f40U};
    const uint16_t bf16_zero[] = {0U};
    const float f32_gate_value[] = {0.5F};
    const float f32_up_value[] = {0.75F};
    const float f32_zero[] = {0.0F};
    m3_op_test_fixture fixture;
    m3_tensor_view bf16_gate;
    m3_tensor_view bf16_up;
    m3_tensor_view bf16_output;
    m3_tensor_view f32_gate;
    m3_tensor_view f32_up;
    m3_tensor_view f32_output;
    m3_command command;
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create gated SiLU precision fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &bf16_gate, M3_DTYPE_BF16, 1U, shape,
                          bf16_gate_value) &&
            m3_op_test_tensor(&fixture, &bf16_up, M3_DTYPE_BF16, 1U, shape,
                              bf16_up_value) &&
            m3_op_test_tensor(&fixture, &bf16_output, M3_DTYPE_BF16, 1U,
                              shape, bf16_zero),
        "create BF16 gated SiLU tensors");
    command.kind = M3_OP_GATED_SILU;
    command.descriptor.gated_silu.left = &bf16_gate;
    command.descriptor.gated_silu.right = &bf16_up;
    command.descriptor.gated_silu.output = &bf16_output;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                M3_STATUS_OK &&
            m3_op_test_u16(&bf16_output)[0] == 0x3e6eU,
        "BF16 gated SiLU rounds activation before multiplication");
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &f32_gate, M3_DTYPE_F32, 1U, shape,
                          f32_gate_value) &&
            m3_op_test_tensor(&fixture, &f32_up, M3_DTYPE_F32, 1U, shape,
                              f32_up_value) &&
            m3_op_test_tensor(&fixture, &f32_output, M3_DTYPE_F32, 1U,
                              shape, f32_zero),
        "create F32 gated SiLU control tensors");
    command.descriptor.gated_silu.left = &f32_gate;
    command.descriptor.gated_silu.right = &f32_up;
    command.descriptor.gated_silu.output = &f32_output;
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
            M3_STATUS_OK,
        "execute F32 gated SiLU control");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&f32_output)[0],
                       0.233422250F, 1.0e-7F, 1.0e-7F,
                       "F32 gated SiLU is unchanged");
    m3_op_test_fixture_dispose(&fixture);
}
