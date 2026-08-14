/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_dense_test.h"

#include <stdint.h>
#include <string.h>

static m3_command m3_test_metal_rope_command(
    const m3_tensor_view *input, const m3_tensor_view *cosines,
    const m3_tensor_view *sines, m3_tensor_view *output,
    uint64_t position_offset, uint32_t rotary_dimension,
    m3_rope_mode mode)
{
    m3_command command = {0};

    command.kind = M3_OP_ROPE;
    command.descriptor.rope.input = input;
    command.descriptor.rope.cosines = cosines;
    command.descriptor.rope.sines = sines;
    command.descriptor.rope.output = output;
    command.descriptor.rope.position_offset = position_offset;
    command.descriptor.rope.rotary_dimension = rotary_dimension;
    command.descriptor.rope.mode = mode;
    return command;
}

static bool m3_test_metal_rope_strided(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 2U, 6U};
    const uint64_t table_shape[] = {3U, 2U};
    const size_t input_strides[] = {
        26U * sizeof(float), 26U * sizeof(float), 13U * sizeof(float),
        2U * sizeof(float)
    };
    const size_t table_strides[] = {5U * sizeof(float),
                                    2U * sizeof(float)};
    const float input_backing[] = {
        1, 91, 2, 92, 3, 93, 4, 94, 5, 95, 6, 96, 97,
        -1, 81, -2, 82, -3, 83, -4, 84, -5, 85, -6
    };
    const float cosine_backing[] = {
        9, 90, 9, 90, 90, 1, 91, 0, 91, 91, -1, 92, 1
    };
    const float sine_backing[] = {
        9, 90, 9, 90, 90, 0, 91, 1, 91, 91, 0, 92, 0
    };
    const float zeros[12] = {0};
    m3_metal_dense_view input;
    m3_metal_dense_view cosines;
    m3_metal_dense_view sines;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &input, M3_DTYPE_F32, 4U, input_shape,
                       input_strides, sizeof(input_backing),
                       input_backing) &&
                       m3_metal_dense_strided(
                           fixture, &cosines, M3_DTYPE_F32, 2U,
                           table_shape, table_strides,
                           sizeof(cosine_backing), cosine_backing) &&
                       m3_metal_dense_strided(
                           fixture, &sines, M3_DTYPE_F32, 2U,
                           table_shape, table_strides,
                           sizeof(sine_backing), sine_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 4U,
                           input_shape, zeros),
                   "create strided position-offset Metal RoPE");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_rope_command(
        &input.host, &cosines.host, &sines.host, &output.host, 1U, 4U,
        M3_ROPE_HALF_SPLIT);
    metal = m3_test_metal_rope_command(
        &input.metal, &cosines.metal, &sines.metal, &output.metal, 1U, 4U,
        M3_ROPE_HALF_SPLIT);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "half-split Metal RoPE matches host with strided inputs");
    M3_TEST_EXPECT(test,
                   m3_op_test_f32(&output.metal)[4] == 5.0F &&
                       m3_op_test_f32(&output.metal)[5] == 6.0F &&
                       m3_op_test_f32(&output.metal)[10] == -5.0F &&
                       m3_op_test_f32(&output.metal)[11] == -6.0F,
                   "partial Metal RoPE preserves the unrotated tail");
    host.descriptor.rope.mode = M3_ROPE_INTERLEAVED;
    metal.descriptor.rope.mode = M3_ROPE_INTERLEAVED;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "interleaved Metal RoPE matches host with position offset");
    return true;
}

static bool m3_test_metal_rope_low_precision(
    m3_test_context *test, m3_metal_dense_fixture *fixture,
    m3_dtype dtype, const char *message)
{
    const uint64_t input_shape[] = {1U, 2U, 1U, 4U};
    const uint64_t table_shape[] = {1U, 2U};
    const uint16_t f16_input[] = {
        0x3c00U, 0x4000U, 0x4200U, 0x4400U,
        0xbc00U, 0xc000U, 0xc200U, 0xc400U
    };
    const uint16_t f16_cosines[] = {0x3c00U, 0x0000U};
    const uint16_t f16_sines[] = {0x0000U, 0x3c00U};
    const uint16_t bf16_input[] = {
        0x3f80U, 0x4000U, 0x4040U, 0x4080U,
        0xbf80U, 0xc000U, 0xc040U, 0xc080U
    };
    const uint16_t bf16_cosines[] = {0x3f80U, 0x0000U};
    const uint16_t bf16_sines[] = {0x0000U, 0x3f80U};
    const uint16_t zeros[8] = {0U};
    const uint16_t *input_values =
        dtype == M3_DTYPE_F16 ? f16_input : bf16_input;
    const uint16_t *cosine_values =
        dtype == M3_DTYPE_F16 ? f16_cosines : bf16_cosines;
    const uint16_t *sine_values =
        dtype == M3_DTYPE_F16 ? f16_sines : bf16_sines;
    m3_metal_dense_view input;
    m3_metal_dense_view cosines;
    m3_metal_dense_view sines;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &input, dtype, 4U, input_shape,
                       input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &cosines, dtype, 2U, table_shape,
                           cosine_values) &&
                       m3_metal_dense_tensor(
                           fixture, &sines, dtype, 2U, table_shape,
                           sine_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, dtype, 4U, input_shape,
                           zeros),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_rope_command(
        &input.host, &cosines.host, &sines.host, &output.host, 0U, 4U,
        M3_ROPE_HALF_SPLIT);
    metal = m3_test_metal_rope_command(
        &input.metal, &cosines.metal, &sines.metal, &output.metal, 0U, 4U,
        M3_ROPE_HALF_SPLIT);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   message);
    return true;
}

static bool m3_test_metal_rope_alias_rounding(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 1U, 4U};
    const uint64_t table_shape[] = {1U, 1U};
    const float input_values[] = {
        0x1.000002p+0F, 1.0F, 7.0F, 8.0F
    };
    const float cosine_values[] = {0x1.fffffcp-1F};
    const float sine_values[] = {1.0F};
    m3_metal_dense_view input;
    m3_metal_dense_view cosines;
    m3_metal_dense_view sines;
    m3_command host;
    m3_command metal;
    m3_error error;
    uint32_t bits;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &input, M3_DTYPE_F32, 4U, input_shape,
                       input_values) &&
                       m3_metal_dense_tensor(
                           fixture, &cosines, M3_DTYPE_F32, 2U,
                           table_shape, cosine_values) &&
                       m3_metal_dense_tensor(
                           fixture, &sines, M3_DTYPE_F32, 2U,
                           table_shape, sine_values),
                   "create exact-alias FMA-sensitive Metal RoPE");
    if (input.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_rope_command(
        &input.host, &cosines.host, &sines.host, &input.host, 0U, 2U,
        M3_ROPE_HALF_SPLIT);
    metal = m3_test_metal_rope_command(
        &input.metal, &cosines.metal, &sines.metal, &input.metal, 0U, 2U,
        M3_ROPE_HALF_SPLIT);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&input),
                   "Metal RoPE permits an exact input alias");
    (void)memcpy(&bits, m3_op_test_f32(&input.metal), sizeof(bits));
    M3_TEST_EXPECT(test, bits == UINT32_C(0xa8800000),
                   "Metal RoPE matches exact host contracted arithmetic");
    M3_TEST_EXPECT(test,
                   m3_op_test_f32(&input.metal)[2] == 7.0F &&
                       m3_op_test_f32(&input.metal)[3] == 8.0F,
                   "exact-alias partial Metal RoPE preserves its tail");
    return true;
}

void m3_test_metal_dense_rope(m3_test_context *test)
{
    m3_metal_dense_fixture fixture;

    if (!m3_metal_dense_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_rope_strided(test, &fixture) ||
        !m3_test_metal_rope_low_precision(
            test, &fixture, M3_DTYPE_F16,
            "F16 Metal RoPE matches host exactly") ||
        !m3_test_metal_rope_low_precision(
            test, &fixture, M3_DTYPE_BF16,
            "BF16 Metal RoPE matches host exactly") ||
        !m3_test_metal_rope_alias_rounding(test, &fixture)) {
        m3_metal_dense_fixture_dispose(&fixture);
        return;
    }
    m3_metal_dense_fixture_dispose(&fixture);
}
