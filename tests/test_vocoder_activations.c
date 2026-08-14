/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint32_t m3_activation_f32_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float m3_activation_f32_from_bits(uint32_t bits)
{
    float value;

    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

static void m3_activation_snake_command(
    m3_command *command, const m3_tensor_view *input,
    const m3_tensor_view *alpha, m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_SNAKE1D;
    command->descriptor.snake1d.input = input;
    command->descriptor.snake1d.alpha = alpha;
    command->descriptor.snake1d.output = output;
}

static void m3_activation_tanh_command(m3_command *command,
                                       const m3_tensor_view *input,
                                       m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_TANH;
    command->descriptor.tanh.input = input;
    command->descriptor.tanh.output = output;
}

static bool m3_activation_strided_tensor(
    m3_op_test_fixture *fixture, m3_tensor_view *view, uint8_t rank,
    const uint64_t *shape, const size_t *strides, const float *values,
    size_t byte_count)
{
    m3_storage *storage = NULL;
    m3_error error;

    m3_tensor_view_init(view);
    return m3_op_test_storage(fixture, byte_count, &storage) &&
           m3_storage_write(storage, 0U, values, byte_count, &error) ==
               M3_STATUS_OK &&
           m3_tensor_view_strided(view, storage, M3_DTYPE_F32, rank, shape,
                                  strides, 0U, &error) == M3_STATUS_OK;
}

static void m3_test_snake_order_and_strides(m3_test_context *test)
{
    const uint64_t discriminator_shape[] = {1U, 2U, 1U};
    const float discriminator_input[] = {
        m3_activation_f32_from_bits(UINT32_C(0x3e1d90cd)), 1.0F
    };
    const float discriminator_alpha[] = {
        m3_activation_f32_from_bits(UINT32_C(0x3e0b9eef)),
        m3_activation_f32_from_bits(UINT32_C(0xb089705f))
    };
    const float zeros[] = {0.0F, 0.0F};
    const uint64_t shape[] = {1U, 2U, 3U};
    const size_t input_strides[] = {48U, 24U, 8U};
    const size_t alpha_strides[] = {16U, 8U, 4U};
    const float input_backing[] = {
        0.25F, -99.0F, -2.0F, -99.0F, 4.0F, -99.0F,
        0.5F, -99.0F, 1.0F, -99.0F, 2.0F
    };
    const float alpha_backing[] = {
        0.0F, -99.0F,
        m3_activation_f32_from_bits(UINT32_C(0xb089705f))
    };
    const float output_zeros[6] = {0};
    const float first_channel_expected[] = {0.25F, -2.0F, 4.0F};
    const uint64_t alias_shape[] = {1U, 1U, 3U};
    const float alias_values[] = {1.0F, -2.0F, 3.0F};
    const float zero_alpha = 0.0F;
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view alpha;
    m3_tensor_view output;
    m3_tensor_view strided_input;
    m3_tensor_view strided_alpha;
    m3_tensor_view strided_output;
    m3_tensor_view alias_input;
    m3_tensor_view alias_alpha;
    m3_command command;
    m3_error error;
    size_t scratch_bytes = SIZE_MAX;
    volatile float denominator = discriminator_alpha[1] +
                                 0x1.12e0bep-30F;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create host waveform activation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_activation_f32_bits(0x1.12e0bep-30F) ==
                           UINT32_C(0x3089705f) &&
                       m3_activation_f32_bits(denominator) == 0U,
                   "Snake1d locks exact epsilon bits and +zero denominator");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 3U,
                                     discriminator_shape,
                                     discriminator_input) &&
                       m3_op_test_tensor(&fixture, &alpha, M3_DTYPE_F32, 3U,
                                         discriminator_shape,
                                         discriminator_alpha) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 3U,
                                         discriminator_shape, zeros),
                   "create Snake1d order discriminator tensors");
    m3_activation_snake_command(&command, &input, &alpha, &output);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U,
                                      &scratch_bytes, &error) ==
                           M3_STATUS_OK &&
                       scratch_bytes == 0U &&
                       m3_activation_f32_bits(
                           m3_op_test_f32(&output)[0]) ==
                           UINT32_C(0x3e20def4) &&
                       m3_activation_f32_bits(
                           m3_op_test_f32(&output)[1]) ==
                           UINT32_C(0x7f800000),
                   "Snake1d preserves seven F32 statements and +infinity discriminator");
    M3_TEST_EXPECT(test,
                   m3_activation_strided_tensor(
                       &fixture, &strided_input, 3U, shape, input_strides,
                       input_backing, sizeof(input_backing)) &&
                       m3_activation_strided_tensor(
                           &fixture, &strided_alpha, 3U,
                           discriminator_shape, alpha_strides,
                           alpha_backing, sizeof(alpha_backing)) &&
                       m3_op_test_tensor(
                           &fixture, &strided_output, M3_DTYPE_F32, 3U,
                           shape, output_zeros),
                   "create strided Snake1d input and channel alpha");
    m3_activation_snake_command(&command, &strided_input, &strided_alpha,
                                &strided_output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       memcmp(m3_op_test_f32(&strided_output),
                              first_channel_expected,
                              sizeof(first_channel_expected)) == 0 &&
                       m3_activation_f32_bits(
                           m3_op_test_f32(&strided_output)[3]) ==
                           UINT32_C(0x7f800000) &&
                       m3_activation_f32_bits(
                           m3_op_test_f32(&strided_output)[4]) ==
                           UINT32_C(0x7f800000) &&
                       m3_activation_f32_bits(
                           m3_op_test_f32(&strided_output)[5]) ==
                           UINT32_C(0x7f800000),
                   "Snake1d broadcasts valid strided alpha by channel");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &alias_input, M3_DTYPE_F32,
                                     3U, alias_shape, alias_values) &&
                       m3_op_test_tensor(&fixture, &alias_alpha,
                                         M3_DTYPE_F32, 3U,
                                         (const uint64_t[]){1U, 1U, 1U},
                                         &zero_alpha),
                   "create exact-alias Snake1d tensors");
    m3_activation_snake_command(&command, &alias_input, &alias_alpha,
                                &alias_input);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       memcmp(m3_op_test_f32(&alias_input), alias_values,
                              sizeof(alias_values)) == 0,
                   "Snake1d supports exact input-output aliasing");
    m3_op_test_fixture_dispose(&fixture);
}

static void m3_test_tanh_semantics(m3_test_context *test)
{
    const uint64_t shape[] = {8U};
    const float values[] = {
        0.0F, -0.0F, 1.0F, -1.0F, INFINITY, -INFINITY,
        m3_activation_f32_from_bits(UINT32_C(0x7fc12345)), 0.5F
    };
    const float zeros[8] = {0};
    const uint32_t expected[] = {
        UINT32_C(0x00000000), UINT32_C(0x80000000),
        UINT32_C(0x3f42f7d6), UINT32_C(0xbf42f7d6),
        UINT32_C(0x3f800000), UINT32_C(0xbf800000), 0U,
        UINT32_C(0x3eec9a9f)
    };
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view output;
    m3_command command;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create host tanh semantics fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 1U,
                                     shape, values) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 1U,
                                         shape, zeros),
                   "create tanh special-value tensors");
    m3_activation_tanh_command(&command, &input, &output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "execute host tanh oracle");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_activation_f32_bits(
                           m3_op_test_f32(&output)[index]) == expected[index],
                       "tanh locks signed zero, finite, and infinity bits");
    }
    M3_TEST_EXPECT(test, isnan(m3_op_test_f32(&output)[6]),
                   "tanh preserves NaN classification");
    M3_TEST_EXPECT(test,
                   m3_activation_f32_bits(m3_op_test_f32(&output)[7]) ==
                       expected[7],
                   "tanh locks the finite half-value host oracle bits");
    m3_activation_tanh_command(&command, &input, &input);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       m3_activation_f32_bits(m3_op_test_f32(&input)[1]) ==
                           UINT32_C(0x80000000),
                   "tanh supports exact in-place execution");
    m3_op_test_fixture_dispose(&fixture);
}

static void m3_test_activation_contracts(m3_test_context *test)
{
    const uint64_t shape[] = {1U, 2U, 1U};
    const uint64_t bad_alpha_shape[] = {1U, 1U, 1U};
    const float values[] = {1.0F, 2.0F};
    const float alphas[] = {1.0F, 1.0F};
    const float sentinel[] = {-7.0F, -7.0F};
    const float scalar = 1.0F;
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view alpha;
    m3_tensor_view bad_alpha;
    m3_tensor_view output;
    m3_tensor_view scalar_input;
    m3_command command;
    m3_command commands[2];
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create activation validation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 3U,
                                     shape, values) &&
                       m3_op_test_tensor(&fixture, &alpha, M3_DTYPE_F32, 3U,
                                         shape, alphas) &&
                       m3_op_test_tensor(&fixture, &bad_alpha, M3_DTYPE_F32,
                                         3U, bad_alpha_shape, alphas) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 3U,
                                         shape, sentinel) &&
                       m3_op_test_tensor(&fixture, &scalar_input,
                                         M3_DTYPE_F32, 0U, NULL, &scalar),
                   "create activation validation tensors");
    m3_activation_snake_command(&command, &input, &bad_alpha, &output);
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "Snake1d rejects a non-channel alpha shape");
    m3_activation_snake_command(&command, &input, &alpha, &alpha);
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "Snake1d rejects alpha-output overlap even when exact");
    m3_activation_tanh_command(&command, &scalar_input, &scalar_input);
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "tanh rejects rank-zero tensors");
    m3_activation_tanh_command(&commands[0], &input, &output);
    m3_activation_snake_command(&commands[1], &input, &alpha, NULL);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, commands, 2U, NULL,
                                      &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       memcmp(m3_op_test_f32(&output), sentinel,
                              sizeof(sentinel)) == 0,
                   "null Snake1d sink rejects the whole list before writes");
    m3_op_test_fixture_dispose(&fixture);
}

static void m3_test_activation_empty_limits(m3_test_context *test)
{
    const uint64_t snake_shape[] = {2U, 0U, 4U};
    const uint64_t alpha_shape[] = {1U, 0U, 1U};
    const uint64_t tanh_shape[] = {
        0U, (uint64_t)UINT32_MAX + 1U, 1U, 1U, 1U, 1U, 1U, 1U
    };
    m3_op_test_fixture fixture;
    m3_tensor_view snake_input;
    m3_tensor_view alpha;
    m3_tensor_view snake_output;
    m3_tensor_view tanh_input;
    m3_tensor_view tanh_output;
    m3_command commands[2];
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create empty activation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &snake_input, M3_DTYPE_F32,
                                     3U, snake_shape, NULL) &&
                       m3_op_test_tensor(&fixture, &alpha, M3_DTYPE_F32, 3U,
                                         alpha_shape, NULL) &&
                       m3_op_test_tensor(&fixture, &snake_output,
                                         M3_DTYPE_F32, 3U, snake_shape,
                                         NULL) &&
                       m3_op_test_tensor(&fixture, &tanh_input,
                                         M3_DTYPE_F32, 8U, tanh_shape,
                                         NULL) &&
                       m3_op_test_tensor(&fixture, &tanh_output,
                                         M3_DTYPE_F32, 8U, tanh_shape,
                                         NULL),
                   "create empty and rank-limit activation tensors");
    m3_activation_snake_command(&commands[0], &snake_input, &alpha,
                                &snake_output);
    m3_activation_tanh_command(&commands[1], &tanh_input, &tanh_output);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &before, &error) == M3_STATUS_OK &&
                       m3_backend_execute(fixture.backend, commands, 2U, NULL,
                                          &error) == M3_STATUS_OK &&
                       m3_backend_get_allocation_stats(
                           fixture.backend, &after, &error) == M3_STATUS_OK &&
                       before.live_storage_count == after.live_storage_count &&
                       before.live_allocated_bytes == after.live_allocated_bytes,
                   "empty activation work is a stats-neutral no-op");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_vocoder_activations(m3_test_context *test)
{
    m3_test_snake_order_and_strides(test);
    m3_test_tanh_semantics(test);
    m3_test_activation_contracts(test);
    m3_test_activation_empty_limits(test);
}
