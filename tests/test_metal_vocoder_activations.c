/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Darwin libm and Metal precise transcendental implementations may round
 * finite values differently; signed zero, infinity, and NaN classification
 * remain exact gates, while finite parity is bounded in F32. */
#define M3_METAL_PRECISE_TRANSCENDENTAL_TOLERANCE 0x1p-19F

typedef struct {
    m3_op_test_fixture host;
    m3_op_test_fixture metal;
} m3_metal_activation_pair;

static uint32_t m3_metal_activation_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float m3_metal_activation_from_bits(uint32_t bits)
{
    float value;

    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool m3_metal_activation_pair_init(
    m3_test_context *test, m3_metal_activation_pair *pair)
{
    m3_error error;
    m3_status status;

    (void)memset(pair, 0, sizeof(*pair));
    if (!m3_op_test_fixture_init(&pair->host)) {
        M3_TEST_EXPECT(test, false, "create activation host oracle");
        return false;
    }
    status = m3_backend_create_metal(&pair->metal.backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        m3_op_test_fixture_dispose(&pair->host);
        M3_TEST_SKIP(test, m3_error_message(&error));
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "compile aggregate Metal activation library");
    if (status != M3_STATUS_OK) {
        m3_op_test_fixture_dispose(&pair->host);
        return false;
    }
    return true;
}

static void m3_metal_activation_pair_dispose(
    m3_metal_activation_pair *pair)
{
    m3_op_test_fixture_dispose(&pair->metal);
    m3_op_test_fixture_dispose(&pair->host);
}

static bool m3_metal_activation_pair_tensor(
    m3_metal_activation_pair *pair, m3_tensor_view *host,
    m3_tensor_view *metal, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, const void *values)
{
    return m3_op_test_tensor(&pair->host, host, dtype, rank, shape, values) &&
           m3_op_test_tensor(&pair->metal, metal, dtype, rank, shape, values);
}

static bool m3_metal_activation_strided_one(
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

static bool m3_metal_activation_pair_strided(
    m3_metal_activation_pair *pair, m3_tensor_view *host,
    m3_tensor_view *metal, uint8_t rank, const uint64_t *shape,
    const size_t *strides, const float *values, size_t byte_count)
{
    return m3_metal_activation_strided_one(
               &pair->host, host, rank, shape, strides, values, byte_count) &&
           m3_metal_activation_strided_one(
               &pair->metal, metal, rank, shape, strides, values, byte_count);
}

static void m3_metal_snake_command(
    m3_command *command, const m3_tensor_view *input,
    const m3_tensor_view *alpha, m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_SNAKE1D;
    command->descriptor.snake1d.input = input;
    command->descriptor.snake1d.alpha = alpha;
    command->descriptor.snake1d.output = output;
}

static void m3_metal_tanh_command(m3_command *command,
                                  const m3_tensor_view *input,
                                  m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_TANH;
    command->descriptor.tanh.input = input;
    command->descriptor.tanh.output = output;
}

static bool m3_metal_activation_close(float actual, float expected)
{
    return m3_test_f32_close(
        actual, expected, M3_METAL_PRECISE_TRANSCENDENTAL_TOLERANCE,
        M3_METAL_PRECISE_TRANSCENDENTAL_TOLERANCE);
}

static void m3_test_metal_activation_precision(m3_test_context *test)
{
    const uint64_t shape[] = {1U, 2U, 4U};
    const uint64_t alpha_shape[] = {1U, 2U, 1U};
    const float inputs[] = {
        m3_metal_activation_from_bits(UINT32_C(0x3e1d90cd)),
        0.5F, -0.75F, 2.0F, 1.0F, 0.25F, -1.0F, 3.0F
    };
    const float alphas[] = {
        m3_metal_activation_from_bits(UINT32_C(0x3e0b9eef)), 1.25F
    };
    const float zeros[8] = {0};
    const uint64_t one_shape[] = {1U, 1U, 1U};
    const float one = 1.0F;
    const float negative_epsilon =
        m3_metal_activation_from_bits(UINT32_C(0xb089705f));
    const float zero = 0.0F;
    m3_metal_activation_pair pair;
    m3_tensor_view host_input;
    m3_tensor_view metal_input;
    m3_tensor_view host_alpha;
    m3_tensor_view metal_alpha;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    m3_tensor_view host_one;
    m3_tensor_view metal_one;
    m3_tensor_view host_negative_epsilon;
    m3_tensor_view metal_negative_epsilon;
    m3_tensor_view host_special_output;
    m3_tensor_view metal_special_output;
    m3_command host_command;
    m3_command metal_command;
    m3_error error;
    size_t index;
    bool created;

    if (!m3_metal_activation_pair_init(test, &pair)) {
        return;
    }
    created = m3_metal_activation_pair_tensor(
                  &pair, &host_input, &metal_input, M3_DTYPE_F32, 3U,
                  shape, inputs) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_alpha, &metal_alpha, M3_DTYPE_F32, 3U,
                  alpha_shape, alphas) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_output, &metal_output, M3_DTYPE_F32, 3U,
                  shape, zeros);
    M3_TEST_EXPECT(test, created, "create Metal Snake1d precision pair");
    if (!created) {
        m3_metal_activation_pair_dispose(&pair);
        return;
    }
    m3_metal_snake_command(&host_command, &host_input, &host_alpha,
                           &host_output);
    m3_metal_snake_command(&metal_command, &metal_input, &metal_alpha,
                           &metal_output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, &host_command, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, &metal_command,
                                          1U, NULL, &error) == M3_STATUS_OK,
                   "execute ordered Host and Metal Snake1d");
    M3_TEST_EXPECT(test,
                   m3_metal_activation_bits(
                       m3_op_test_f32(&metal_output)[0]) ==
                       UINT32_C(0x3e20def4),
                   "Metal Snake1d rejects a contracted final multiply-add bit");
    for (index = 0U; index < 8U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_metal_activation_close(
                           m3_op_test_f32(&metal_output)[index],
                           m3_op_test_f32(&host_output)[index]),
                       "Metal precise sin matches the finite F32 tolerance");
    }
    m3_metal_snake_command(&host_command, &host_output, &host_alpha,
                           &host_output);
    m3_metal_snake_command(&metal_command, &metal_output, &metal_alpha,
                           &metal_output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, &host_command, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, &metal_command,
                                          1U, NULL, &error) == M3_STATUS_OK,
                   "execute exact-alias Snake1d on Host and Metal");
    for (index = 0U; index < 8U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_metal_activation_close(
                           m3_op_test_f32(&metal_output)[index],
                           m3_op_test_f32(&host_output)[index]),
                       "in-place Metal Snake1d matches host tolerance");
    }
    created = m3_metal_activation_pair_tensor(
                  &pair, &host_one, &metal_one, M3_DTYPE_F32, 3U,
                  one_shape, &one) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_negative_epsilon, &metal_negative_epsilon,
                  M3_DTYPE_F32, 3U, one_shape, &negative_epsilon) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_special_output, &metal_special_output,
                  M3_DTYPE_F32, 3U, one_shape, &zero);
    M3_TEST_EXPECT(test, created,
                   "create Metal exact-epsilon discriminator pair");
    m3_metal_snake_command(&host_command, &host_one,
                           &host_negative_epsilon, &host_special_output);
    m3_metal_snake_command(&metal_command, &metal_one,
                           &metal_negative_epsilon, &metal_special_output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, &host_command, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, &metal_command,
                                          1U, NULL, &error) == M3_STATUS_OK &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&host_special_output)[0]) ==
                           UINT32_C(0x7f800000) &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&metal_special_output)[0]) ==
                           UINT32_C(0x7f800000),
                   "negative exact epsilon makes a +zero denominator and +infinity");
    m3_metal_activation_pair_dispose(&pair);
}

static void m3_test_metal_activation_specials(m3_test_context *test)
{
    const uint64_t snake_shape[] = {1U, 1U, 5U};
    const uint64_t alpha_shape[] = {1U, 1U, 1U};
    const float snake_inputs[] = {
        0.0F, -0.0F, INFINITY, -INFINITY,
        m3_metal_activation_from_bits(UINT32_C(0x7fc12345))
    };
    const float alpha = 1.0F;
    const float snake_zeros[5] = {0};
    const uint64_t tanh_shape[] = {7U};
    const float tanh_inputs[] = {
        0.0F, -0.0F, 1.0F, -1.0F, INFINITY, -INFINITY,
        m3_metal_activation_from_bits(UINT32_C(0x7fc12345))
    };
    const float tanh_zeros[7] = {0};
    m3_metal_activation_pair pair;
    m3_tensor_view host_snake_input;
    m3_tensor_view metal_snake_input;
    m3_tensor_view host_alpha;
    m3_tensor_view metal_alpha;
    m3_tensor_view host_snake_output;
    m3_tensor_view metal_snake_output;
    m3_tensor_view host_tanh_input;
    m3_tensor_view metal_tanh_input;
    m3_tensor_view host_tanh_output;
    m3_tensor_view metal_tanh_output;
    m3_command host_command;
    m3_command metal_command;
    m3_error error;
    size_t index;
    bool created;

    if (!m3_metal_activation_pair_init(test, &pair)) {
        return;
    }
    created = m3_metal_activation_pair_tensor(
                  &pair, &host_snake_input, &metal_snake_input,
                  M3_DTYPE_F32, 3U, snake_shape, snake_inputs) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_alpha, &metal_alpha, M3_DTYPE_F32, 3U,
                  alpha_shape, &alpha) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_snake_output, &metal_snake_output,
                  M3_DTYPE_F32, 3U, snake_shape, snake_zeros);
    M3_TEST_EXPECT(test, created, "create Snake1d special-value pair");
    if (!created) {
        m3_metal_activation_pair_dispose(&pair);
        return;
    }
    m3_metal_snake_command(&host_command, &host_snake_input, &host_alpha,
                           &host_snake_output);
    m3_metal_snake_command(&metal_command, &metal_snake_input, &metal_alpha,
                           &metal_snake_output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, &host_command, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, &metal_command,
                                          1U, NULL, &error) == M3_STATUS_OK &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&metal_snake_output)[0]) == 0U &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&metal_snake_output)[1]) == 0U,
                   "Metal Snake1d locks ordered signed-zero results");
    for (index = 2U; index < 5U; ++index) {
        M3_TEST_EXPECT(test,
                       isnan(m3_op_test_f32(&host_snake_output)[index]) &&
                           isnan(m3_op_test_f32(&metal_snake_output)[index]),
                       "Snake1d preserves non-finite NaN classification");
    }
    created = m3_metal_activation_pair_tensor(
                  &pair, &host_tanh_input, &metal_tanh_input, M3_DTYPE_F32,
                  1U, tanh_shape, tanh_inputs) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_tanh_output, &metal_tanh_output,
                  M3_DTYPE_F32, 1U, tanh_shape, tanh_zeros);
    M3_TEST_EXPECT(test, created, "create tanh special-value pair");
    m3_metal_tanh_command(&host_command, &host_tanh_input,
                          &host_tanh_output);
    m3_metal_tanh_command(&metal_command, &metal_tanh_input,
                          &metal_tanh_output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, &host_command, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, &metal_command,
                                          1U, NULL, &error) == M3_STATUS_OK,
                   "execute Host and Metal tanh special values");
    M3_TEST_EXPECT(test,
                   m3_metal_activation_bits(
                       m3_op_test_f32(&metal_tanh_output)[0]) == 0U &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&metal_tanh_output)[1]) ==
                           UINT32_C(0x80000000) &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&metal_tanh_output)[4]) ==
                           UINT32_C(0x3f800000) &&
                       m3_metal_activation_bits(
                           m3_op_test_f32(&metal_tanh_output)[5]) ==
                           UINT32_C(0xbf800000),
                   "Metal tanh locks signed zero and infinity bits");
    M3_TEST_EXPECT(test,
                   m3_metal_activation_close(
                       m3_op_test_f32(&metal_tanh_output)[2],
                       m3_op_test_f32(&host_tanh_output)[2]) &&
                       m3_metal_activation_close(
                           m3_op_test_f32(&metal_tanh_output)[3],
                           m3_op_test_f32(&host_tanh_output)[3]),
                   "Metal precise tanh matches finite host libm tolerance");
    M3_TEST_EXPECT(test,
                   isnan(m3_op_test_f32(&host_tanh_output)[6]) &&
                       isnan(m3_op_test_f32(&metal_tanh_output)[6]),
                   "Host and Metal tanh preserve NaN classification only");
    m3_metal_activation_pair_dispose(&pair);
}

static void m3_test_metal_activation_strides_dependencies(
    m3_test_context *test)
{
    const uint64_t shape[] = {1U, 2U, 2U};
    const uint64_t alpha_shape[] = {1U, 2U, 1U};
    const size_t input_strides[] = {32U, 16U, 8U};
    const size_t alpha_strides[] = {16U, 8U, 4U};
    const float input_backing[] = {
        0.25F, -9.0F, -0.5F, -9.0F, 1.0F, -9.0F, 2.0F
    };
    const float alpha_backing[] = {0.5F, -9.0F, 1.25F};
    const float zeros[4] = {0};
    const float right_values[] = {0.5F, 1.0F, -0.25F, 2.0F};
    m3_metal_activation_pair pair;
    m3_tensor_view host_input;
    m3_tensor_view metal_input;
    m3_tensor_view host_alpha;
    m3_tensor_view metal_alpha;
    m3_tensor_view host_snake;
    m3_tensor_view metal_snake;
    m3_tensor_view host_tanh;
    m3_tensor_view metal_tanh;
    m3_tensor_view host_right;
    m3_tensor_view metal_right;
    m3_command host_commands[3];
    m3_command metal_commands[3];
    m3_error error;
    size_t index;
    bool created;

    if (!m3_metal_activation_pair_init(test, &pair)) {
        return;
    }
    created = m3_metal_activation_pair_strided(
                  &pair, &host_input, &metal_input, 3U, shape,
                  input_strides, input_backing, sizeof(input_backing)) &&
              m3_metal_activation_pair_strided(
                  &pair, &host_alpha, &metal_alpha, 3U, alpha_shape,
                  alpha_strides, alpha_backing, sizeof(alpha_backing)) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_snake, &metal_snake, M3_DTYPE_F32, 3U,
                  shape, zeros) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_tanh, &metal_tanh, M3_DTYPE_F32, 3U,
                  shape, zeros) &&
              m3_metal_activation_pair_tensor(
                  &pair, &host_right, &metal_right, M3_DTYPE_F32, 3U,
                  shape, right_values);
    M3_TEST_EXPECT(test, created,
                   "create strided activation dependency tensors");
    if (!created) {
        m3_metal_activation_pair_dispose(&pair);
        return;
    }
    m3_metal_tanh_command(&host_commands[0], &host_input, &host_tanh);
    m3_metal_tanh_command(&metal_commands[0], &metal_input, &metal_tanh);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, host_commands, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, metal_commands,
                                          1U, NULL, &error) == M3_STATUS_OK,
                   "execute tanh from valid strided Host and Metal input");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_metal_activation_close(
                           m3_op_test_f32(&metal_tanh)[index],
                           m3_op_test_f32(&host_tanh)[index]),
                       "strided-input Metal tanh matches host tolerance");
    }
    m3_metal_snake_command(&host_commands[0], &host_input, &host_alpha,
                           &host_snake);
    m3_metal_snake_command(&metal_commands[0], &metal_input, &metal_alpha,
                           &metal_snake);
    m3_metal_tanh_command(&host_commands[1], &host_snake, &host_tanh);
    m3_metal_tanh_command(&metal_commands[1], &metal_snake, &metal_tanh);
    host_commands[2].kind = M3_OP_ADD;
    host_commands[2].descriptor.add.left = &host_tanh;
    host_commands[2].descriptor.add.right = &host_right;
    host_commands[2].descriptor.add.output = &host_tanh;
    metal_commands[2].kind = M3_OP_ADD;
    metal_commands[2].descriptor.add.left = &metal_tanh;
    metal_commands[2].descriptor.add.right = &metal_right;
    metal_commands[2].descriptor.add.output = &metal_tanh;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, host_commands, 3U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, metal_commands,
                                          3U, NULL, &error) == M3_STATUS_OK,
                   "execute strided Snake1d-to-tanh GPU dependency chain");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_metal_activation_close(
                           m3_op_test_f32(&metal_tanh)[index],
                           m3_op_test_f32(&host_tanh)[index]),
                       "GPU activation chain matches ordered host oracle");
    }
    m3_metal_tanh_command(&host_commands[0], &host_tanh, &host_tanh);
    m3_metal_tanh_command(&metal_commands[0], &metal_tanh, &metal_tanh);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(pair.host.backend, host_commands, 1U,
                                      NULL, &error) == M3_STATUS_OK &&
                       m3_backend_execute(pair.metal.backend, metal_commands,
                                          1U, NULL, &error) == M3_STATUS_OK,
                   "execute exact-alias tanh on Host and Metal");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_metal_activation_close(
                           m3_op_test_f32(&metal_tanh)[index],
                           m3_op_test_f32(&host_tanh)[index]),
                       "in-place Metal tanh matches host tolerance");
    }
    m3_metal_activation_pair_dispose(&pair);
}

static void m3_test_metal_activation_writer_atomicity(
    m3_test_context *test)
{
    const uint64_t shape[] = {1U, 1U, 2U};
    const uint64_t alpha_shape[] = {1U, 1U, 1U};
    const float inputs[] = {0.5F, 1.0F};
    const float alpha = 1.0F;
    const float sentinel[] = {-7.0F, -7.0F};
    const int32_t index_sentinel[] = {-9, -9};
    const uint64_t empty_shape[] = {1U, 0U, 2U};
    const uint64_t empty_alpha_shape[] = {1U, 0U, 1U};
    m3_op_test_fixture fixture = {0};
    m3_tensor_view input;
    m3_tensor_view alpha_view;
    m3_tensor_view output;
    m3_tensor_view indices;
    m3_tensor_view empty_input;
    m3_tensor_view empty_alpha;
    m3_tensor_view empty_output;
    m3_command commands[2];
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_error error;
    m3_status status = m3_backend_create_metal(&fixture.backend, &error);
    bool created;

    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal writer-tracking fixture");
    if (status != M3_STATUS_OK) {
        return;
    }
    created = m3_op_test_tensor(&fixture, &input, M3_DTYPE_F32, 3U,
                                shape, inputs) &&
              m3_op_test_tensor(&fixture, &alpha_view, M3_DTYPE_F32, 3U,
                                alpha_shape, &alpha) &&
              m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 3U,
                                shape, sentinel) &&
              m3_op_test_tensor(&fixture, &indices, M3_DTYPE_I32, 3U,
                                shape, index_sentinel);
    M3_TEST_EXPECT(test, created, "create Metal writer-tracking tensors");
    if (!created) {
        m3_op_test_fixture_dispose(&fixture);
        return;
    }
    m3_metal_snake_command(&commands[0], &input, &alpha_view, &output);
    commands[1].kind = M3_OP_CAST;
    commands[1].descriptor.cast.input = &output;
    commands[1].descriptor.cast.output = &indices;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, commands, 2U, NULL,
                                      &error) == M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&output), sentinel,
                              sizeof(sentinel)) == 0 &&
                       memcmp(m3_op_test_i32(&indices), index_sentinel,
                              sizeof(index_sentinel)) == 0,
                   "Snake1d writer tracking rejects stale-host cast atomically");
    m3_metal_tanh_command(&commands[0], &input, &output);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, commands, 2U, NULL,
                                      &error) == M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&output), sentinel,
                              sizeof(sentinel)) == 0,
                   "tanh writer tracking rejects stale-host cast atomically");
    created = m3_op_test_tensor(&fixture, &empty_input, M3_DTYPE_F32, 3U,
                                empty_shape, NULL) &&
              m3_op_test_tensor(&fixture, &empty_alpha, M3_DTYPE_F32, 3U,
                                empty_alpha_shape, NULL) &&
              m3_op_test_tensor(&fixture, &empty_output, M3_DTYPE_F32, 3U,
                                empty_shape, NULL);
    M3_TEST_EXPECT(test, created, "create empty Metal activation tensors");
    m3_metal_snake_command(&commands[0], &empty_input, &empty_alpha,
                           &empty_output);
    m3_metal_tanh_command(&commands[1], &empty_output, &empty_output);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &before, &error) == M3_STATUS_OK &&
                       m3_backend_execute(fixture.backend, commands, 2U, NULL,
                                          &error) == M3_STATUS_OK &&
                       m3_backend_get_allocation_stats(
                           fixture.backend, &after, &error) == M3_STATUS_OK &&
                       before.live_storage_count == after.live_storage_count &&
                       before.live_allocated_bytes == after.live_allocated_bytes,
                   "empty Metal activations dispatch no work and preserve stats");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_metal_vocoder_activations(m3_test_context *test)
{
    m3_test_metal_activation_precision(test);
    m3_test_metal_activation_specials(test);
    m3_test_metal_activation_strides_dependencies(test);
    m3_test_metal_activation_writer_atomicity(test);
}
