/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    m3_op_test_fixture host;
    m3_op_test_fixture metal;
} m3_metal_softmax_pair;

static bool m3_metal_softmax_pair_init(m3_test_context *test,
                                       m3_metal_softmax_pair *pair)
{
    m3_error error;
    m3_status status;

    (void)memset(pair, 0, sizeof(*pair));
    if (!m3_op_test_fixture_init(&pair->host)) {
        M3_TEST_EXPECT(test, false, "create host softmax oracle");
        return false;
    }
    status = m3_backend_create_metal(&pair->metal.backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        m3_op_test_fixture_dispose(&pair->host);
        M3_TEST_SKIP(test, m3_error_message(&error));
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal softmax fixture");
    if (status != M3_STATUS_OK) {
        m3_op_test_fixture_dispose(&pair->host);
        return false;
    }
    return true;
}

static void m3_metal_softmax_pair_dispose(m3_metal_softmax_pair *pair)
{
    m3_op_test_fixture_dispose(&pair->metal);
    m3_op_test_fixture_dispose(&pair->host);
}

static bool m3_metal_softmax_pair_tensor(
    m3_metal_softmax_pair *pair, m3_tensor_view *host,
    m3_tensor_view *metal, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, const void *values)
{
    return m3_op_test_tensor(&pair->host, host, dtype, rank, shape,
                             values) &&
           m3_op_test_tensor(&pair->metal, metal, dtype, rank, shape,
                             values);
}

static bool m3_metal_softmax_strided_one(
    m3_op_test_fixture *fixture, m3_tensor_view *view, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, const size_t *strides,
    const void *values, size_t byte_count)
{
    m3_storage *storage = NULL;
    m3_error error;

    m3_tensor_view_init(view);
    return m3_op_test_storage(fixture, byte_count, &storage) &&
           m3_storage_write(storage, 0U, values, byte_count, &error) ==
               M3_STATUS_OK &&
           m3_tensor_view_strided(view, storage, dtype, rank, shape,
                                  strides, 0U, &error) == M3_STATUS_OK;
}

static bool m3_metal_softmax_pair_strided(
    m3_metal_softmax_pair *pair, m3_tensor_view *host,
    m3_tensor_view *metal, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, const size_t *strides, const void *values,
    size_t byte_count)
{
    return m3_metal_softmax_strided_one(
               &pair->host, host, dtype, rank, shape, strides, values,
               byte_count) &&
           m3_metal_softmax_strided_one(
               &pair->metal, metal, dtype, rank, shape, strides, values,
               byte_count);
}

static bool m3_metal_softmax_run(
    m3_metal_softmax_pair *pair, const m3_tensor_view *host_input,
    m3_tensor_view *host_output, const m3_tensor_view *metal_input,
    m3_tensor_view *metal_output)
{
    m3_command host_command;
    m3_command metal_command;
    m3_error error;

    host_command.kind = M3_OP_SOFTMAX;
    host_command.descriptor.softmax.input = host_input;
    host_command.descriptor.softmax.output = host_output;
    metal_command.kind = M3_OP_SOFTMAX;
    metal_command.descriptor.softmax.input = metal_input;
    metal_command.descriptor.softmax.output = metal_output;
    return m3_backend_execute(pair->host.backend, &host_command, 1U, NULL,
                              &error) == M3_STATUS_OK &&
           m3_backend_execute(pair->metal.backend, &metal_command, 1U, NULL,
                              &error) == M3_STATUS_OK;
}

static bool m3_metal_softmax_f32_strided(
    m3_test_context *test, m3_metal_softmax_pair *pair)
{
    const uint64_t shape[] = {2U, 4U};
    const size_t strides[] = {5U * sizeof(float), sizeof(float)};
    const float backing[] = {
        1.0F, 2.0F, 3.0F, 4.0F, 91.0F,
        -3.0F, -1.0F, 0.0F, 2.0F, 92.0F
    };
    const float zeros[8] = {0};
    m3_tensor_view host_input;
    m3_tensor_view metal_input;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    float *host_values;
    float *metal_values;
    size_t index;
    bool created;

    created = m3_metal_softmax_pair_strided(
                  pair, &host_input, &metal_input, M3_DTYPE_F32, 2U,
                  shape, strides, backing, sizeof(backing)) &&
              m3_metal_softmax_pair_tensor(
                  pair, &host_output, &metal_output, M3_DTYPE_F32, 2U,
                  shape, zeros);
    M3_TEST_EXPECT(test, created,
                   "create strided F32 softmax oracle pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_softmax_run(
                       pair, &host_input, &host_output, &metal_input,
                       &metal_output),
                   "execute ordered-row Metal F32 SOFTMAX");
    host_values = m3_op_test_f32(&host_output);
    metal_values = m3_op_test_f32(&metal_output);
    for (index = 0U; index < 8U; ++index) {
        M3_TEST_EXPECT_F32(test, metal_values[index], host_values[index],
                           2.0e-7F, 2.0e-6F,
                           "Metal softmax matches host ordered accumulation");
    }
    M3_TEST_EXPECT_F32(test,
                       metal_values[0] + metal_values[1] +
                           metal_values[2] + metal_values[3],
                       1.0F, 2.0e-7F, 0.0F,
                       "first Metal softmax row sums to one");
    return true;
}

static bool m3_metal_softmax_f16_alias(
    m3_test_context *test, m3_metal_softmax_pair *pair)
{
    const uint64_t shape[] = {2U, 3U};
    const uint16_t values[] = {
        0x0000U, 0x3c00U, 0x4000U, 0xc000U, 0x0000U, 0x4000U
    };
    m3_tensor_view host_input;
    m3_tensor_view metal_input;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    bool created;

    created = m3_metal_softmax_pair_tensor(
        pair, &host_input, &metal_input, M3_DTYPE_F16, 2U, shape, values);
    M3_TEST_EXPECT(test, created,
                   "create exact-alias F16 softmax pair");
    if (!created) {
        return false;
    }
    host_output = host_input;
    metal_output = metal_input;
    M3_TEST_EXPECT(test,
                   m3_metal_softmax_run(
                       pair, &host_input, &host_output, &metal_input,
                       &metal_output),
                   "execute Metal F16 SOFTMAX in place");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&host_output),
                          m3_op_test_u16(&metal_output), sizeof(values)) == 0,
                   "in-place F16 softmax matches host rounded bits");
    return true;
}

static bool m3_metal_softmax_bf16_special(
    m3_test_context *test, m3_metal_softmax_pair *pair)
{
    const uint64_t shape[] = {2U, 4U};
    const uint16_t input_values[] = {
        0xff80U, 0xff80U, 0xff80U, 0xff80U,
        0x7f80U, 0x3f80U, 0x7f80U, 0xff80U
    };
    const uint16_t sentinel[8] = {
        0xdeadU, 0xdeadU, 0xdeadU, 0xdeadU,
        0xdeadU, 0xdeadU, 0xdeadU, 0xdeadU
    };
    const uint16_t expected[] = {
        0x0000U, 0x0000U, 0x0000U, 0x0000U,
        0x3f00U, 0x0000U, 0x3f00U, 0x0000U
    };
    m3_tensor_view host_input;
    m3_tensor_view metal_input;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    bool created;

    created = m3_metal_softmax_pair_tensor(
                  pair, &host_input, &metal_input, M3_DTYPE_BF16, 2U,
                  shape, input_values) &&
              m3_metal_softmax_pair_tensor(
                  pair, &host_output, &metal_output, M3_DTYPE_BF16, 2U,
                  shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create BF16 non-finite softmax pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_softmax_run(
                       pair, &host_input, &host_output, &metal_input,
                       &metal_output),
                   "execute BF16 special-row Metal SOFTMAX");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&metal_output), expected,
                          sizeof(expected)) == 0 &&
                       memcmp(m3_op_test_u16(&host_output),
                              m3_op_test_u16(&metal_output),
                              sizeof(expected)) == 0,
                   "all-negative-infinity and positive-infinity rows match");
    return true;
}

static bool m3_metal_softmax_preflight_order(
    m3_test_context *test, m3_metal_softmax_pair *pair)
{
    const uint64_t shape[] = {3U};
    const float left_values[] = {3.0F, 4.0F, 5.0F};
    const float right_values[] = {2.0F, 2.0F, 2.0F};
    const float nan_values[] = {0.0F, NAN, 1.0F};
    const float finite_values[] = {0.0F, 1.0F, 2.0F};
    const float sentinel[] = {-9.0F, -9.0F, -9.0F};
    m3_tensor_view left;
    m3_tensor_view right;
    m3_tensor_view unrelated_output;
    m3_tensor_view softmax_input;
    m3_tensor_view softmax_output;
    m3_command commands[2];
    m3_error error;
    m3_status status;
    bool created;

    created = m3_op_test_tensor(
                  &pair->metal, &left, M3_DTYPE_F32, 1U, shape,
                  left_values) &&
              m3_op_test_tensor(
                  &pair->metal, &right, M3_DTYPE_F32, 1U, shape,
                  right_values) &&
              m3_op_test_tensor(
                  &pair->metal, &unrelated_output, M3_DTYPE_F32, 1U,
                  shape, sentinel) &&
              m3_op_test_tensor(
                  &pair->metal, &softmax_input, M3_DTYPE_F32, 1U, shape,
                  nan_values) &&
              m3_op_test_tensor(
                  &pair->metal, &softmax_output, M3_DTYPE_F32, 1U, shape,
                  sentinel);
    M3_TEST_EXPECT(test, created,
                   "create Metal softmax preflight-order tensors");
    if (!created) {
        return false;
    }
    commands[0].kind = M3_OP_MUL;
    commands[0].descriptor.mul.left = &left;
    commands[0].descriptor.mul.right = &right;
    commands[0].descriptor.mul.output = &unrelated_output;
    commands[1].kind = M3_OP_SOFTMAX;
    commands[1].descriptor.softmax.input = &softmax_input;
    commands[1].descriptor.softmax.output = &softmax_output;
    status = m3_backend_execute(
        pair->metal.backend, commands, 2U, NULL, NULL);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&unrelated_output), sentinel,
                              sizeof(sentinel)) == 0 &&
                       memcmp(m3_op_test_f32(&softmax_output), sentinel,
                              sizeof(sentinel)) == 0,
                   "NaN softmax rejects a whole list with a null error sink");
    (void)m3_storage_write(softmax_input.storage, softmax_input.byte_offset,
                           finite_values, sizeof(finite_values), &error);
    status = m3_backend_execute(
        pair->metal.backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OK &&
                       m3_op_test_f32(&unrelated_output)[0] == 6.0F &&
                       m3_op_test_f32(&unrelated_output)[2] == 10.0F,
                   "an unrelated prior writer commits with static softmax");
    (void)m3_storage_write(softmax_input.storage, softmax_input.byte_offset,
                           nan_values, sizeof(nan_values), &error);
    (void)m3_storage_write(softmax_output.storage, softmax_output.byte_offset,
                           sentinel, sizeof(sentinel), &error);
    commands[0].kind = M3_OP_ADD;
    commands[0].descriptor.add.left = &left;
    commands[0].descriptor.add.right = &right;
    commands[0].descriptor.add.output = &softmax_input;
    status = m3_backend_execute(
        pair->metal.backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       strstr(m3_error_message(&error),
                              "depends on an earlier output") != NULL &&
                       memcmp(m3_op_test_f32(&softmax_input), nan_values,
                              sizeof(nan_values)) == 0 &&
                       memcmp(m3_op_test_f32(&softmax_output), sentinel,
                              sizeof(sentinel)) == 0,
                   "Metal rejects producer-dependent softmax without CPU "
                   "fallback or whole-list mutation");
    M3_TEST_EXPECT(test,
                   m3_backend_execute(
                       pair->metal.backend, commands, 2U, NULL, NULL) ==
                           M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&softmax_input), nan_values,
                              sizeof(nan_values)) == 0,
                   "producer-dependent softmax accepts a null error sink");
    commands[0].kind = M3_OP_MUL;
    commands[0].descriptor.mul.left = &left;
    commands[0].descriptor.mul.right = &right;
    commands[0].descriptor.mul.output = &softmax_input;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(
                       pair->metal.backend, commands, 2U, NULL, &error) ==
                           M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&softmax_input), nan_values,
                              sizeof(nan_values)) == 0,
                   "MUL output is visible to the global prior-writer query");
    commands[0].kind = M3_OP_SOFTMAX;
    commands[0].descriptor.softmax.input = &left;
    commands[0].descriptor.softmax.output = &softmax_input;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(
                       pair->metal.backend, commands, 2U, NULL, &error) ==
                           M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&softmax_input), nan_values,
                              sizeof(nan_values)) == 0 &&
                       memcmp(m3_op_test_f32(&softmax_output), sentinel,
                              sizeof(sentinel)) == 0,
                   "SOFTMAX output is visible before any list commit");
    return true;
}

static bool m3_metal_softmax_edge_shapes(
    m3_test_context *test, m3_metal_softmax_pair *pair)
{
    const uint64_t empty_shape[] = {0U, 3U};
    const uint64_t width_one_shape[] = {3U, 1U};
    const float width_one_values[] = {-INFINITY, INFINITY, 2.0F};
    const float zeros[] = {0.0F, 0.0F, 0.0F};
    const float expected[] = {0.0F, 1.0F, 1.0F};
    m3_tensor_view host_empty_input;
    m3_tensor_view metal_empty_input;
    m3_tensor_view host_empty_output;
    m3_tensor_view metal_empty_output;
    m3_tensor_view host_width_one_input;
    m3_tensor_view metal_width_one_input;
    m3_tensor_view host_width_one_output;
    m3_tensor_view metal_width_one_output;
    bool created;

    created = m3_metal_softmax_pair_tensor(
                  pair, &host_empty_input, &metal_empty_input,
                  M3_DTYPE_F32, 2U, empty_shape, NULL) &&
              m3_metal_softmax_pair_tensor(
                  pair, &host_empty_output, &metal_empty_output,
                  M3_DTYPE_F32, 2U, empty_shape, NULL);
    M3_TEST_EXPECT(test, created,
                   "create empty softmax pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_softmax_run(
                       pair, &host_empty_input, &host_empty_output,
                       &metal_empty_input, &metal_empty_output),
                   "empty-leading-dimension softmax performs zero work");
    created = m3_metal_softmax_pair_tensor(
                  pair, &host_width_one_input, &metal_width_one_input,
                  M3_DTYPE_F32, 2U, width_one_shape, width_one_values) &&
              m3_metal_softmax_pair_tensor(
                  pair, &host_width_one_output, &metal_width_one_output,
                  M3_DTYPE_F32, 2U, width_one_shape, zeros);
    M3_TEST_EXPECT(test, created,
                   "create width-one softmax pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_softmax_run(
                       pair, &host_width_one_input, &host_width_one_output,
                       &metal_width_one_input, &metal_width_one_output) &&
                       memcmp(m3_op_test_f32(&metal_width_one_output),
                              expected, sizeof(expected)) == 0,
                   "width-one rows preserve special softmax contracts");
    return true;
}

void m3_test_metal_basic_softmax(m3_test_context *test)
{
    m3_metal_softmax_pair pair;

    if (!m3_metal_softmax_pair_init(test, &pair)) {
        return;
    }
    if (!m3_metal_softmax_f32_strided(test, &pair) ||
        !m3_metal_softmax_f16_alias(test, &pair) ||
        !m3_metal_softmax_bf16_special(test, &pair) ||
        !m3_metal_softmax_preflight_order(test, &pair) ||
        !m3_metal_softmax_edge_shapes(test, &pair)) {
        m3_metal_softmax_pair_dispose(&pair);
        return;
    }
    m3_metal_softmax_pair_dispose(&pair);
}
