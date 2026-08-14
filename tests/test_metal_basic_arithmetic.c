/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    m3_op_test_fixture host;
    m3_op_test_fixture metal;
} m3_metal_basic_pair;

static bool m3_metal_basic_pair_init(m3_test_context *test,
                                     m3_metal_basic_pair *pair)
{
    m3_error error;
    m3_status status;

    (void)memset(pair, 0, sizeof(*pair));
    if (!m3_op_test_fixture_init(&pair->host)) {
        M3_TEST_EXPECT(test, false, "create host arithmetic oracle");
        return false;
    }
    status = m3_backend_create_metal(&pair->metal.backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        m3_op_test_fixture_dispose(&pair->host);
        M3_TEST_SKIP(test, m3_error_message(&error));
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal arithmetic fixture");
    if (status != M3_STATUS_OK) {
        m3_op_test_fixture_dispose(&pair->host);
        return false;
    }
    return true;
}

static void m3_metal_basic_pair_dispose(m3_metal_basic_pair *pair)
{
    m3_op_test_fixture_dispose(&pair->metal);
    m3_op_test_fixture_dispose(&pair->host);
}

static bool m3_metal_basic_pair_tensor(
    m3_metal_basic_pair *pair, m3_tensor_view *host,
    m3_tensor_view *metal, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, const void *values)
{
    return m3_op_test_tensor(&pair->host, host, dtype, rank, shape,
                             values) &&
           m3_op_test_tensor(&pair->metal, metal, dtype, rank, shape,
                             values);
}

static bool m3_metal_basic_strided_one(
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

static bool m3_metal_basic_pair_strided(
    m3_metal_basic_pair *pair, m3_tensor_view *host,
    m3_tensor_view *metal, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, const size_t *strides, const void *values,
    size_t byte_count)
{
    return m3_metal_basic_strided_one(
               &pair->host, host, dtype, rank, shape, strides, values,
               byte_count) &&
           m3_metal_basic_strided_one(
               &pair->metal, metal, dtype, rank, shape, strides, values,
               byte_count);
}

static bool m3_metal_basic_run_binary(
    m3_metal_basic_pair *pair, m3_op_kind kind,
    const m3_tensor_view *host_left, const m3_tensor_view *host_right,
    m3_tensor_view *host_output, const m3_tensor_view *metal_left,
    const m3_tensor_view *metal_right, m3_tensor_view *metal_output)
{
    m3_command host_command;
    m3_command metal_command;
    m3_error error;

    (void)memset(&host_command, 0, sizeof(host_command));
    (void)memset(&metal_command, 0, sizeof(metal_command));
    host_command.kind = kind;
    metal_command.kind = kind;
    if (kind == M3_OP_ADD) {
        host_command.descriptor.add.left = host_left;
        host_command.descriptor.add.right = host_right;
        host_command.descriptor.add.output = host_output;
        metal_command.descriptor.add.left = metal_left;
        metal_command.descriptor.add.right = metal_right;
        metal_command.descriptor.add.output = metal_output;
    } else {
        host_command.descriptor.mul.left = host_left;
        host_command.descriptor.mul.right = host_right;
        host_command.descriptor.mul.output = host_output;
        metal_command.descriptor.mul.left = metal_left;
        metal_command.descriptor.mul.right = metal_right;
        metal_command.descriptor.mul.output = metal_output;
    }
    return m3_backend_execute(pair->host.backend, &host_command, 1U, NULL,
                              &error) == M3_STATUS_OK &&
           m3_backend_execute(pair->metal.backend, &metal_command, 1U, NULL,
                              &error) == M3_STATUS_OK;
}

static bool m3_metal_basic_f32_broadcast(
    m3_test_context *test, m3_metal_basic_pair *pair)
{
    const uint64_t matrix_shape[] = {2U, 3U};
    const uint64_t vector_shape[] = {3U};
    const size_t strides[] = {sizeof(float), 2U * sizeof(float)};
    const float backing[] = {
        16777216.0F, -16777216.0F, 1.00000011920928955078125F,
        0x1p-149F, -3.5F, 8.0F
    };
    const float right_values[] = {1.0F, 0x1p-24F, 0.25F};
    const float zeros[6] = {0};
    m3_tensor_view host_left;
    m3_tensor_view metal_left;
    m3_tensor_view host_right;
    m3_tensor_view metal_right;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    uint32_t first_bits;
    bool created;

    created = m3_metal_basic_pair_strided(
                  pair, &host_left, &metal_left, M3_DTYPE_F32, 2U,
                  matrix_shape, strides, backing, sizeof(backing)) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_right, &metal_right, M3_DTYPE_F32, 1U,
                  vector_shape, right_values) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_output, &metal_output, M3_DTYPE_F32, 2U,
                  matrix_shape, zeros);
    M3_TEST_EXPECT(test, created,
                   "create strided F32 broadcast arithmetic pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_basic_run_binary(
                       pair, M3_OP_ADD, &host_left, &host_right,
                       &host_output, &metal_left, &metal_right,
                       &metal_output),
                   "execute Metal F32 right-aligned broadcast ADD");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_f32(&host_output),
                          m3_op_test_f32(&metal_output), sizeof(zeros)) == 0,
                   "Metal F32 ADD matches the ordered host oracle bits");
    (void)memcpy(&first_bits, m3_op_test_f32(&metal_output),
                 sizeof(first_bits));
    M3_TEST_EXPECT(test, first_bits == UINT32_C(0x4b800000),
                   "F32 ADD rounds 2^24 plus one back to even");
    return true;
}

static bool m3_metal_basic_f16_broadcast(
    m3_test_context *test, m3_metal_basic_pair *pair)
{
    const uint64_t matrix_shape[] = {2U, 2U};
    const uint64_t vector_shape[] = {2U};
    const size_t strides[] = {2U * sizeof(uint16_t)};
    const uint16_t left_values[] = {0x3c00U, 0x4000U, 0xc200U, 0x3800U};
    const uint16_t right_backing[] = {0x3e00U, 0x7e00U, 0x3555U};
    const uint16_t zeros[4] = {0U};
    m3_tensor_view host_left;
    m3_tensor_view metal_left;
    m3_tensor_view host_right;
    m3_tensor_view metal_right;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    bool created;

    created = m3_metal_basic_pair_tensor(
                  pair, &host_left, &metal_left, M3_DTYPE_F16, 2U,
                  matrix_shape, left_values) &&
              m3_metal_basic_pair_strided(
                  pair, &host_right, &metal_right, M3_DTYPE_F16, 1U,
                  vector_shape, strides, right_backing,
                  sizeof(right_backing)) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_output, &metal_output, M3_DTYPE_F16, 2U,
                  matrix_shape, zeros);
    M3_TEST_EXPECT(test, created,
                   "create strided F16 broadcast arithmetic pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_basic_run_binary(
                       pair, M3_OP_MUL, &host_left, &host_right,
                       &host_output, &metal_left, &metal_right,
                       &metal_output),
                   "execute Metal F16 right-aligned broadcast MUL");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&host_output),
                          m3_op_test_u16(&metal_output), sizeof(zeros)) == 0,
                   "Metal F16 MUL matches host load-compute-store bits");
    return true;
}

static bool m3_metal_basic_bf16_left_alias(
    m3_test_context *test, m3_metal_basic_pair *pair)
{
    const uint64_t shape[] = {4U};
    const uint16_t left_values[] = {0x3f80U, 0x3f81U, 0xbf80U, 0x7f7fU};
    const uint16_t right_values[] = {0x3b80U, 0x3b80U, 0x3f00U, 0xff7fU};
    const uint16_t expected[] = {0x3f80U, 0x3f82U, 0xbf00U, 0x0000U};
    m3_tensor_view host_left;
    m3_tensor_view metal_left;
    m3_tensor_view host_right;
    m3_tensor_view metal_right;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    bool created;

    created = m3_metal_basic_pair_tensor(
                  pair, &host_left, &metal_left, M3_DTYPE_BF16, 1U,
                  shape, left_values) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_right, &metal_right, M3_DTYPE_BF16, 1U,
                  shape, right_values);
    M3_TEST_EXPECT(test, created,
                   "create BF16 exact-left-alias ADD pair");
    if (!created) {
        return false;
    }
    host_output = host_left;
    metal_output = metal_left;
    M3_TEST_EXPECT(test,
                   m3_metal_basic_run_binary(
                       pair, M3_OP_ADD, &host_left, &host_right,
                       &host_output, &metal_left, &metal_right,
                       &metal_output),
                   "execute Metal BF16 ADD in place on the left input");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&metal_output), expected,
                          sizeof(expected)) == 0 &&
                       memcmp(m3_op_test_u16(&host_output),
                              m3_op_test_u16(&metal_output),
                              sizeof(expected)) == 0,
                   "BF16 ADD exact alias uses one RN-even output store");
    return true;
}

static bool m3_metal_basic_mixed_right_alias(
    m3_test_context *test, m3_metal_basic_pair *pair)
{
    const uint64_t shape[] = {4U};
    const float left_values[] = {2.0F, -2.0F, 0.5F, 4.0F};
    const uint16_t right_values[] = {0x3fc0U, 0xc000U, 0x3f00U, 0x3e80U};
    const uint16_t expected[] = {0x4200U, 0x4400U, 0x3400U, 0x3c00U};
    m3_tensor_view host_left;
    m3_tensor_view metal_left;
    m3_tensor_view host_right;
    m3_tensor_view metal_right;
    m3_tensor_view host_output;
    m3_tensor_view metal_output;
    m3_error error;
    bool created;

    m3_tensor_view_init(&host_output);
    m3_tensor_view_init(&metal_output);
    created = m3_metal_basic_pair_tensor(
                  pair, &host_left, &metal_left, M3_DTYPE_F32, 1U,
                  shape, left_values) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_right, &metal_right, M3_DTYPE_BF16, 1U,
                  shape, right_values) &&
              m3_tensor_view_contiguous(
                  &host_output, host_right.storage, M3_DTYPE_F16, 1U,
                  shape, 0U, &error) == M3_STATUS_OK &&
              m3_tensor_view_contiguous(
                  &metal_output, metal_right.storage, M3_DTYPE_F16, 1U,
                  shape, 0U, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, created,
                   "create mixed-dtype exact-right-alias MUL pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_basic_run_binary(
                       pair, M3_OP_MUL, &host_left, &host_right,
                       &host_output, &metal_left, &metal_right,
                       &metal_output),
                   "execute mixed-dtype Metal MUL in place on the right");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&metal_output), expected,
                          sizeof(expected)) == 0 &&
                       memcmp(m3_op_test_u16(&host_output),
                              m3_op_test_u16(&metal_output),
                              sizeof(expected)) == 0,
                   "mixed MUL loads BF16 then stores one F16 result");
    return true;
}

static bool m3_metal_basic_edge_shapes(
    m3_test_context *test, m3_metal_basic_pair *pair)
{
    const uint64_t empty_shape[] = {0U, 3U};
    const uint64_t row_shape[] = {1U, 3U};
    const float row_values[] = {1.0F, 2.0F, 3.0F};
    const float scalar_left = 1.5F;
    const float scalar_right = 2.25F;
    const float scalar_zero = 0.0F;
    m3_tensor_view host_empty;
    m3_tensor_view metal_empty;
    m3_tensor_view host_row;
    m3_tensor_view metal_row;
    m3_tensor_view host_empty_output;
    m3_tensor_view metal_empty_output;
    m3_tensor_view host_scalar_left;
    m3_tensor_view metal_scalar_left;
    m3_tensor_view host_scalar_right;
    m3_tensor_view metal_scalar_right;
    m3_tensor_view host_scalar_output;
    m3_tensor_view metal_scalar_output;
    bool created;

    created = m3_metal_basic_pair_tensor(
                  pair, &host_empty, &metal_empty, M3_DTYPE_F32, 2U,
                  empty_shape, NULL) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_row, &metal_row, M3_DTYPE_F32, 2U,
                  row_shape, row_values) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_empty_output, &metal_empty_output,
                  M3_DTYPE_F32, 2U, empty_shape, NULL);
    M3_TEST_EXPECT(test, created,
                   "create empty broadcast arithmetic pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_basic_run_binary(
                       pair, M3_OP_ADD, &host_empty, &host_row,
                       &host_empty_output, &metal_empty, &metal_row,
                       &metal_empty_output),
                   "empty Metal ADD validates and performs zero work");
    created = m3_metal_basic_pair_tensor(
                  pair, &host_scalar_left, &metal_scalar_left,
                  M3_DTYPE_F32, 0U, NULL, &scalar_left) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_scalar_right, &metal_scalar_right,
                  M3_DTYPE_F32, 0U, NULL, &scalar_right) &&
              m3_metal_basic_pair_tensor(
                  pair, &host_scalar_output, &metal_scalar_output,
                  M3_DTYPE_F32, 0U, NULL, &scalar_zero);
    M3_TEST_EXPECT(test, created,
                   "create rank-zero Metal arithmetic pair");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_metal_basic_run_binary(
                       pair, M3_OP_ADD, &host_scalar_left,
                       &host_scalar_right, &host_scalar_output,
                       &metal_scalar_left, &metal_scalar_right,
                       &metal_scalar_output) &&
                       m3_op_test_f32(&metal_scalar_output)[0] == 3.75F,
                   "rank-zero Metal ADD dispatches one scalar");
    return true;
}

void m3_test_metal_basic_arithmetic(m3_test_context *test)
{
    m3_metal_basic_pair pair;
    m3_backend_allocation_stats stats;
    m3_error error;

    if (!m3_metal_basic_pair_init(test, &pair)) {
        return;
    }
    if (!m3_metal_basic_f32_broadcast(test, &pair) ||
        !m3_metal_basic_f16_broadcast(test, &pair) ||
        !m3_metal_basic_bf16_left_alias(test, &pair) ||
        !m3_metal_basic_mixed_right_alias(test, &pair) ||
        !m3_metal_basic_edge_shapes(test, &pair)) {
        m3_metal_basic_pair_dispose(&pair);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       pair.metal.backend, &stats, &error) == M3_STATUS_OK &&
                       stats.live_storage_count == pair.metal.storage_count &&
                       stats.live_allocated_bytes != 0U,
                   "Metal arithmetic storage remains live and accounted");
    m3_metal_basic_pair_dispose(&pair);
}
