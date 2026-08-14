/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_sampling_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static void m3_test_top_k_command(
    m3_command *command, const m3_metal_sampling_view *logits,
    m3_metal_sampling_view *values, m3_metal_sampling_view *indices,
    uint64_t k, bool metal)
{
    command->kind = M3_OP_TOP_K;
    command->descriptor.top_k.logits = metal ? &logits->metal : &logits->host;
    command->descriptor.top_k.values = metal ? &values->metal : &values->host;
    command->descriptor.top_k.indices =
        metal ? &indices->metal : &indices->host;
    command->descriptor.top_k.k = k;
}

static bool m3_test_top_k_run(
    m3_metal_sampling_fixture *fixture,
    const m3_metal_sampling_view *logits,
    m3_metal_sampling_view *values, m3_metal_sampling_view *indices,
    uint64_t k, m3_error *error)
{
    m3_command host;
    m3_command metal;

    m3_test_top_k_command(&host, logits, values, indices, k, false);
    m3_test_top_k_command(&metal, logits, values, indices, k, true);
    return m3_metal_sampling_execute(fixture, &host, &metal, 1U, error) &&
           m3_metal_sampling_equal(values) &&
           m3_metal_sampling_equal(indices);
}

static bool m3_test_top_k_f32_strided(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t input_shape[] = {2U, 6U};
    const uint64_t output_shape[] = {2U, 4U};
    const size_t strides[] = {8U * sizeof(float), sizeof(float)};
    const float padded[] = {
        101.0F, 3.0F, INFINITY, INFINITY, -INFINITY, -0.0F, 0.0F, 102.0F,
        103.0F, -INFINITY, -INFINITY, 7.0F, 7.0F, 6.0F, 8.0F, 104.0F
    };
    const float value_sentinel[8] = {
        -99.0F, -99.0F, -99.0F, -99.0F,
        -99.0F, -99.0F, -99.0F, -99.0F
    };
    const int32_t index_sentinel[8] = {
        -99, -99, -99, -99, -99, -99, -99, -99
    };
    const float expected_values[] = {
        INFINITY, INFINITY, 3.0F, -0.0F, 8.0F, 7.0F, 7.0F, 6.0F
    };
    const int32_t expected_indices[] = {1, 2, 0, 4, 5, 2, 3, 4};
    m3_metal_sampling_view logits;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_error error;
    bool created;

    created = m3_metal_sampling_strided(
                  fixture, &logits, M3_DTYPE_F32, 2U, input_shape, strides,
                  sizeof(float), sizeof(padded), padded) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 2U, output_shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 2U, output_shape,
                  index_sentinel);
    M3_TEST_EXPECT(test, created, "create strided F32 Metal top-k tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_test_top_k_run(
                       fixture, &logits, &values, &indices, 4U, &error),
                   "Metal top-k exactly matches the host on strided F32");
    M3_TEST_EXPECT(
        test,
        memcmp(m3_op_test_f32(&values.metal), expected_values,
               sizeof(expected_values)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), expected_indices,
                   sizeof(expected_indices)) == 0,
        "Metal top-k preserves infinities, signed zero, and tie order");
    return true;
}

static bool m3_test_top_k_f16_all_negative(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t shape[] = {1U, 4U};
    const uint16_t logits_bits[] = {0xfc00U, 0xfc00U, 0xfc00U, 0xfc00U};
    const float value_sentinel[] = {-9.0F, -9.0F, -9.0F, -9.0F};
    const int32_t index_sentinel[] = {-9, -9, -9, -9};
    const float expected_values[] = {
        -INFINITY, -INFINITY, -INFINITY, -INFINITY
    };
    const int32_t expected_indices[] = {0, 1, 2, 3};
    m3_metal_sampling_view logits;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits, M3_DTYPE_F16, 2U, shape, logits_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 2U, shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 2U, shape,
                  index_sentinel);
    M3_TEST_EXPECT(test, created, "create all-infinite F16 top-k tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_test_top_k_run(
                       fixture, &logits, &values, &indices, 4U, &error),
                   "Metal top-k exactly matches the host for F16");
    M3_TEST_EXPECT(
        test,
        memcmp(m3_op_test_f32(&values.metal), expected_values,
               sizeof(expected_values)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), expected_indices,
                   sizeof(expected_indices)) == 0,
        "all-negative-infinity top-k falls back through low indices");
    return true;
}

static bool m3_test_top_k_bf16(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 5U};
    const uint64_t output_shape[] = {1U, 1U, 3U};
    const uint16_t logits_bits[] = {
        0x4000U, 0x4000U, 0xbf80U, 0x4040U, 0x4040U
    };
    const float value_sentinel[] = {-8.0F, -8.0F, -8.0F};
    const int32_t index_sentinel[] = {-8, -8, -8};
    const float expected_values[] = {3.0F, 3.0F, 2.0F};
    const int32_t expected_indices[] = {3, 4, 0};
    m3_metal_sampling_view logits;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &logits, M3_DTYPE_BF16, 3U, input_shape,
                  logits_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 3U, output_shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 3U, output_shape,
                  index_sentinel);
    M3_TEST_EXPECT(test, created, "create rank-three BF16 top-k tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_test_top_k_run(
                       fixture, &logits, &values, &indices, 3U, &error),
                   "Metal top-k exactly matches the host for BF16");
    M3_TEST_EXPECT(
        test,
        memcmp(m3_op_test_f32(&values.metal), expected_values,
               sizeof(expected_values)) == 0 &&
            memcmp(m3_op_test_i32(&indices.metal), expected_indices,
                   sizeof(expected_indices)) == 0,
        "BF16 top-k preserves descending values and low-index ties");
    return true;
}

static bool m3_test_top_k_bit_patterns(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    enum {
        row_count = 16,
        width = 19,
        element_count = row_count * width
    };
    const uint64_t shape[] = {row_count, width};
    uint32_t logits_bits[element_count];
    float value_sentinel[element_count];
    int32_t index_sentinel[element_count];
    m3_metal_sampling_view logits;
    m3_metal_sampling_view values;
    m3_metal_sampling_view indices;
    m3_error error;
    uint32_t state = 0x31415926U;
    size_t index;
    bool created;

    for (index = 0U; index < element_count; ++index) {
        uint32_t exponent;

        state = state * 1664525U + 1013904223U;
        exponent = (state >> 23) % 255U;
        logits_bits[index] = (state & 0x807fffffU) | (exponent << 23);
        if (index % 37U == 0U) {
            logits_bits[index] &= 0x80000000U;
        } else if (index % 41U == 0U) {
            logits_bits[index] = (state & 0x80000000U) | 1U;
        }
        value_sentinel[index] = -123.0F;
        index_sentinel[index] = -123;
    }
    created = m3_metal_sampling_tensor(
                  fixture, &logits, M3_DTYPE_F32, 2U, shape,
                  logits_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &values, M3_DTYPE_F32, 2U, shape,
                  value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &indices, M3_DTYPE_I32, 2U, shape,
                  index_sentinel);
    M3_TEST_EXPECT(test, created,
                   "create diverse F32 bit-pattern top-k tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_top_k_run(
            fixture, &logits, &values, &indices, width, &error),
        "Metal top-k exactly orders diverse finite F32 bit patterns");
    return true;
}

static bool m3_test_top_k_boundaries(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t empty_input_shape[] = {0U, 3U};
    const uint64_t empty_output_shape[] = {0U, 2U};
    const uint64_t one_shape[] = {1U};
    const uint64_t subnormal_shape[] = {1U, 4U};
    const float one_value[] = {4.25F};
    const float one_sentinel[] = {-1.0F};
    const int32_t one_index_sentinel[] = {-1};
    const float subnormal_logits[] = {
        0x1p-149F, 0.0F, -0.0F, -0x1p-149F
    };
    const float subnormal_value_sentinel[] = {
        -2.0F, -2.0F, -2.0F, -2.0F
    };
    const int32_t subnormal_index_sentinel[] = {-2, -2, -2, -2};
    const int32_t subnormal_expected_indices[] = {0, 1, 2, 3};
    m3_metal_sampling_view empty_logits;
    m3_metal_sampling_view empty_values;
    m3_metal_sampling_view empty_indices;
    m3_metal_sampling_view one_logits;
    m3_metal_sampling_view one_values;
    m3_metal_sampling_view one_indices;
    m3_metal_sampling_view subnormal_input;
    m3_metal_sampling_view subnormal_values;
    m3_metal_sampling_view subnormal_indices;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &empty_logits, M3_DTYPE_F32, 2U,
                  empty_input_shape, NULL) &&
              m3_metal_sampling_tensor(
                  fixture, &empty_values, M3_DTYPE_F32, 2U,
                  empty_output_shape, NULL) &&
              m3_metal_sampling_tensor(
                  fixture, &empty_indices, M3_DTYPE_I32, 2U,
                  empty_output_shape, NULL) &&
              m3_metal_sampling_tensor(
                  fixture, &one_logits, M3_DTYPE_F32, 1U, one_shape,
                  one_value) &&
              m3_metal_sampling_tensor(
                  fixture, &one_values, M3_DTYPE_F32, 1U, one_shape,
                  one_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &one_indices, M3_DTYPE_I32, 1U, one_shape,
                  one_index_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &subnormal_input, M3_DTYPE_F32, 2U,
                  subnormal_shape, subnormal_logits) &&
              m3_metal_sampling_tensor(
                  fixture, &subnormal_values, M3_DTYPE_F32, 2U,
                  subnormal_shape, subnormal_value_sentinel) &&
              m3_metal_sampling_tensor(
                  fixture, &subnormal_indices, M3_DTYPE_I32, 2U,
                  subnormal_shape, subnormal_index_sentinel);
    M3_TEST_EXPECT(test, created, "create boundary top-k tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_top_k_run(fixture, &empty_logits, &empty_values,
                          &empty_indices, 2U, &error),
        "Metal top-k accepts a legal empty leading dimension");
    M3_TEST_EXPECT(
        test,
        m3_test_top_k_run(fixture, &one_logits, &one_values, &one_indices,
                          1U, &error) &&
            m3_op_test_f32(&one_values.metal)[0] == 4.25F &&
            m3_op_test_i32(&one_indices.metal)[0] == 0,
        "Metal top-k accepts the legal width-one boundary");
    M3_TEST_EXPECT(
        test,
        m3_test_top_k_run(
            fixture, &subnormal_input, &subnormal_values,
            &subnormal_indices, 4U, &error) &&
            memcmp(m3_op_test_f32(&subnormal_values.metal),
                   subnormal_logits, sizeof(subnormal_logits)) == 0 &&
            memcmp(m3_op_test_i32(&subnormal_indices.metal),
                   subnormal_expected_indices,
                   sizeof(subnormal_expected_indices)) == 0,
        "Metal top-k orders subnormals and signed zeros exactly");
    return true;
}

void m3_test_metal_sampling_top_k(m3_test_context *test)
{
    m3_metal_sampling_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_error error;

    if (!m3_metal_sampling_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_top_k_f32_strided(test, &fixture) ||
        !m3_test_top_k_f16_all_negative(test, &fixture) ||
        !m3_test_top_k_bf16(test, &fixture) ||
        !m3_test_top_k_bit_patterns(test, &fixture) ||
        !m3_test_top_k_boundaries(test, &fixture)) {
        m3_metal_sampling_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(
            fixture.metal.backend, &stats, &error) == M3_STATUS_OK &&
            stats.live_storage_count == fixture.metal.storage_count &&
            stats.live_storage_count != 0U &&
            stats.live_allocated_bytes != 0U,
        "Metal top-k storage remains live and accounted");
    m3_metal_sampling_fixture_dispose(&fixture);
}
