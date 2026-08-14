/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_sampling_test.h"

#include <stdint.h>
#include <string.h>

static void m3_test_categorical_command(
    m3_command *command,
    const m3_metal_sampling_view *probabilities,
    const m3_metal_sampling_view *uniforms,
    m3_metal_sampling_view *output, bool metal)
{
    command->kind = M3_OP_CATEGORICAL;
    command->descriptor.categorical.probabilities =
        metal ? &probabilities->metal : &probabilities->host;
    command->descriptor.categorical.uniforms =
        metal ? &uniforms->metal : &uniforms->host;
    command->descriptor.categorical.output =
        metal ? &output->metal : &output->host;
}

static bool m3_test_categorical_run(
    m3_metal_sampling_fixture *fixture,
    const m3_metal_sampling_view *probabilities,
    const m3_metal_sampling_view *uniforms,
    m3_metal_sampling_view *output, m3_error *error)
{
    m3_command host;
    m3_command metal;

    m3_test_categorical_command(
        &host, probabilities, uniforms, output, false);
    m3_test_categorical_command(
        &metal, probabilities, uniforms, output, true);
    return m3_metal_sampling_execute(fixture, &host, &metal, 1U, error) &&
           m3_metal_sampling_equal(output);
}

static bool m3_test_categorical_f32_strided(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t probability_shape[] = {4U, 3U};
    const uint64_t row_shape[] = {4U};
    const size_t probability_strides[] = {
        4U * sizeof(float), sizeof(float)
    };
    const size_t uniform_strides[] = {2U * sizeof(float)};
    const float probability_storage[] = {
        91.0F, 0.2F, 0.3F, 0.5F, 92.0F, 0.2F, 0.3F, 0.5F, 93.0F,
        0.2F, 0.3F, 0.5F, 94.0F, 0.2F, 0.3F, 0.5F, 95.0F
    };
    const float uniform_storage[] = {
        81.0F, 0.0F, 82.0F, 0.2F, 83.0F,
        0.5F, 84.0F, 0.999F, 85.0F
    };
    const int32_t sentinel[] = {-7, -7, -7, -7};
    const int32_t expected[] = {0, 1, 2, 2};
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_error error;
    bool created;

    created = m3_metal_sampling_strided(
                  fixture, &probabilities, M3_DTYPE_F32, 2U,
                  probability_shape, probability_strides, sizeof(float),
                  sizeof(probability_storage), probability_storage) &&
              m3_metal_sampling_strided(
                  fixture, &uniforms, M3_DTYPE_F32, 1U, row_shape,
                  uniform_strides, sizeof(float), sizeof(uniform_storage),
                  uniform_storage) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create strided F32 Metal categorical tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &probabilities, &uniforms, &output, &error),
        "Metal categorical exactly matches the host on strided F32");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_i32(&output.metal), expected,
                          sizeof(expected)) == 0,
                   "Metal categorical uses half-open cumulative bins");
    return true;
}

static bool m3_test_categorical_f16_bf16(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t probability_shape[] = {4U, 3U};
    const uint64_t row_shape[] = {4U};
    const uint16_t probability_bits[] = {
        0x3400U, 0x3400U, 0x3800U, 0x3400U, 0x3400U, 0x3800U,
        0x3400U, 0x3400U, 0x3800U, 0x3400U, 0x3400U, 0x3800U
    };
    const uint16_t uniform_bits[] = {
        0x0000U, 0x3e80U, 0x3f00U, 0x3f7fU
    };
    const int32_t sentinel[] = {-6, -6, -6, -6};
    const int32_t expected[] = {0, 1, 2, 2};
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &probabilities, M3_DTYPE_F16, 2U,
                  probability_shape, probability_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_BF16, 1U, row_shape,
                  uniform_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create F16-probability BF16-uniform tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &probabilities, &uniforms, &output, &error),
        "Metal categorical matches mixed F16 and BF16 host arithmetic");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_i32(&output.metal), expected,
                          sizeof(expected)) == 0,
                   "mixed categorical dtypes preserve exact boundaries");
    return true;
}

static bool m3_test_categorical_bf16_f16(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t probability_shape[] = {2U, 4U};
    const uint64_t row_shape[] = {2U};
    const uint16_t probability_bits[] = {
        0x0000U, 0x3f00U, 0x0000U, 0x3f00U,
        0x0000U, 0x3f00U, 0x0000U, 0x3f00U
    };
    const uint16_t uniform_bits[] = {0x0000U, 0x3bffU};
    const int32_t sentinel[] = {-5, -5};
    const int32_t expected[] = {1, 3};
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &probabilities, M3_DTYPE_BF16, 2U,
                  probability_shape, probability_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_F16, 1U, row_shape,
                  uniform_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create BF16-probability F16-uniform tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &probabilities, &uniforms, &output, &error),
        "Metal categorical matches the reverse mixed dtype host path");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_i32(&output.metal), expected,
                          sizeof(expected)) == 0,
                   "categorical skips zero-probability bins in order");
    return true;
}

static bool m3_test_categorical_fallback(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t probability_shape[] = {2U, 3U};
    const uint64_t row_shape[] = {2U};
    const float probability_values[] = {
        0.0F, 0x1p-126F, 0.0F, 0.0F, 0x1p-149F, 0.0F
    };
    const float uniform_values[] = {
        0x1.fffffep-1F, 0x1.fffffep-1F
    };
    const int32_t sentinel[] = {-4, -4};
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &probabilities, M3_DTYPE_F32, 2U,
                  probability_shape, probability_values) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_F32, 1U, row_shape,
                  uniform_values) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create rounded-target categorical fallback tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &probabilities, &uniforms, &output, &error) &&
            m3_op_test_i32(&output.host)[0] == 1 &&
            m3_op_test_i32(&output.host)[1] == 1 &&
            m3_op_test_i32(&output.metal)[0] == 1 &&
            m3_op_test_i32(&output.metal)[1] == 1,
        "rounded normal and subnormal targets use last-positive fallback");
    return true;
}

static bool m3_test_categorical_bit_patterns(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    enum {
        row_count = 32,
        width = 17,
        probability_count = row_count * width
    };
    const uint64_t probability_shape[] = {row_count, width};
    const uint64_t row_shape[] = {row_count};
    uint32_t probability_bits[probability_count];
    uint32_t uniform_bits[row_count];
    int32_t sentinel[row_count];
    m3_metal_sampling_view probabilities;
    m3_metal_sampling_view uniforms;
    m3_metal_sampling_view output;
    m3_error error;
    uint32_t state = 0x27182818U;
    size_t row;
    size_t index;
    bool created;

    for (index = 0U; index < probability_count; ++index) {
        uint32_t exponent;

        state = state * 1664525U + 1013904223U;
        exponent = (state >> 23) % 127U;
        probability_bits[index] =
            (exponent << 23) | (state & 0x7fffffU);
        if (index % 29U == 0U) {
            probability_bits[index] = 0U;
        } else if (index % 31U == 0U) {
            probability_bits[index] = 1U;
        } else if (index % 43U == 0U) {
            probability_bits[index] = 0x80000000U;
        }
    }
    for (row = 0U; row < row_count; ++row) {
        uint32_t exponent;

        probability_bits[row * width] =
            row % 2U == 0U ? 0x00800000U : 1U;
        state = state * 1664525U + 1013904223U;
        exponent = (state >> 23) % 127U;
        uniform_bits[row] = (exponent << 23) | (state & 0x7fffffU);
        sentinel[row] = -3;
    }
    uniform_bits[0] = 0x3f7fffffU;
    uniform_bits[1] = 0U;
    uniform_bits[2] = 0x80000000U;
    created = m3_metal_sampling_tensor(
                  fixture, &probabilities, M3_DTYPE_F32, 2U,
                  probability_shape, probability_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &uniforms, M3_DTYPE_F32, 1U, row_shape,
                  uniform_bits) &&
              m3_metal_sampling_tensor(
                  fixture, &output, M3_DTYPE_I32, 1U, row_shape, sentinel);
    M3_TEST_EXPECT(test, created,
                   "create diverse F32 bit-pattern categorical tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &probabilities, &uniforms, &output, &error),
        "Metal categorical exactly matches diverse sequential F32 rows");
    return true;
}

static bool m3_test_categorical_boundaries(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    const uint64_t empty_probability_shape[] = {0U, 3U};
    const uint64_t empty_row_shape[] = {0U};
    const uint64_t one_probability_shape[] = {1U, 1U};
    const uint64_t one_row_shape[] = {1U};
    const float one_probability[] = {2.0F};
    const float one_uniform[] = {0.5F};
    const int32_t one_sentinel[] = {-1};
    m3_metal_sampling_view empty_probabilities;
    m3_metal_sampling_view empty_uniforms;
    m3_metal_sampling_view empty_output;
    m3_metal_sampling_view one_probabilities;
    m3_metal_sampling_view one_uniforms;
    m3_metal_sampling_view one_output;
    m3_error error;
    bool created;

    created = m3_metal_sampling_tensor(
                  fixture, &empty_probabilities, M3_DTYPE_F32, 2U,
                  empty_probability_shape, NULL) &&
              m3_metal_sampling_tensor(
                  fixture, &empty_uniforms, M3_DTYPE_F32, 1U,
                  empty_row_shape, NULL) &&
              m3_metal_sampling_tensor(
                  fixture, &empty_output, M3_DTYPE_I32, 1U,
                  empty_row_shape, NULL) &&
              m3_metal_sampling_tensor(
                  fixture, &one_probabilities, M3_DTYPE_F32, 2U,
                  one_probability_shape, one_probability) &&
              m3_metal_sampling_tensor(
                  fixture, &one_uniforms, M3_DTYPE_F32, 1U, one_row_shape,
                  one_uniform) &&
              m3_metal_sampling_tensor(
                  fixture, &one_output, M3_DTYPE_I32, 1U, one_row_shape,
                  one_sentinel);
    M3_TEST_EXPECT(test, created, "create boundary categorical tensors");
    if (!created) {
        return false;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &empty_probabilities, &empty_uniforms, &empty_output,
            &error),
        "Metal categorical accepts a legal empty leading dimension");
    M3_TEST_EXPECT(
        test,
        m3_test_categorical_run(
            fixture, &one_probabilities, &one_uniforms, &one_output,
            &error) &&
            m3_op_test_i32(&one_output.metal)[0] == 0,
        "Metal categorical accepts the legal width-one boundary");
    return true;
}

void m3_test_metal_sampling_categorical(m3_test_context *test)
{
    m3_metal_sampling_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_error error;

    if (!m3_metal_sampling_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_categorical_f32_strided(test, &fixture) ||
        !m3_test_categorical_f16_bf16(test, &fixture) ||
        !m3_test_categorical_bf16_f16(test, &fixture) ||
        !m3_test_categorical_fallback(test, &fixture) ||
        !m3_test_categorical_bit_patterns(test, &fixture) ||
        !m3_test_categorical_boundaries(test, &fixture)) {
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
        "Metal categorical storage remains live and accounted");
    m3_metal_sampling_fixture_dispose(&fixture);
}
