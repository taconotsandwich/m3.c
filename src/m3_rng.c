/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rng.h"

#include <math.h>

#define M3_PCG32_MULTIPLIER UINT64_C(6364136223846793005)

static uint32_t m3_rng_u32_raw(m3_rng *rng)
{
    uint64_t old_state = rng->state;
    uint32_t xorshifted;
    uint32_t rotation;

    rng->state = old_state * M3_PCG32_MULTIPLIER + rng->increment;
    xorshifted = (uint32_t)(((old_state >> 18U) ^ old_state) >> 27U);
    rotation = (uint32_t)(old_state >> 59U);
    return (xorshifted >> rotation) |
           (xorshifted << ((0U - rotation) & 31U));
}

m3_status m3_rng_seed(m3_rng *rng, uint64_t seed, uint64_t sequence,
                      m3_error *error)
{
    if (rng == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RNG state is null");
    }

    rng->state = 0U;
    rng->increment = (sequence << 1U) | UINT64_C(1);
    rng->spare_normal = 0.0F;
    rng->has_spare_normal = false;
    (void)m3_rng_u32_raw(rng);
    rng->state += seed;
    (void)m3_rng_u32_raw(rng);
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_rng_u32(m3_rng *rng, uint32_t *value, m3_error *error)
{
    if (rng == NULL || value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RNG state and output must be non-null");
    }

    *value = m3_rng_u32_raw(rng);
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_rng_uniform_f32(m3_rng *rng, float *value, m3_error *error)
{
    uint32_t bits;

    if (rng == NULL || value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RNG state and uniform output must be non-null");
    }

    bits = m3_rng_u32_raw(rng) >> 8U;
    *value = (float)bits * 0x1.0p-24F;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_rng_normal_f32_fill(m3_rng *rng, float *values, size_t count,
                                 m3_error *error)
{
    static const double inverse_u32_range = 0x1.0p-32;
    static const double two_pi = 6.283185307179586476925286766559;
    size_t index = 0U;

    if (rng == NULL || (count != 0U && values == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RNG state and normal output must be non-null");
    }

    if (count != 0U && rng->has_spare_normal) {
        values[index] = rng->spare_normal;
        rng->spare_normal = 0.0F;
        rng->has_spare_normal = false;
        index += 1U;
    }

    while (index < count) {
        double uniform_radius =
            ((double)m3_rng_u32_raw(rng) + 0.5) * inverse_u32_range;
        double uniform_angle =
            (double)m3_rng_u32_raw(rng) * inverse_u32_range;
        double radius = sqrt(-2.0 * log(uniform_radius));
        double angle = two_pi * uniform_angle;
        float first = (float)(radius * cos(angle));
        float second = (float)(radius * sin(angle));

        values[index] = first;
        index += 1U;
        if (index < count) {
            values[index] = second;
            index += 1U;
        } else {
            rng->spare_normal = second;
            rng->has_spare_normal = true;
        }
    }

    m3_error_reset(error);
    return M3_STATUS_OK;
}
