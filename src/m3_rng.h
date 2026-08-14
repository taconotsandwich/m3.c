/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_RNG_H
#define M3_RNG_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare_normal;
    bool has_spare_normal;
} m3_rng;

m3_status m3_rng_seed(m3_rng *rng, uint64_t seed, uint64_t sequence,
                      m3_error *error);
m3_status m3_rng_u32(m3_rng *rng, uint32_t *value, m3_error *error);
m3_status m3_rng_uniform_f32(m3_rng *rng, float *value, m3_error *error);
m3_status m3_rng_normal_f32_fill(m3_rng *rng, float *values, size_t count,
                                 m3_error *error);

#endif
