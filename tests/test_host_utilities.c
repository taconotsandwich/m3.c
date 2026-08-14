/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rng.h"
#include "test_cases.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

void m3_test_rng_contract(m3_test_context *test)
{
    static const uint32_t expected[] = {
        UINT32_C(0xa15c02b7), UINT32_C(0x7b47f409),
        UINT32_C(0xba1d3330), UINT32_C(0x83d2f293),
        UINT32_C(0xbfa4784b)
    };
    m3_rng rng = {0};
    m3_rng reference = {0};
    m3_error error = {M3_STATUS_INTERNAL, "stale"};
    float normal_all[3] = {0.0F};
    float normal_split[3] = {0.0F};
    float reseeded[2] = {0.0F};
    float reseeded_reference[2] = {0.0F};
    uint32_t value = 0U;
    float uniform = 0.0F;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_rng_seed(&rng, UINT64_C(42), UINT64_C(54), &error) ==
                       M3_STATUS_OK,
                   "seed PCG32 state");
    M3_TEST_EXPECT(test, error.status == M3_STATUS_OK,
                   "successful RNG seed clears error");
    for (index = 0U; index < sizeof(expected) / sizeof(expected[0]); ++index) {
        M3_TEST_EXPECT(test,
                       m3_rng_u32(&rng, &value, &error) == M3_STATUS_OK,
                       "generate PCG32 value");
        M3_TEST_EXPECT(test, value == expected[index],
                       "PCG32 value matches known sequence");
    }

    M3_TEST_EXPECT(test,
                   m3_rng_seed(&rng, UINT64_C(42), UINT64_C(54), &error) ==
                       M3_STATUS_OK,
                   "reseed before uniform draw");
    M3_TEST_EXPECT(test,
                   m3_rng_uniform_f32(&rng, &uniform, &error) ==
                       M3_STATUS_OK,
                   "generate uniform float");
    M3_TEST_EXPECT(test,
                   uniform ==
                       (float)(expected[0] >> 8U) * 0x1.0p-24F,
                   "uniform float uses deterministic high-bit mapping");
    M3_TEST_EXPECT(test, uniform >= 0.0F && uniform < 1.0F,
                   "uniform float lies in half-open unit interval");

    M3_TEST_EXPECT(test,
                   m3_rng_seed(&rng, UINT64_C(123), UINT64_C(7), &error) ==
                       M3_STATUS_OK &&
                       m3_rng_seed(&reference, UINT64_C(123), UINT64_C(7),
                                   &error) == M3_STATUS_OK,
                   "seed matching normal streams");
    M3_TEST_EXPECT(test,
                   m3_rng_normal_f32_fill(&rng, normal_all, 3U, &error) ==
                       M3_STATUS_OK,
                   "fill odd normal sample count");
    M3_TEST_EXPECT(test,
                   m3_rng_normal_f32_fill(&reference, normal_split, 1U,
                                          &error) == M3_STATUS_OK &&
                       m3_rng_normal_f32_fill(&reference, normal_split + 1U,
                                              2U, &error) == M3_STATUS_OK,
                   "consume cached normal across split fills");
    M3_TEST_EXPECT(test,
                   memcmp(normal_all, normal_split, sizeof(normal_all)) == 0,
                   "split normal fill preserves exact stream");
    for (index = 0U; index < 3U; ++index) {
        M3_TEST_EXPECT(test, isfinite(normal_all[index]),
                       "normal sample is finite");
    }

    M3_TEST_EXPECT(test,
                   m3_rng_seed(&rng, UINT64_C(9), UINT64_C(11), &error) ==
                           M3_STATUS_OK &&
                       m3_rng_normal_f32_fill(&rng, normal_all, 1U, &error) ==
                           M3_STATUS_OK &&
                       m3_rng_seed(&rng, UINT64_C(9), UINT64_C(11), &error) ==
                           M3_STATUS_OK &&
                       m3_rng_normal_f32_fill(&rng, reseeded, 2U, &error) ==
                           M3_STATUS_OK &&
                       m3_rng_seed(&reference, UINT64_C(9), UINT64_C(11),
                                   &error) == M3_STATUS_OK &&
                       m3_rng_normal_f32_fill(&reference,
                                              reseeded_reference, 2U,
                                              &error) == M3_STATUS_OK,
                   "reseed normal streams after a cached spare");
    M3_TEST_EXPECT(test,
                   memcmp(reseeded, reseeded_reference, sizeof(reseeded)) == 0,
                   "RNG seed clears cached normal");

    M3_TEST_EXPECT(test,
                   m3_rng_seed(NULL, 0U, 0U, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null RNG seed target");
    M3_TEST_EXPECT(test,
                   m3_rng_u32(NULL, &value, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null RNG state");
    M3_TEST_EXPECT(test,
                   m3_rng_u32(&rng, NULL, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null integer output");
    M3_TEST_EXPECT(test,
                   m3_rng_uniform_f32(&rng, NULL, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null uniform output");
    M3_TEST_EXPECT(test,
                   m3_rng_normal_f32_fill(&rng, NULL, 1U, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null non-empty normal output");
    M3_TEST_EXPECT(test,
                   m3_rng_normal_f32_fill(&rng, NULL, 0U, &error) ==
                       M3_STATUS_OK,
                   "allow empty normal output");
}
