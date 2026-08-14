/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rng.h"
#include "m3_wav.h"
#include "test_cases.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static bool m3_test_read_exact_file(const char *path, uint8_t *data,
                                    size_t size)
{
    FILE *stream = fopen(path, "rb");
    bool success;

    if (stream == NULL) {
        return false;
    }
    success = fread(data, 1U, size, stream) == size &&
              fgetc(stream) == EOF && ferror(stream) == 0;
    if (fclose(stream) != 0) {
        success = false;
    }
    return success;
}

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

void m3_test_wav_contract(m3_test_context *test)
{
    static const uint8_t expected[] = {
        0x52U, 0x49U, 0x46U, 0x46U, 0x2cU, 0x00U, 0x00U, 0x00U,
        0x57U, 0x41U, 0x56U, 0x45U, 0x66U, 0x6dU, 0x74U, 0x20U,
        0x10U, 0x00U, 0x00U, 0x00U, 0x03U, 0x00U, 0x02U, 0x00U,
        0x40U, 0x1fU, 0x00U, 0x00U, 0x00U, 0xfaU, 0x00U, 0x00U,
        0x08U, 0x00U, 0x20U, 0x00U, 0x64U, 0x61U, 0x74U, 0x61U,
        0x08U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U, 0x3fU,
        0x00U, 0x00U, 0x00U, 0xbfU
    };
    static const m3_test_fixture empty_fixture = {
        "empty WAVE destination", NULL, 0U
    };
    const float samples[] = {1.0F, -0.5F};
    uint8_t actual[sizeof(expected)] = {0U};
    m3_test_temp_file file = {{0}};
    m3_error error = {M3_STATUS_INTERNAL, "stale"};
    char invalid_path[M3_TEST_TEMP_PATH_CAPACITY + 16U] = {0};
    int path_result;

    M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &empty_fixture),
                   "create WAVE destination");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, 8000U, 2U, 1U,
                                    &error) == M3_STATUS_OK,
                   "write tiny float32 WAVE");
    M3_TEST_EXPECT(test, error.status == M3_STATUS_OK,
                   "successful WAVE write clears error");
    M3_TEST_EXPECT(test,
                   m3_test_read_exact_file(file.path, actual, sizeof(actual)),
                   "read exact tiny WAVE size");
    M3_TEST_EXPECT(test, memcmp(actual, expected, sizeof(expected)) == 0,
                   "tiny WAVE bytes match canonical little-endian output");

    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(NULL, samples, 8000U, 2U, 1U, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null WAVE path");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32("", samples, 8000U, 2U, 1U, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject empty WAVE path");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, NULL, 8000U, 2U, 1U,
                                    &error) == M3_STATUS_INVALID_ARGUMENT,
                   "reject null non-empty WAVE samples");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, 0U, 2U, 1U, &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject zero WAVE sample rate");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, 8000U, 0U, 1U,
                                    &error) == M3_STATUS_OUT_OF_RANGE,
                   "reject zero WAVE channels");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, 8000U, UINT16_MAX,
                                    0U, &error) == M3_STATUS_OVERFLOW,
                   "reject overflowing WAVE block alignment");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, UINT32_MAX, 1U, 0U,
                                    &error) == M3_STATUS_OVERFLOW,
                   "reject overflowing WAVE byte rate");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, 8000U, 2U,
                                    UINT64_MAX, &error) ==
                       M3_STATUS_OVERFLOW,
                   "reject overflowing WAVE sample count");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, samples, 8000U, 1U,
                                    (uint64_t)(UINT32_MAX - 36U) / 4U + 1U,
                                    &error) == M3_STATUS_OVERFLOW,
                   "reject WAVE data beyond RIFF limit");

    path_result = snprintf(invalid_path, sizeof(invalid_path), "%s/child.wav",
                           file.path);
    M3_TEST_EXPECT(test,
                   path_result > 0 && (size_t)path_result < sizeof(invalid_path),
                   "construct deterministic invalid WAVE path");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(invalid_path, samples, 8000U, 2U, 1U,
                                    &error) == M3_STATUS_IO,
                   "propagate WAVE open failure");
    M3_TEST_EXPECT(test, error.status == M3_STATUS_IO &&
                             m3_error_message(&error)[0] != '\0',
                   "WAVE I/O failure records structured error");
    M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                   "remove WAVE destination");
}
