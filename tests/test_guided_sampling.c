/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "guided_sampling_test.h"
#include "m3_guided_sampling.h"

#include <math.h>

static bool m3_guided_test_semantic_pair(m3_op_test_fixture *fixture,
                                         m3_tensor_view *eos,
                                         m3_tensor_view *codes,
                                         m3_dtype eos_dtype,
                                         m3_dtype code_dtype,
                                         float initial)
{
    return m3_guided_test_tensor(fixture, eos, eos_dtype, 1U, initial) &&
           m3_guided_test_tensor(fixture, codes, code_dtype,
                                 M3_GUIDED_SEMANTIC_CODE_COUNT, initial);
}

void m3_test_guided_cfg_distribution(m3_test_context *test)
{
    const float first_probability = 0.982013762F;
    m3_op_test_fixture fixture;
    m3_tensor_view logits;
    m3_error error;
    uint32_t code = UINT32_MAX;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create guided CFG fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_test_tensor(&fixture, &logits, M3_DTYPE_F32,
                              M3_GUIDED_RESIDUAL_CODE_COUNT, -INFINITY),
        "create guided CFG logits");
    m3_guided_test_set(&logits, 0U, 0U, 2.0F);
    m3_guided_test_set(&logits, 1U, 0U, 0.0F);
    m3_guided_test_set(&logits, 0U, 1U, 0.0F);
    m3_guided_test_set(&logits, 1U, 1U, 2.0F);
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(
            fixture.backend, &logits,
            nextafterf(first_probability, 0.0F), &code, &error) ==
                M3_STATUS_OK &&
            code == 0U,
        "CFG [3,-1] selects inside the pinned first probability");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &logits,
                                  first_probability, &code, &error) ==
                M3_STATUS_OK &&
            code == 1U,
        "CFG [3,-1] uses the pinned strict CDF boundary");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_guided_semantic_masks(m3_test_context *test)
{
    m3_op_test_fixture fixture;
    m3_tensor_view eos;
    m3_tensor_view codes;
    m3_guided_semantic_sample sample = {true, UINT32_MAX};
    m3_error error;
    size_t candidate;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create semantic mask fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_guided_test_semantic_pair(
                       &fixture, &eos, &codes, M3_DTYPE_F32, M3_DTYPE_F32,
                       -INFINITY),
                   "create semantic first-mask logits");
    m3_guided_test_set(&eos, 0U, 0U, 1.0F);
    m3_guided_test_set(&eos, 1U, 0U, 1.0F);
    for (candidate = 0U; candidate < 49U; ++candidate) {
        m3_guided_test_set(&codes, 0U, candidate, 1.0F);
        m3_guided_test_set(&codes, 1U, candidate, 1.0F);
    }
    for (candidate = 49U; candidate < 51U; ++candidate) {
        m3_guided_test_set(&codes, 0U, candidate, 0.0F);
        m3_guided_test_set(&codes, 1U, candidate, -100.0F);
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(
            fixture.backend, &eos, &codes, nextafterf(1.0F, 0.0F),
            &sample, &error) == M3_STATUS_OK &&
            !sample.eos && sample.code == 48U,
        "semantic conditional top-50 mask excludes two guided maxima");

    m3_guided_test_set(&eos, 0U, 0U, 1.0F);
    m3_guided_test_set(&eos, 1U, 0U, 1.0F);
    for (candidate = 0U; candidate < 48U; ++candidate) {
        m3_guided_test_set(&codes, 0U, candidate, 1.0F);
        m3_guided_test_set(&codes, 1U, candidate, 1.0F);
    }
    for (candidate = 48U; candidate < 51U; ++candidate) {
        m3_guided_test_set(&codes, 0U, candidate, 0.0F);
        m3_guided_test_set(&codes, 1U, candidate, 0.0F);
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(fixture.backend, &eos, &codes, 0.999F,
                                  &sample, &error) == M3_STATUS_OK &&
            !sample.eos && sample.code == 50U,
        "strict top-50 threshold retains all three zero ties");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_guided_boundaries_and_dtypes(m3_test_context *test)
{
    static const m3_dtype dtypes[] = {
        M3_DTYPE_F16, M3_DTYPE_BF16, M3_DTYPE_F32
    };
    m3_op_test_fixture fixture;
    m3_tensor_view eos;
    m3_tensor_view codes;
    m3_tensor_view residual;
    m3_tensor_view strided = {0};
    m3_storage *strided_storage = NULL;
    m3_guided_semantic_sample sample = {false, UINT32_MAX};
    m3_error error;
    uint32_t code = UINT32_MAX;
    size_t dtype_index;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create guided boundary fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_guided_test_semantic_pair(
                       &fixture, &eos, &codes, M3_DTYPE_F16,
                       M3_DTYPE_BF16, -INFINITY),
                   "create mixed-precision semantic logits");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(fixture.backend, &eos, &codes, 0.0F,
                                  &sample, &error) == M3_STATUS_OK &&
            sample.eos && sample.code == 0U,
        "semantic candidate zero maps atomically to EOS");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(
            fixture.backend, &eos, &codes,
            nextafterf(1.0F / 16385.0F, 1.0F), &sample, &error) ==
                M3_STATUS_OK &&
            !sample.eos && sample.code == 0U,
        "semantic candidate one maps to code zero");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(
            fixture.backend, &eos, &codes, nextafterf(1.0F, 0.0F),
            &sample, &error) == M3_STATUS_OK &&
            !sample.eos && sample.code == 16383U,
        "semantic final candidate maps to code 16383");

    for (dtype_index = 0U;
         dtype_index < sizeof(dtypes) / sizeof(dtypes[0]); ++dtype_index) {
        M3_TEST_EXPECT(
            test,
            m3_guided_test_tensor(&fixture, &residual, dtypes[dtype_index],
                                  M3_GUIDED_RESIDUAL_CODE_COUNT, 0.0F),
            "create residual dtype logits");
        M3_TEST_EXPECT(
            test,
            m3_guided_sample_residual(fixture.backend, &residual, 0.0F,
                                      &code, &error) == M3_STATUS_OK &&
                code == 0U,
            "residual uniform zero selects code zero");
        M3_TEST_EXPECT(
            test,
            m3_guided_sample_residual(
                fixture.backend, &residual, nextafterf(1.0F, 0.0F),
                &code, &error) == M3_STATUS_OK &&
                code == 1023U,
            "residual near-one uniform selects code 1023");
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_storage(&fixture, 16384U, &strided_storage) &&
            m3_tensor_view_strided(
                &strided, strided_storage, M3_DTYPE_F32, 2U,
                (const uint64_t[]){2U, 1024U},
                (const size_t[]){8192U, 8U}, 0U, &error) == M3_STATUS_OK,
        "create valid padded residual layout");
    if (strided.storage != NULL) {
        for (index = 0U; index < strided.metadata.element_count; ++index) {
            m3_guided_test_set(&strided, index / 1024U, index % 1024U,
                               0.0F);
        }
        M3_TEST_EXPECT(
            test,
            m3_guided_sample_residual(fixture.backend, &strided, 0.5F,
                                      &code, &error) == M3_STATUS_OK &&
                code == 512U,
            "guided sampler reads validated strided logits in row order");
    }
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_guided_nonfinite_common(m3_test_context *test)
{
    m3_op_test_fixture fixture;
    m3_tensor_view logits;
    m3_error error;
    uint32_t code = UINT32_MAX;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create guided nonfinite fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_test_tensor(&fixture, &logits, M3_DTYPE_F32,
                              M3_GUIDED_RESIDUAL_CODE_COUNT, -INFINITY),
        "create guided nonfinite logits");
    m3_guided_test_set(&logits, 0U, 0U, 0.0F);
    m3_guided_test_set(&logits, 1U, 0U, NAN);
    m3_guided_test_set(&logits, 0U, 1U, INFINITY);
    m3_guided_test_set(&logits, 1U, 1U, 0.0F);
    m3_guided_test_set(&logits, 0U, 2U, -INFINITY);
    m3_guided_test_set(&logits, 1U, 2U, 0.0F);
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &logits, 0.5F, &code,
                                  &error) == M3_STATUS_OK &&
            code == 1U,
        "common sampler maps NaN and infinities to finite sentinels");
    m3_op_test_fixture_dispose(&fixture);
}
