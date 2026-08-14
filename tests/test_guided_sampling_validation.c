/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "guided_sampling_test.h"
#include "m3_guided_sampling.h"

#include <math.h>

void m3_test_guided_validation(m3_test_context *test)
{
    const uint64_t wrong_rows_shape[] = {1U, 1024U};
    const uint64_t wrong_codes_shape[] = {2U, 1023U};
    const int32_t integer_values[2048] = {0};
    const float float_values[2048] = {0};
    m3_op_test_fixture fixture;
    m3_op_test_fixture other;
    m3_tensor_view residual;
    m3_tensor_view wrong_rows;
    m3_tensor_view wrong_codes;
    m3_tensor_view integers;
    m3_tensor_view foreign;
    m3_tensor_view malformed;
    m3_error error;
    uint32_t code = 77U;
    bool created_other = false;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create guided validation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_test_tensor(&fixture, &residual, M3_DTYPE_F32,
                              M3_GUIDED_RESIDUAL_CODE_COUNT, 0.0F) &&
            m3_op_test_tensor(&fixture, &wrong_rows, M3_DTYPE_F32, 2U,
                              wrong_rows_shape, float_values) &&
            m3_op_test_tensor(&fixture, &wrong_codes, M3_DTYPE_F32, 2U,
                              wrong_codes_shape, float_values) &&
            m3_op_test_tensor(&fixture, &integers, M3_DTYPE_I32, 2U,
                              (const uint64_t[]){2U, 1024U},
                              integer_values),
        "create malformed guided inputs");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &wrong_rows, 0.0F,
                                  &code, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            code == 77U,
        "reject residual row count without changing output");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &wrong_codes, 0.0F,
                                  &code, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            code == 77U,
        "reject residual code shape with null error sink");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &integers, 0.0F, &code,
                                  &error) == M3_STATUS_INVALID_ARGUMENT &&
            code == 77U,
        "reject residual integer dtype without changing output");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &residual, NAN, &code,
                                  &error) == M3_STATUS_OUT_OF_RANGE &&
            code == 77U,
        "reject NaN uniform without changing output");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &residual, -0.1F, &code,
                                  NULL) == M3_STATUS_OUT_OF_RANGE &&
            code == 77U &&
            m3_guided_sample_residual(fixture.backend, &residual, 1.0F,
                                      &code, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            code == 77U &&
            m3_guided_sample_residual(fixture.backend, &residual, INFINITY,
                                      &code, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            code == 77U,
        "reject every non-half-open uniform boundary atomically");
    malformed = residual;
    malformed.byte_strides[0] = SIZE_MAX - 3U;
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &malformed, 0.0F, &code,
                                  &error) == M3_STATUS_OVERFLOW &&
            code == 77U,
        "reject overflowing residual layout without changing output");
    malformed = residual;
    malformed.storage = NULL;
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &malformed, 0.0F, &code,
                                  NULL) == M3_STATUS_INVALID_ARGUMENT &&
            code == 77U,
        "reject missing residual storage with a null error sink");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(NULL, &residual, 0.0F, &code, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            code == 77U &&
            m3_guided_sample_residual(fixture.backend, &residual, 0.0F,
                                      NULL, NULL) ==
                M3_STATUS_INVALID_ARGUMENT,
        "reject null guided backend and output with null error sinks");

    created_other = m3_op_test_fixture_init(&other);
    M3_TEST_EXPECT(test, created_other, "create foreign backend fixture");
    if (created_other) {
        M3_TEST_EXPECT(
            test,
            m3_guided_test_tensor(&other, &foreign, M3_DTYPE_F32,
                                  M3_GUIDED_RESIDUAL_CODE_COUNT, 0.0F),
            "create foreign backend logits");
        M3_TEST_EXPECT(
            test,
            m3_guided_sample_residual(fixture.backend, &foreign, 0.0F,
                                      &code, &error) ==
                    M3_STATUS_INVALID_ARGUMENT &&
                code == 77U,
            "reject logits from another backend without changing output");
        m3_op_test_fixture_dispose(&other);
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_residual(fixture.backend, &residual, 0.0F, &code,
                                  NULL) == M3_STATUS_OK &&
            code == 0U,
        "sample successfully with a null error sink");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_guided_semantic_validation(m3_test_context *test)
{
    m3_op_test_fixture fixture;
    m3_tensor_view eos;
    m3_tensor_view codes;
    m3_tensor_view wrong_eos;
    m3_guided_semantic_sample sample = {true, 91U};
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create semantic validation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_guided_test_tensor(&fixture, &eos, M3_DTYPE_F32, 1U, 0.0F) &&
            m3_guided_test_tensor(&fixture, &codes, M3_DTYPE_F32,
                                  M3_GUIDED_SEMANTIC_CODE_COUNT, 0.0F) &&
            m3_guided_test_tensor(&fixture, &wrong_eos, M3_DTYPE_F32, 2U,
                                  0.0F),
        "create semantic validation inputs");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(fixture.backend, &wrong_eos, &codes, 0.0F,
                                  &sample, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            sample.eos && sample.code == 91U,
        "reject malformed EOS shape without changing semantic output");
    m3_guided_test_set(&codes, 0U, 19U, NAN);
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(fixture.backend, &eos, &codes, 0.0F,
                                  &sample, NULL) == M3_STATUS_OUT_OF_RANGE &&
            sample.eos && sample.code == 91U,
        "reject conditional semantic NaN atomically with null error sink");
    M3_TEST_EXPECT(
        test,
        m3_guided_sample_semantic(fixture.backend, NULL, &codes, 0.0F,
                                  &sample, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            sample.eos && sample.code == 91U,
        "reject null semantic EOS view without changing output");
    m3_op_test_fixture_dispose(&fixture);
}
