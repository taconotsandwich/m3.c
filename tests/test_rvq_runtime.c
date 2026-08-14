/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_internal.h"
#include "m3_rvq_condition_internal.h"
#include "rvq_condition_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static float m3_rvq_expected_norm(float first, float second,
                                  size_t channel, m3_dtype dtype)
{
    float square_sum = first * first + second * second;
    float inverse = 1.0F / sqrtf(square_sum * 0.5F + 1.0e-6F);
    float value = channel == 0U ? first : second;

    return m3_op_round_float(
        dtype, m3_op_round_float(dtype, value * inverse));
}

void m3_test_rvq_decode_schedule(m3_test_context *test)
{
    m3_rvq_test_fixture fixture;
    m3_runtime_progress_log progress = {0};
    float uniforms[M3_RVQ_RESIDUAL_COUNT];
    uint32_t expected_codes[M3_RVQ_RESIDUAL_COUNT];
    m3_rvq_frame frame;
    m3_storage *churn = NULL;
    m3_error error;
    size_t index;
    bool ready = m3_rvq_test_fixture_init(&fixture, M3_DTYPE_BF16);

    M3_TEST_EXPECT(test, ready, "create reduced BF16 RVQ runtime fixture");
    if (!ready) {
        return;
    }
    progress.cancel_at = UINT64_MAX;
    m3_rvq_test_uniforms(uniforms, expected_codes);
    m3_rvq_frame_init(&frame);
    M3_TEST_EXPECT(
        test,
        m3_rvq_decode_frame_core(
            fixture.fixture.backend, &fixture.config, &fixture.weights,
            &fixture.last_hidden, &fixture.semantic_embedding, uniforms,
            M3_RVQ_RESIDUAL_COUNT, m3_rc_progress, &progress, &frame,
            &error) == M3_STATUS_OK,
        "execute the production RVQ schedule at reduced dimensions");
    M3_TEST_EXPECT(
        test,
        frame.storage != NULL && frame.conditioning.metadata.rank == 3U &&
            frame.conditioning.metadata.dtype == M3_DTYPE_BF16 &&
            frame.conditioning.metadata.shape[0] == 1U &&
            frame.conditioning.metadata.shape[1] == 8U &&
            frame.conditioning.metadata.shape[2] == 2U,
        "publish one owned [global,c1..c7] BF16 frame");
    M3_TEST_EXPECT(
        test,
        memcmp(frame.codes, expected_codes, sizeof(frame.codes)) == 0,
        "seven uniform draws map through tied 1024-way residual heads");
    M3_TEST_EXPECT(
        test,
        progress.call_count == 8U && progress.completed[0] == 0U &&
            progress.completed[7] == 7U && progress.total[0] == 7U &&
            progress.total[7] == 7U,
        "RVQ progress covers the initial gate and seven residual heads");
    M3_TEST_EXPECT(
        test,
        m3_rc_get(&frame.conditioning, 0U) == 3.0F &&
            m3_rc_get(&frame.conditioning, 1U) == 4.0F,
        "c0 preserves unprojected conditional last_hidden row zero");
    for (index = 1U; index < M3_RVQ_CODEBOOK_COUNT; ++index) {
        float first;
        float second;
        float actual_first = m3_rc_get(
            &frame.conditioning, index * 2U);
        float actual_second = m3_rc_get(
            &frame.conditioning, index * 2U + 1U);

        if (index == 1U) {
            first = 0.5F + 0.25F;
            second = 1.0F + 0.125F;
        } else {
            float book = (float)(index - 1U);

            first = book + (float)index * 0.25F;
            second = -0.5F * book + (float)index * 0.125F;
        }
        M3_TEST_EXPECT(
            test,
            actual_first == m3_rvq_expected_norm(
                                first, second, 0U, M3_DTYPE_BF16) &&
                actual_second == m3_rvq_expected_norm(
                                     first, second, 1U, M3_DTYPE_BF16),
            "depth conditioning uses the exact learned position and prior-book offset");
    }
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
        int32_t embedding_id = -1;
        bool has_next = m3_rvq_next_embedding_id(
            index, expected_codes[index], &embedding_id);

        M3_TEST_EXPECT(
            test,
            m3_rvq_sequence_length(index) == index + 2U &&
                (index == 6U
                     ? (!has_next && embedding_id == -1)
                     : (has_next &&
                        embedding_id ==
                            (int32_t)(index * M3_RVQ_CODEBOOK_SIZE +
                                      expected_codes[index]))),
            "production schedule exposes lengths 2..8 and embeds only c1..c6");
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_storage(&fixture.fixture, 4096U, &churn) &&
            m3_storage_write(churn, 0U,
                             (const uint8_t[16]){0}, 16U, &error) ==
                M3_STATUS_OK &&
            m3_rc_get(&frame.conditioning, 0U) == 3.0F,
        "published conditioning survives transient workspace storage reuse");
    m3_rvq_frame_dispose(&frame);
    m3_rvq_test_fixture_dispose(&fixture);
}

static bool m3_rvq_strided_inputs(m3_rvq_test_fixture *fixture,
                                  m3_tensor_view *last,
                                  m3_tensor_view *semantic)
{
    const uint64_t last_shape[] = {2U, 2U};
    const uint64_t semantic_shape[] = {2U};
    const size_t last_strides[] = {16U, 4U};
    const size_t semantic_strides[] = {4U};
    m3_storage *last_storage = NULL;
    m3_storage *semantic_storage = NULL;
    m3_error error;

    m3_tensor_view_init(last);
    m3_tensor_view_init(semantic);
    if (!m3_op_test_storage(&fixture->fixture, 24U, &last_storage) ||
        !m3_op_test_storage(&fixture->fixture, 8U, &semantic_storage) ||
        m3_tensor_view_strided(
            last, last_storage, M3_DTYPE_BF16, 2U, last_shape,
            last_strides, 0U, &error) != M3_STATUS_OK ||
        m3_tensor_view_strided(
            semantic, semantic_storage, M3_DTYPE_BF16, 1U,
            semantic_shape, semantic_strides, 0U, &error) != M3_STATUS_OK) {
        return false;
    }
    m3_rc_set(last, 0U, 3.0F);
    m3_rc_set(last, 1U, 4.0F);
    m3_rc_set(last, 2U, -2.0F);
    m3_rc_set(last, 3U, 1.0F);
    m3_rc_set(semantic, 0U, 0.5F);
    m3_rc_set(semantic, 1U, 1.0F);
    return true;
}

void m3_test_rvq_atomic_validation(m3_test_context *test)
{
    m3_rvq_test_fixture fixture;
    m3_runtime_progress_log cancel = {0};
    float uniforms[M3_RVQ_RESIDUAL_COUNT];
    float invalid_uniforms[M3_RVQ_RESIDUAL_COUNT];
    uint32_t expected_codes[M3_RVQ_RESIDUAL_COUNT];
    uint32_t preserved_codes[M3_RVQ_RESIDUAL_COUNT];
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_tensor_view strided_last;
    m3_tensor_view strided_semantic;
    m3_rvq_frame frame;
    m3_rvq_config bad_config;
    m3_rvq_weights bad_weights;
    m3_storage *preserved_storage;
    float preserved_value;
    int32_t sentinel = 77;
    m3_error error;
    bool ready = m3_rvq_test_fixture_init(&fixture, M3_DTYPE_BF16);

    M3_TEST_EXPECT(test, ready,
                   "create RVQ atomic-validation fixture");
    if (!ready) {
        return;
    }
    m3_rvq_test_uniforms(uniforms, expected_codes);
    m3_rvq_frame_init(&frame);
    ready = m3_rvq_decode_frame_core(
                fixture.fixture.backend, &fixture.config, &fixture.weights,
                &fixture.last_hidden, &fixture.semantic_embedding, uniforms,
                7U, NULL, NULL, &frame, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "establish published RVQ state");
    if (!ready) {
        m3_rvq_test_fixture_dispose(&fixture);
        return;
    }
    preserved_storage = frame.storage;
    preserved_value = m3_rc_get(&frame.conditioning, 0U);
    (void)memcpy(preserved_codes, frame.codes, sizeof(preserved_codes));
    (void)memcpy(invalid_uniforms, uniforms, sizeof(invalid_uniforms));
    invalid_uniforms[4] = NAN;
    (void)m3_backend_get_allocation_stats(
        fixture.fixture.backend, &before, &error);
    M3_TEST_EXPECT(
        test,
        m3_rvq_decode_frame_core(
            fixture.fixture.backend, &fixture.config, &fixture.weights,
            &fixture.last_hidden, &fixture.semantic_embedding,
            invalid_uniforms, 7U, NULL, NULL, &frame, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            frame.storage == preserved_storage &&
            m3_rc_get(&frame.conditioning, 0U) == preserved_value &&
            memcmp(frame.codes, preserved_codes, sizeof(frame.codes)) == 0,
        "invalid uniform fails before work and preserves published state");
    bad_weights = fixture.weights;
    bad_weights.heads[3] = &fixture.last_hidden;
    bad_config = fixture.config;
    bad_config.dtype = M3_DTYPE_F32;
    M3_TEST_EXPECT(
        test,
        m3_rvq_decode_frame_core(
            fixture.fixture.backend, &fixture.config, &bad_weights,
            &fixture.last_hidden, &fixture.semantic_embedding, uniforms,
            7U, NULL, NULL, &frame, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            frame.storage == preserved_storage &&
            m3_rvq_decode_frame_core(
                fixture.fixture.backend, &bad_config, &fixture.weights,
                &fixture.last_hidden, &fixture.semantic_embedding, uniforms,
                7U, NULL, NULL, &frame, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            frame.storage == preserved_storage,
        "wrong weight shape and dtype are atomic without an error sink");
    cancel.cancel_at = 3U;
    M3_TEST_EXPECT(
        test,
        m3_rvq_decode_frame_core(
            fixture.fixture.backend, &fixture.config, &fixture.weights,
            &fixture.last_hidden, &fixture.semantic_embedding, uniforms,
            7U, m3_rc_progress, &cancel, &frame, &error) ==
                M3_STATUS_CANCELLED &&
            cancel.call_count == 4U && frame.storage == preserved_storage &&
            memcmp(frame.codes, preserved_codes, sizeof(frame.codes)) == 0 &&
            m3_backend_get_allocation_stats(
                fixture.fixture.backend, &after, &error) == M3_STATUS_OK &&
            after.live_allocated_bytes == before.live_allocated_bytes &&
            after.live_storage_count == before.live_storage_count,
        "mid-frame cancellation cleans transient state and preserves output");
    ready = m3_rvq_strided_inputs(
        &fixture, &strided_last, &strided_semantic);
    M3_TEST_EXPECT(test, ready, "construct valid strided RVQ handoff views");
    if (ready) {
        M3_TEST_EXPECT(
            test,
            m3_rvq_decode_frame_core(
                fixture.fixture.backend, &fixture.config, &fixture.weights,
                &strided_last, &strided_semantic, uniforms, 7U, NULL, NULL,
                &frame, &error) == M3_STATUS_OK &&
                memcmp(frame.codes, expected_codes, sizeof(frame.codes)) ==
                    0 &&
                m3_rc_get(&frame.conditioning, 0U) == 3.0F,
            "RVQ production path consumes strided hidden and semantic inputs");
    }
    M3_TEST_EXPECT(
        test,
        !m3_rvq_next_embedding_id(0U, M3_RVQ_CODEBOOK_SIZE, &sentinel) &&
            sentinel == 77 &&
            !m3_rvq_next_embedding_id(6U, 0U, &sentinel) &&
            sentinel == 77 &&
            !m3_rvq_next_embedding_id(SIZE_MAX, 0U, &sentinel) &&
            sentinel == 77 && m3_rvq_sequence_length(7U) == 0U,
        "invalid codes and the final c7 code never produce an embedding ID");
    m3_rvq_frame_dispose(&frame);
    m3_rvq_test_fixture_dispose(&fixture);
}
