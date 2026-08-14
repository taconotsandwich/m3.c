/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"
#include "m3_rvq_condition_internal.h"
#include "m3_test.h"
#include "rvq_condition_test.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static bool m3_rvq_feedback_fixture_prepare(
    m3_rvq_test_fixture *fixture, m3_rvq_frame *frame,
    uint32_t expected_codes[M3_RVQ_RESIDUAL_COUNT])
{
    static const float first_channel[M3_RVQ_RESIDUAL_COUNT] = {
        1.0F,
        0.000652313232421875F,
        0.000652313232421875F,
        0.000652313232421875F,
        0.000652313232421875F,
        0.000652313232421875F,
        0.000652313232421875F
    };
    static const float second_channel[M3_RVQ_RESIDUAL_COUNT] = {
        0.0F, 0.0078125F, 0.0078125F, 0.0078125F,
        0.0078125F, 0.0078125F, 0.0078125F
    };
    float uniforms[M3_RVQ_RESIDUAL_COUNT];
    m3_error error;
    size_t index;

    m3_rvq_test_uniforms(uniforms, expected_codes);
    m3_rvq_frame_init(frame);
    if (m3_rvq_decode_frame_core(
            fixture->fixture.backend, &fixture->config,
            &fixture->weights, &fixture->last_hidden,
            &fixture->semantic_embedding, uniforms,
            M3_RVQ_RESIDUAL_COUNT, NULL, NULL, frame,
            &error) != M3_STATUS_OK) {
        return false;
    }
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
        size_t row = index * M3_RVQ_CODEBOOK_SIZE + frame->codes[index];

        m3_rc_set((m3_tensor_view *)fixture->weights.audio_embeddings,
                  row * 2U, first_channel[index]);
        m3_rc_set((m3_tensor_view *)fixture->weights.audio_embeddings,
                  row * 2U + 1U, second_channel[index]);
    }
    m3_rc_set(&fixture->semantic_embedding, 0U, 0.043212890625F);
    m3_rc_set(&fixture->semantic_embedding, 1U, 1.0F);
    return true;
}

static const uint16_t *m3_rvq_feedback_bits(
    const m3_rvq_feedback *feedback)
{
    const uint8_t *data = m3_storage_const_data(feedback->tensor.storage);

    return (const uint16_t *)(data + feedback->tensor.byte_offset);
}

void m3_test_rvq_feedback_oracle(m3_test_context *test)
{
    static const uint16_t expected[] = {
        0x3ebfU, 0x3ebdU, 0x3ebfU, 0x3ebdU
    };
    m3_rvq_test_fixture fixture;
    m3_rvq_frame frame;
    m3_rvq_feedback feedback;
    uint32_t expected_codes[M3_RVQ_RESIDUAL_COUNT];
    const uint16_t *bits;
    m3_error error;
    bool ready = m3_rvq_test_fixture_init(&fixture, M3_DTYPE_BF16);

    m3_rvq_feedback_init(&feedback);
    M3_TEST_EXPECT(test, ready, "create reduced RVQ feedback fixture");
    if (!ready) {
        return;
    }
    ready = m3_rvq_feedback_fixture_prepare(
        &fixture, &frame, expected_codes);
    M3_TEST_EXPECT(test, ready,
                   "decode complete frame with endpoint residual codes");
    if (!ready) {
        m3_rvq_frame_dispose(&frame);
        m3_rvq_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        memcmp(frame.codes, expected_codes, sizeof(frame.codes)) == 0 &&
            expected_codes[0] == 0U && expected_codes[6] == 1023U,
        "preserve all seven codebooks including code endpoints");
    ready = m3_rvq_feedback_build_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.semantic_embedding, &frame,
                &feedback, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "build complete repeated Qwen feedback atomically");
    if (ready) {
        bits = m3_rvq_feedback_bits(&feedback);
        M3_TEST_EXPECT(
            test,
            feedback.storage != NULL &&
                feedback.tensor.storage == feedback.storage &&
                feedback.tensor.metadata.dtype == M3_DTYPE_BF16 &&
                feedback.tensor.metadata.rank == 3U &&
                feedback.tensor.metadata.shape[0] == 2U &&
                feedback.tensor.metadata.shape[1] == 1U &&
                feedback.tensor.metadata.shape[2] == 2U &&
                memcmp(bits, expected, sizeof(expected)) == 0,
            "lock ordered residual sum, semantic add, BF16 0x3eb5 "
            "scale, and repeated rows");
    }
    m3_rvq_feedback_dispose(&feedback);
    m3_rvq_frame_dispose(&frame);
    m3_rvq_test_fixture_dispose(&fixture);
}

void m3_test_rvq_feedback_atomic_validation(m3_test_context *test)
{
    m3_rvq_test_fixture fixture;
    m3_rvq_frame frame;
    m3_rvq_frame bad_frame;
    m3_rvq_feedback feedback;
    m3_rvq_weights bad_weights;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    uint32_t expected_codes[M3_RVQ_RESIDUAL_COUNT];
    uint16_t preserved[4];
    m3_storage *preserved_storage;
    uint32_t preserved_code;
    m3_error error;
    bool ready = m3_rvq_test_fixture_init(&fixture, M3_DTYPE_BF16);

    m3_rvq_feedback_init(&feedback);
    M3_TEST_EXPECT(test, ready,
                   "create RVQ feedback validation fixture");
    if (!ready) {
        return;
    }
    ready = m3_rvq_feedback_fixture_prepare(
                &fixture, &frame, expected_codes) &&
            m3_rvq_feedback_build_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.semantic_embedding, &frame,
                &feedback, &error) == M3_STATUS_OK &&
            m3_backend_get_allocation_stats(
                fixture.fixture.backend, &before, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "establish owned RVQ feedback publication");
    if (!ready) {
        m3_rvq_feedback_dispose(&feedback);
        m3_rvq_frame_dispose(&frame);
        m3_rvq_test_fixture_dispose(&fixture);
        return;
    }
    preserved_storage = feedback.storage;
    (void)memcpy(preserved, m3_rvq_feedback_bits(&feedback),
                 sizeof(preserved));
    preserved_code = frame.codes[6];
    frame.codes[6] = M3_RVQ_CODEBOOK_SIZE;
    M3_TEST_EXPECT(
        test,
        m3_rvq_feedback_build_core(
            fixture.fixture.backend, &fixture.config, &fixture.weights,
            &fixture.semantic_embedding, &frame, &feedback,
            NULL) == M3_STATUS_OUT_OF_RANGE &&
            feedback.storage == preserved_storage &&
            memcmp(m3_rvq_feedback_bits(&feedback), preserved,
                   sizeof(preserved)) == 0,
        "reject code 1024 with null error and preserve feedback");
    frame.codes[6] = preserved_code;
    bad_weights = fixture.weights;
    bad_weights.audio_embeddings = &fixture.semantic_embedding;
    bad_frame = frame;
    bad_frame.storage = NULL;
    M3_TEST_EXPECT(
        test,
        m3_rvq_feedback_build_core(
            fixture.fixture.backend, &fixture.config, &bad_weights,
            &fixture.semantic_embedding, &frame, &feedback,
            NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_rvq_feedback_build_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.last_hidden, &frame,
                &feedback, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_rvq_feedback_build_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.semantic_embedding, &bad_frame,
                &feedback, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_rvq_feedback_build_core(
                NULL, &fixture.config, &fixture.weights,
                &fixture.semantic_embedding, &frame, &feedback,
                NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_rvq_feedback_build(
                NULL, &fixture.weights, &fixture.semantic_embedding,
                &frame, &feedback, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            feedback.storage == preserved_storage,
        "reject malformed feedback inputs without replacing output");
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(
            fixture.fixture.backend, &after, &error) == M3_STATUS_OK &&
            after.live_storage_count == before.live_storage_count &&
            after.live_allocated_bytes == before.live_allocated_bytes &&
            memcmp(m3_rvq_feedback_bits(&feedback), preserved,
                   sizeof(preserved)) == 0,
        "fail validation before allocations and preserve owned storage");
    m3_rvq_feedback_dispose(&feedback);
    m3_rvq_frame_dispose(&frame);
    m3_rvq_test_fixture_dispose(&fixture);
}
