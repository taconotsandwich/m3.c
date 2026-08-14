/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_test.h"
#include "qwen_runtime_fixture.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    const m3_qwen_forward_state *state;
    uint64_t cancel_at;
    uint64_t completed[8];
    uint64_t totals[8];
    uint64_t observed_counts[8];
    size_t call_count;
} m3_qwen_progress_log;

static bool m3_qwen_test_progress(void *context, uint64_t completed,
                                  uint64_t total)
{
    m3_qwen_progress_log *log = context;
    size_t index = log->call_count;

    if (index < sizeof(log->completed) / sizeof(log->completed[0])) {
        log->completed[index] = completed;
        log->totals[index] = total;
        log->observed_counts[index] = log->state->token_count;
    }
    ++log->call_count;
    return completed != log->cancel_at;
}

static bool m3_qwen_f32_values(const m3_tensor_view *view,
                               const float *expected, size_t count)
{
    size_t index;

    if (view == NULL || view->metadata.dtype != M3_DTYPE_F32 ||
        view->metadata.element_count != count) {
        return false;
    }
    for (index = 0U; index < count; ++index) {
        if (m3_qwen_test_f32_at(view, index) != expected[index]) {
            return false;
        }
    }
    return true;
}

static bool m3_qwen_bf16_values(const m3_tensor_view *view,
                                const uint16_t *expected, size_t count)
{
    size_t index;

    if (view == NULL || view->metadata.dtype != M3_DTYPE_BF16 ||
        view->metadata.element_count != count) {
        return false;
    }
    for (index = 0U; index < count; ++index) {
        if (m3_qwen_test_bf16_at(view, index) != expected[index]) {
            return false;
        }
    }
    return true;
}

static bool m3_qwen_cache_values(const m3_tensor_view *view,
                                 const uint16_t *expected, size_t count)
{
    size_t index;

    if (view == NULL || view->metadata.element_count != count) {
        return false;
    }
    for (index = 0U; index < count; ++index) {
        if (m3_qwen_test_bf16_at(view, index) != expected[index]) {
            return false;
        }
    }
    return true;
}

void m3_test_qwen_forward_oracle(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const int32_t step_ids[] = {4, 4};
    static const float prefill_eos[] = {-0.90625F, -0.05322265625F};
    static const float prefill_semantic[] = {
        2.0F, 2.21875F, -1.453125F,
        -1.3828125F, 1.828125F, 0.875F
    };
    static const uint16_t embedding[] = {
        0x3fa0U, 0x3f00U, 0x3f40U, 0xbf00U
    };
    static const uint16_t hidden[] = {
        0x3fceU, 0x3f97U, 0x3efcU, 0xbe1fU,
        0x3fe6U, 0x3e3eU, 0x3efaU, 0xbe94U
    };
    static const float step_eos[] = {1.0234375F, 1.609375F};
    static const float step_semantic[] = {
        -0.1640625F, 1.515625F, 2.515625F,
        -0.83984375F, 0.220703125F, 3.1875F
    };
    static const uint16_t key_cache[] = {
        0x3e21U, 0x3fdcU, 0x3f01U, 0xbf9fU,
        0xbf8cU, 0xbc81U, 0xbfbfU, 0x3f13U,
        0x3eaaU, 0x3f60U, 0x3eaaU, 0x3f60U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U
    };
    static const uint16_t value_cache[] = {
        0x3fe1U, 0x3d60U, 0xbfcdU, 0x3ee4U,
        0xbec8U, 0x3fcdU, 0x4006U, 0xbe97U,
        0xbe93U, 0x3f14U, 0xbe93U, 0x3f14U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U
    };
    m3_qwen_test_fixture fixture;
    m3_tensor_view ids;
    m3_qwen_forward_result result;
    m3_qwen_progress_log progress;
    m3_error error;
    size_t index;
    bool progress_exact = true;
    bool prefill_ok;
    bool step_ok;

    if (!m3_qwen_test_fixture_init(&fixture, 4U)) {
        M3_TEST_EXPECT(test, false, "initialize reusable Qwen kernel fixture");
        return;
    }
    (void)memset(&result, 0, sizeof(result));
    (void)memset(&progress, 0, sizeof(progress));
    progress.state = &fixture.forward;
    progress.cancel_at = UINT64_MAX;
    prefill_ok = m3_qwen_test_ids(&fixture, &ids, 2U, prompt) &&
                 m3_qwen_forward_execute(
                     &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids,
                     m3_qwen_test_progress, &progress, &result, &error) ==
                     M3_STATUS_OK;
    M3_TEST_EXPECT(test, prefill_ok,
                   "execute production prefill plan on two prompt rows");
    if (!prefill_ok) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(test,
                   fixture.forward.token_count == 2U &&
                       result.token_embedding == NULL &&
                       result.hidden == NULL &&
                       result.eos_logits->metadata.shape[0] == 2U &&
                       result.eos_logits->metadata.shape[1] == 1U &&
                       result.semantic_logits->metadata.shape[0] == 2U &&
                       result.semantic_logits->metadata.shape[1] == 3U,
                   "publish only next logits after prefill");
    M3_TEST_EXPECT(test,
                   m3_qwen_f32_values(result.eos_logits, prefill_eos, 2U) &&
                       m3_qwen_f32_values(result.semantic_logits,
                                          prefill_semantic, 6U),
                   "lock conditional/unconditional prefill row order and head IDs");
    for (index = 0U; index < progress.call_count; ++index) {
        progress_exact = progress_exact &&
                         progress.completed[index] == (uint64_t)index &&
                         progress.totals[index] == 4U &&
                         progress.observed_counts[index] == 0U;
    }
    M3_TEST_EXPECT(test, progress.call_count == 5U && progress_exact,
                   "checkpoint prefill without early token publication");
    (void)memset(&progress, 0, sizeof(progress));
    progress.state = &fixture.forward;
    progress.cancel_at = UINT64_MAX;
    step_ok = m3_qwen_test_ids(&fixture, &ids, 1U, step_ids) &&
              m3_qwen_forward_execute(
                  &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                  m3_qwen_test_progress, &progress, &result, &error) ==
                  M3_STATUS_OK;
    M3_TEST_EXPECT(test, step_ok,
                   "execute one-token production step at published position two");
    if (!step_ok) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(test,
                   fixture.forward.token_count == 3U &&
                       result.token_embedding->metadata.rank == 1U &&
                       result.hidden->metadata.shape[0] == 2U &&
                       result.hidden->metadata.shape[1] == 4U,
                   "publish step embedding, hidden rows, and following state");
    M3_TEST_EXPECT(test,
                   m3_qwen_bf16_values(result.token_embedding, embedding,
                                        4U) &&
                       m3_qwen_bf16_values(result.hidden, hidden, 8U),
                   "lock raw shared embedding and final normalized hidden rows");
    M3_TEST_EXPECT(test,
                   m3_qwen_test_bf16_at(
                       &fixture.cache.workspace.views[0], 8U) ==
                           m3_qwen_test_bf16_at(
                               &fixture.cache.workspace.views[0], 10U) &&
                       m3_qwen_test_bf16_at(
                           &fixture.cache.workspace.views[0], 9U) ==
                           m3_qwen_test_bf16_at(
                               &fixture.cache.workspace.views[0], 11U) &&
                       m3_qwen_test_bf16_at(
                           &fixture.cache.workspace.views[1], 8U) ==
                           m3_qwen_test_bf16_at(
                               &fixture.cache.workspace.views[1], 10U) &&
                       m3_qwen_test_bf16_at(
                           &fixture.cache.workspace.views[1], 9U) ==
                           m3_qwen_test_bf16_at(
                               &fixture.cache.workspace.views[1], 11U),
                   "feed and embed the same selected token for both step rows");
    M3_TEST_EXPECT(test,
                   m3_qwen_f32_values(result.eos_logits, step_eos, 2U) &&
                       m3_qwen_f32_values(result.semantic_logits,
                                          step_semantic, 6U),
                   "lock BF16 projection then F32 sliced-head step logits");
    M3_TEST_EXPECT(test,
                   m3_qwen_cache_values(&fixture.cache.workspace.views[0],
                                        key_cache, 16U) &&
                       m3_qwen_cache_values(
                           &fixture.cache.workspace.views[1], value_cache,
                           16U),
                   "write prefill and step K/V in token-major physical order");
    progress_exact = progress.call_count == 5U;
    for (index = 0U; index < progress.call_count; ++index) {
        progress_exact = progress_exact &&
                         progress.completed[index] == (uint64_t)index &&
                         progress.totals[index] == 4U &&
                         progress.observed_counts[index] == 2U;
    }
    M3_TEST_EXPECT(test, progress_exact,
                   "use published count as step causal offset until commit");
    m3_qwen_test_fixture_dispose(&fixture);
}

void m3_test_qwen_forward_atomicity(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const int32_t first_step[] = {4, 4};
    static const int32_t second_step[] = {5, 5};
    m3_qwen_test_fixture fixture;
    m3_tensor_view ids;
    m3_qwen_forward_result result;
    m3_qwen_forward_result unchanged;
    m3_qwen_progress_log progress;
    const m3_tensor_view *published_embedding;
    const m3_tensor_view *published_hidden;
    const m3_tensor_view *published_logits;
    uint16_t embedding_first;
    float logits_first;
    m3_error error;
    size_t index;
    bool count_held = true;

    if (!m3_qwen_test_fixture_init(&fixture, 4U)) {
        M3_TEST_EXPECT(test, false, "initialize Qwen atomicity fixture");
        return;
    }
    (void)memset(&result, 0, sizeof(result));
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 2U, prompt) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids,
                           NULL, NULL, &result, &error) == M3_STATUS_OK &&
                       m3_qwen_test_ids(&fixture, &ids, 1U, first_step) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                           NULL, NULL, &result, &error) == M3_STATUS_OK,
                   "publish a baseline step result");
    published_embedding = result.token_embedding;
    published_hidden = result.hidden;
    published_logits = result.semantic_logits;
    embedding_first = m3_qwen_test_bf16_at(published_embedding, 0U);
    logits_first = m3_qwen_test_f32_at(published_logits, 0U);
    unchanged.token_embedding = (const m3_tensor_view *)(uintptr_t)1U;
    unchanged.hidden = (const m3_tensor_view *)(uintptr_t)2U;
    unchanged.eos_logits = (const m3_tensor_view *)(uintptr_t)3U;
    unchanged.semantic_logits = (const m3_tensor_view *)(uintptr_t)4U;
    (void)memset(&progress, 0, sizeof(progress));
    progress.state = &fixture.forward;
    progress.cancel_at = 2U;
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 1U, second_step) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                           m3_qwen_test_progress, &progress, &unchanged,
                           NULL) == M3_STATUS_CANCELLED,
                   "cancel a step after the reusable transformer layer");
    for (index = 0U; index < progress.call_count; ++index) {
        count_held = count_held && progress.observed_counts[index] == 3U;
    }
    M3_TEST_EXPECT(test,
                   fixture.forward.token_count == 3U && count_held &&
                       unchanged.token_embedding ==
                           (const m3_tensor_view *)(uintptr_t)1U &&
                       unchanged.hidden ==
                           (const m3_tensor_view *)(uintptr_t)2U &&
                       unchanged.eos_logits ==
                           (const m3_tensor_view *)(uintptr_t)3U &&
                       unchanged.semantic_logits ==
                           (const m3_tensor_view *)(uintptr_t)4U,
                   "leave count and caller result untouched on cancellation");
    M3_TEST_EXPECT(test,
                   published_embedding == result.token_embedding &&
                       published_hidden == result.hidden &&
                       published_logits == result.semantic_logits &&
                       m3_qwen_test_bf16_at(published_embedding, 0U) ==
                           embedding_first &&
                       m3_qwen_test_f32_at(published_logits, 0U) ==
                           logits_first,
                   "retain published embedding, hidden, and logits lifetime "
                   "on failure");
    M3_TEST_EXPECT(test,
                   m3_qwen_forward_execute(
                       &fixture.forward, M3_QWEN_FORWARD_STEP, &ids, NULL,
                       NULL, &result, &error) == M3_STATUS_OK &&
                       fixture.forward.token_count == 4U &&
                       m3_qwen_test_bf16_at(result.token_embedding, 0U) ==
                           0xbf00U,
                   "retry the unpublished cache position and commit atomically");
    m3_qwen_test_fixture_dispose(&fixture);
}

void m3_test_qwen_forward_validation(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const int32_t one_step[] = {4, 4};
    static const int32_t mismatched[] = {4, 5};
    static const int32_t outside_vocab[] = {0, 8};
    m3_qwen_test_fixture fixture;
    m3_tensor_view ids;
    m3_qwen_forward_result result;
    m3_qwen_forward_result unchanged;
    m3_error error;

    if (!m3_qwen_test_fixture_init(&fixture, 3U)) {
        M3_TEST_EXPECT(test, false, "initialize Qwen validation fixture");
        return;
    }
    (void)memset(&result, 0, sizeof(result));
    unchanged.token_embedding = (const m3_tensor_view *)(uintptr_t)1U;
    unchanged.hidden = (const m3_tensor_view *)(uintptr_t)2U;
    unchanged.eos_logits = (const m3_tensor_view *)(uintptr_t)3U;
    unchanged.semantic_logits = (const m3_tensor_view *)(uintptr_t)4U;
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 1U, one_step) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                           NULL, NULL, &unchanged, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       fixture.forward.token_count == 0U,
                   "reject step before prefill with null error");
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 1U, outside_vocab) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids,
                           NULL, NULL, &unchanged, NULL) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       fixture.forward.token_count == 0U,
                   "reject prompt IDs outside the exact vocabulary");
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 2U, prompt) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids,
                           NULL, NULL, &result, &error) == M3_STATUS_OK &&
                       fixture.forward.token_count == 2U,
                   "prefill validation fixture to its prompt length");
    M3_TEST_EXPECT(test,
                   m3_qwen_forward_execute(
                       &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids,
                       NULL, NULL, &unchanged, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       fixture.forward.token_count == 2U,
                   "reject a second prefill without mutating state");
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 1U, mismatched) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                           NULL, NULL, &unchanged, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       fixture.forward.token_count == 2U,
                   "reject different conditional/unconditional step tokens");
    M3_TEST_EXPECT(test,
                   m3_qwen_test_ids(&fixture, &ids, 1U, one_step) &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                           NULL, NULL, &result, &error) == M3_STATUS_OK &&
                       fixture.forward.token_count == 3U &&
                       m3_qwen_forward_execute(
                           &fixture.forward, M3_QWEN_FORWARD_STEP, &ids,
                           NULL, NULL, &unchanged, NULL) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       fixture.forward.token_count == 3U,
                   "enforce cache capacity without publishing overflow");
    M3_TEST_EXPECT(test,
                   unchanged.token_embedding ==
                           (const m3_tensor_view *)(uintptr_t)1U &&
                       unchanged.hidden ==
                           (const m3_tensor_view *)(uintptr_t)2U &&
                       unchanged.eos_logits ==
                           (const m3_tensor_view *)(uintptr_t)3U &&
                       unchanged.semantic_logits ==
                           (const m3_tensor_view *)(uintptr_t)4U,
                   "keep caller outputs unchanged across validation failures");
    m3_qwen_test_fixture_dispose(&fixture);
}
