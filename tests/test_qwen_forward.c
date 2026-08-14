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

static bool m3_qwen_feedback_differs_from_tokens(
    const m3_qwen_test_fixture *fixture,
    const m3_tensor_view *feedback)
{
    size_t token;

    for (token = 0U; token < fixture->dimensions.vocab_size; ++token) {
        size_t channel;
        bool equal = true;

        for (channel = 0U; channel < fixture->dimensions.hidden_size;
             ++channel) {
            equal = equal &&
                m3_qwen_test_bf16_at(
                    fixture->weights.embedding,
                    token * fixture->dimensions.hidden_size + channel) ==
                m3_qwen_test_bf16_at(feedback, channel);
        }
        if (equal) {
            return false;
        }
    }
    return true;
}

static bool m3_qwen_strided_feedback(
    m3_qwen_test_fixture *fixture, m3_tensor_view *feedback)
{
    static const uint16_t values[] = {
        0x3ec0U, 0xbfa0U, 0x3f20U, 0x3fe0U
    };
    const uint64_t shape[] = {2U, 1U, 4U};
    const size_t strides[] = {32U, 24U, 4U};
    const size_t byte_offset = 6U;
    m3_storage *storage = NULL;
    uint8_t *data;
    m3_error error;
    size_t row;
    size_t channel;

    m3_tensor_view_init(feedback);
    if (!m3_op_test_storage(&fixture->tensors, 64U, &storage)) {
        return false;
    }
    data = m3_storage_data(storage);
    (void)memset(data, 0xcc, 64U);
    (void)memset(data + byte_offset, 0x11, strides[0]);
    (void)memset(data + byte_offset + strides[0], 0xa7,
                 64U - byte_offset - strides[0]);
    if (m3_tensor_view_strided(
            feedback, storage, M3_DTYPE_BF16, 3U, shape, strides,
            byte_offset, &error) != M3_STATUS_OK) {
        return false;
    }
    for (row = 0U; row < 2U; ++row) {
        for (channel = 0U; channel < 4U; ++channel) {
            size_t offset = byte_offset + row * strides[0] +
                            channel * strides[2];

            if (m3_storage_write(storage, offset, &values[channel],
                                 sizeof(values[channel]),
                                 &error) != M3_STATUS_OK) {
                return false;
            }
        }
    }
    return true;
}

static uint16_t m3_qwen_strided_value(
    const m3_tensor_view *feedback, size_t row, size_t channel)
{
    const uint8_t *data = m3_storage_const_data(feedback->storage);
    size_t offset = feedback->byte_offset +
                    row * feedback->byte_strides[0] +
                    channel * feedback->byte_strides[2];
    uint16_t value;

    (void)memcpy(&value, data + offset, sizeof(value));
    return value;
}

void m3_test_qwen_forward_oracle(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const float feedback_values[] = {
        0.375F, -1.25F, 0.625F, 1.75F,
        0.375F, -1.25F, 0.625F, 1.75F
    };
    static const uint16_t prefill_hidden[] = {
        0xbf06U, 0x3fe8U, 0x3f15U, 0x3f01U,
        0x3f6fU, 0x3f80U, 0xbf8eU, 0xbe8fU
    };
    static const float prefill_eos[] = {-0.90625F, -0.05322265625F};
    static const float prefill_semantic[] = {
        2.0F, 2.21875F, -1.453125F,
        -1.3828125F, 1.828125F, 0.875F
    };
    static const uint16_t advance_hidden[] = {
        0x3f93U, 0xbf88U, 0x3f3cU, 0x3f00U,
        0x3ee0U, 0xbf90U, 0x3f9aU, 0x3ea3U
    };
    static const float advance_eos[] = {2.234375F, 1.5390625F};
    static const float advance_semantic[] = {
        -0.53125F, -1.03125F, 1.859375F,
        0.392578125F, -1.65625F, 1.21875F
    };
    static const uint16_t key_cache[] = {
        0x3e21U, 0x3fdcU, 0x3f01U, 0xbf9fU,
        0xbf8cU, 0xbc81U, 0xbfbfU, 0x3f13U,
        0xbea3U, 0x3f22U, 0xbea3U, 0x3f22U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U
    };
    static const uint16_t value_cache[] = {
        0x3fe1U, 0x3d60U, 0xbfcdU, 0x3ee4U,
        0xbec8U, 0x3fcdU, 0x4006U, 0xbe97U,
        0x3e59U, 0xbec0U, 0x3e59U, 0xbec0U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U
    };
    m3_qwen_test_fixture fixture;
    m3_tensor_view input;
    m3_qwen_forward_result result;
    m3_qwen_progress_log progress;
    m3_error error;
    size_t index;
    bool progress_exact = true;
    bool ready;

    if (!m3_qwen_test_fixture_init(&fixture, 4U)) {
        M3_TEST_EXPECT(test, false, "initialize reusable Qwen kernel fixture");
        return;
    }
    (void)memset(&result, 0, sizeof(result));
    (void)memset(&progress, 0, sizeof(progress));
    progress.state = &fixture.forward;
    progress.cancel_at = UINT64_MAX;
    ready = m3_qwen_test_ids(&fixture, &input, 2U, prompt) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_PREFILL, &input,
                m3_qwen_test_progress, &progress, &result, &error) ==
                M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "execute production prefill plan on two prompt rows");
    if (!ready) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        fixture.forward.token_count == 2U &&
            result.hidden->metadata.shape[0] == 2U &&
            result.hidden->metadata.shape[1] == 4U &&
            result.eos_logits->metadata.shape[0] == 2U &&
            result.eos_logits->metadata.shape[1] == 1U &&
            result.semantic_logits->metadata.shape[0] == 2U &&
            result.semantic_logits->metadata.shape[1] == 3U,
        "publish prefill hidden rows and next-token logits");
    M3_TEST_EXPECT(
        test,
        m3_qwen_bf16_values(result.hidden, prefill_hidden, 8U) &&
            m3_qwen_f32_values(result.eos_logits, prefill_eos, 2U) &&
            m3_qwen_f32_values(result.semantic_logits, prefill_semantic,
                               6U),
        "lock prefill hidden and conditional/unconditional head order");
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
    ready = m3_qwen_test_feedback(&fixture, &input, feedback_values);
    M3_TEST_EXPECT(
        test,
        ready && m3_qwen_feedback_differs_from_tokens(&fixture, &input),
        "construct complete feedback distinct from every token row");
    ready = ready && m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &input,
                m3_qwen_test_progress, &progress, &result, &error) ==
                M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "advance with a repeated complete-frame embedding");
    if (!ready) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        fixture.forward.token_count == 3U &&
            result.hidden->metadata.shape[0] == 2U &&
            result.hidden->metadata.shape[1] == 4U,
        "publish one complete-frame advance state");
    M3_TEST_EXPECT(
        test,
        m3_qwen_bf16_values(result.hidden, advance_hidden, 8U) &&
            m3_qwen_f32_values(result.eos_logits, advance_eos, 2U) &&
            m3_qwen_f32_values(result.semantic_logits, advance_semantic,
                               6U),
        "lock complete-frame hidden and sliced-head logits");
    M3_TEST_EXPECT(
        test,
        m3_qwen_bf16_values(&fixture.cache.workspace.views[0], key_cache,
                             16U) &&
            m3_qwen_bf16_values(&fixture.cache.workspace.views[1],
                                 value_cache, 16U),
        "write prefill and advance K/V in token-major physical order");
    progress_exact = progress.call_count == 5U;
    for (index = 0U; index < progress.call_count; ++index) {
        progress_exact = progress_exact &&
                         progress.completed[index] == (uint64_t)index &&
                         progress.totals[index] == 4U &&
                         progress.observed_counts[index] == 2U;
    }
    M3_TEST_EXPECT(test, progress_exact,
                   "hold the published causal offset until advance commit");
    m3_qwen_test_fixture_dispose(&fixture);
}

void m3_test_qwen_forward_atomicity(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const float first_feedback[] = {
        0.375F, -1.25F, 0.625F, 1.75F,
        0.375F, -1.25F, 0.625F, 1.75F
    };
    static const float second_feedback[] = {
        -0.625F, 0.875F, -1.375F, 0.125F,
        -0.625F, 0.875F, -1.375F, 0.125F
    };
    m3_qwen_test_fixture fixture;
    m3_tensor_view input;
    m3_qwen_forward_result result;
    m3_qwen_forward_result unchanged;
    m3_qwen_progress_log progress;
    const m3_tensor_view *published_hidden;
    const m3_tensor_view *published_logits;
    uint16_t hidden_first;
    float logits_first;
    m3_error error;
    size_t index;
    bool count_held = true;

    if (!m3_qwen_test_fixture_init(&fixture, 4U)) {
        M3_TEST_EXPECT(test, false, "initialize Qwen atomicity fixture");
        return;
    }
    (void)memset(&result, 0, sizeof(result));
    M3_TEST_EXPECT(
        test,
        m3_qwen_test_ids(&fixture, &input, 2U, prompt) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_PREFILL, &input, NULL,
                NULL, &result, &error) == M3_STATUS_OK &&
            m3_qwen_test_feedback(&fixture, &input, first_feedback) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &input, NULL,
                NULL, &result, &error) == M3_STATUS_OK,
        "publish a baseline complete-frame state");
    published_hidden = result.hidden;
    published_logits = result.semantic_logits;
    hidden_first = m3_qwen_test_bf16_at(published_hidden, 0U);
    logits_first = m3_qwen_test_f32_at(published_logits, 0U);
    unchanged.hidden = (const m3_tensor_view *)(uintptr_t)1U;
    unchanged.eos_logits = (const m3_tensor_view *)(uintptr_t)2U;
    unchanged.semantic_logits = (const m3_tensor_view *)(uintptr_t)3U;
    (void)memset(&progress, 0, sizeof(progress));
    progress.state = &fixture.forward;
    progress.cancel_at = 2U;
    M3_TEST_EXPECT(
        test,
        m3_qwen_test_feedback(&fixture, &input, second_feedback) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &input,
                m3_qwen_test_progress, &progress, &unchanged, NULL) ==
                M3_STATUS_CANCELLED,
        "cancel an advance after the reusable transformer layer");
    for (index = 0U; index < progress.call_count; ++index) {
        count_held = count_held && progress.observed_counts[index] == 3U;
    }
    M3_TEST_EXPECT(
        test,
        fixture.forward.token_count == 3U && count_held &&
            unchanged.hidden == (const m3_tensor_view *)(uintptr_t)1U &&
            unchanged.eos_logits ==
                (const m3_tensor_view *)(uintptr_t)2U &&
            unchanged.semantic_logits ==
                (const m3_tensor_view *)(uintptr_t)3U,
        "leave count and caller state untouched on cancellation");
    M3_TEST_EXPECT(
        test,
        published_hidden == result.hidden &&
            published_logits == result.semantic_logits &&
            m3_qwen_test_bf16_at(published_hidden, 0U) == hidden_first &&
            m3_qwen_test_f32_at(published_logits, 0U) == logits_first,
        "retain the published state lifetime on failed advance");
    M3_TEST_EXPECT(
        test,
        m3_qwen_forward_execute(
            &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &input, NULL, NULL,
            &result, &error) == M3_STATUS_OK &&
            fixture.forward.token_count == 4U,
        "retry the unpublished cache position and commit atomically");
    m3_qwen_test_fixture_dispose(&fixture);
}

void m3_test_qwen_forward_validation(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const int32_t outside_vocab[] = {0, 8};
    static const float feedback_values[] = {
        0.375F, -1.25F, 0.625F, 1.75F,
        0.375F, -1.25F, 0.625F, 1.75F
    };
    static const float unequal_rows[] = {
        0.375F, -1.25F, 0.625F, 1.75F,
        0.375F, -1.25F, 0.625F, 1.625F
    };
    m3_qwen_test_fixture fixture;
    m3_tensor_view feedback;
    m3_tensor_view ids;
    m3_qwen_forward_result result;
    m3_qwen_forward_result unchanged;
    const m3_tensor_view *published_hidden;
    uint16_t published_first;
    m3_error error;
    bool ready;

    if (!m3_qwen_test_fixture_init(&fixture, 3U)) {
        M3_TEST_EXPECT(test, false, "initialize Qwen validation fixture");
        return;
    }
    (void)memset(&result, 0, sizeof(result));
    unchanged.hidden = (const m3_tensor_view *)(uintptr_t)1U;
    unchanged.eos_logits = (const m3_tensor_view *)(uintptr_t)2U;
    unchanged.semantic_logits = (const m3_tensor_view *)(uintptr_t)3U;
    M3_TEST_EXPECT(
        test,
        m3_qwen_test_feedback(&fixture, &feedback, feedback_values) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &feedback,
                NULL, NULL, &unchanged, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            fixture.forward.token_count == 0U,
        "reject complete-frame advance before prefill with null error");
    M3_TEST_EXPECT(
        test,
        m3_qwen_test_ids(&fixture, &ids, 1U, outside_vocab) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids, NULL,
                NULL, &unchanged, NULL) == M3_STATUS_OUT_OF_RANGE &&
            fixture.forward.token_count == 0U,
        "reject prompt IDs outside the exact vocabulary");
    ready = m3_qwen_test_ids(&fixture, &ids, 2U, prompt) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids, NULL,
                NULL, &result, &error) == M3_STATUS_OK &&
            fixture.forward.token_count == 2U;
    M3_TEST_EXPECT(test, ready,
                   "prefill validation fixture to its prompt length");
    if (!ready) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    published_hidden = result.hidden;
    published_first = m3_qwen_test_bf16_at(published_hidden, 0U);
    M3_TEST_EXPECT(
        test,
        m3_qwen_forward_execute(
            &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids, NULL, NULL,
            &unchanged, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &ids, NULL,
                NULL, &unchanged, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            fixture.forward.token_count == 2U,
        "reject second prefill and token IDs as feedback atomically");
    M3_TEST_EXPECT(
        test,
        m3_qwen_test_feedback(&fixture, &feedback, unequal_rows) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &feedback,
                NULL, NULL, &unchanged, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            fixture.forward.token_count == 2U &&
            result.hidden == published_hidden &&
            m3_qwen_test_bf16_at(result.hidden, 0U) == published_first &&
            m3_qwen_test_bf16_at(&fixture.cache.workspace.views[0], 8U) ==
                0U,
        "reject unequal rows before state publication or cache writes");
    M3_TEST_EXPECT(
        test,
        m3_qwen_test_feedback(&fixture, &feedback, feedback_values) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &feedback,
                NULL, NULL, &result, &error) == M3_STATUS_OK &&
            fixture.forward.token_count == 3U &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &feedback,
                NULL, NULL, &unchanged, NULL) == M3_STATUS_OUT_OF_RANGE &&
            fixture.forward.token_count == 3U,
        "enforce cache capacity without publishing overflow");
    M3_TEST_EXPECT(
        test,
        unchanged.hidden == (const m3_tensor_view *)(uintptr_t)1U &&
            unchanged.eos_logits ==
                (const m3_tensor_view *)(uintptr_t)2U &&
            unchanged.semantic_logits ==
                (const m3_tensor_view *)(uintptr_t)3U,
        "keep caller state unchanged across validation failures");
    m3_qwen_test_fixture_dispose(&fixture);
}

void m3_test_qwen_strided_feedback(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const uint16_t expected_hidden[] = {
        0x3f93U, 0xbf88U, 0x3f3cU, 0x3f00U,
        0x3ee0U, 0xbf90U, 0x3f9aU, 0x3ea3U
    };
    static const float expected_eos[] = {2.234375F, 1.5390625F};
    static const float expected_semantic[] = {
        -0.53125F, -1.03125F, 1.859375F,
        0.392578125F, -1.65625F, 1.21875F
    };
    m3_qwen_test_fixture fixture;
    m3_tensor_view ids;
    m3_tensor_view feedback;
    m3_qwen_forward_result state = {0};
    m3_qwen_forward_result unchanged;
    const m3_tensor_view *published_hidden;
    const m3_tensor_view *published_semantic;
    uint16_t key_cache[16];
    uint16_t value_cache[16];
    uint16_t hidden_first;
    float semantic_first;
    uint16_t changed = 0x3fd0U;
    const uint8_t *data;
    m3_error error;
    size_t index;
    bool cache_unchanged = true;
    bool logical_rows_equal = true;
    bool ready;

    if (!m3_qwen_test_fixture_init(&fixture, 4U)) {
        M3_TEST_EXPECT(test, false,
                       "initialize strided Qwen feedback fixture");
        return;
    }
    ready = m3_qwen_test_ids(&fixture, &ids, 2U, prompt) &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_PREFILL, &ids, NULL,
                NULL, &state, &error) == M3_STATUS_OK &&
            m3_qwen_strided_feedback(&fixture, &feedback);
    M3_TEST_EXPECT(test, ready,
                   "prefill and build padded strided BF16 feedback");
    if (!ready) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    for (index = 0U; index < 4U; ++index) {
        logical_rows_equal = logical_rows_equal &&
            m3_qwen_strided_value(&feedback, 0U, index) ==
                m3_qwen_strided_value(&feedback, 1U, index);
    }
    data = m3_storage_const_data(feedback.storage);
    M3_TEST_EXPECT(
        test,
        feedback.byte_offset == 6U && feedback.byte_strides[0] == 32U &&
            feedback.byte_strides[2] == 4U && logical_rows_equal &&
            data[feedback.byte_offset + 2U] !=
                data[feedback.byte_offset + feedback.byte_strides[0] +
                     2U],
        "keep equal logical rows with unequal row and channel padding");
    ready = m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &feedback,
                NULL, NULL, &state, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ready && fixture.forward.token_count == 3U &&
            m3_qwen_bf16_values(
                state.hidden, expected_hidden, 8U) &&
            m3_qwen_f32_values(state.eos_logits, expected_eos, 2U) &&
            m3_qwen_f32_values(
                state.semantic_logits, expected_semantic, 6U),
        "copy only strided logical BF16 values into Qwen advance");
    if (!ready) {
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    published_hidden = state.hidden;
    published_semantic = state.semantic_logits;
    hidden_first = m3_qwen_test_bf16_at(published_hidden, 0U);
    semantic_first = m3_qwen_test_f32_at(published_semantic, 0U);
    for (index = 0U; index < 16U; ++index) {
        key_cache[index] = m3_qwen_test_bf16_at(
            &fixture.cache.workspace.views[0], index);
        value_cache[index] = m3_qwen_test_bf16_at(
            &fixture.cache.workspace.views[1], index);
    }
    unchanged.hidden = (const m3_tensor_view *)(uintptr_t)1U;
    unchanged.eos_logits = (const m3_tensor_view *)(uintptr_t)2U;
    unchanged.semantic_logits =
        (const m3_tensor_view *)(uintptr_t)3U;
    M3_TEST_EXPECT(
        test,
        m3_storage_write(
            feedback.storage,
            feedback.byte_offset + feedback.byte_strides[0] +
                3U * feedback.byte_strides[2],
            &changed, sizeof(changed), &error) == M3_STATUS_OK &&
            m3_qwen_forward_execute(
                &fixture.forward, M3_QWEN_FORWARD_ADVANCE, &feedback,
                NULL, NULL, &unchanged, NULL) ==
                M3_STATUS_INVALID_ARGUMENT,
        "reject one changed logical BF16 in the second row");
    for (index = 0U; index < 16U; ++index) {
        cache_unchanged = cache_unchanged &&
            key_cache[index] == m3_qwen_test_bf16_at(
                &fixture.cache.workspace.views[0], index) &&
            value_cache[index] == m3_qwen_test_bf16_at(
                &fixture.cache.workspace.views[1], index);
    }
    M3_TEST_EXPECT(
        test,
        fixture.forward.token_count == 3U && cache_unchanged &&
            state.hidden == published_hidden &&
            state.semantic_logits == published_semantic &&
            m3_qwen_test_bf16_at(state.hidden, 0U) == hidden_first &&
            m3_qwen_test_f32_at(state.semantic_logits, 0U) ==
                semantic_first &&
            unchanged.hidden == (const m3_tensor_view *)(uintptr_t)1U &&
            unchanged.eos_logits ==
                (const m3_tensor_view *)(uintptr_t)2U &&
            unchanged.semantic_logits ==
                (const m3_tensor_view *)(uintptr_t)3U,
        "reject strided row mismatch before cache or state mutation");
    m3_qwen_test_fixture_dispose(&fixture);
}
