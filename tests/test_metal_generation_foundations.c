/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_qwen_internal.h"
#include "m3_rvq_condition_internal.h"
#include "qwen_runtime_fixture.h"
#include "rvq_condition_test.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static bool m3_generation_feedback_exact(
    const m3_rvq_feedback *feedback)
{
    static const uint16_t expected[] = {
        0x411cU, 0xc08dU, 0x3e35U, 0x3eb5U,
        0x411cU, 0xc08dU, 0x3e35U, 0x3eb5U
    };
    const uint8_t *data;

    if (feedback == NULL || feedback->storage == NULL ||
        feedback->tensor.storage != feedback->storage ||
        feedback->tensor.metadata.dtype != M3_DTYPE_BF16 ||
        feedback->tensor.metadata.rank != 3U ||
        feedback->tensor.metadata.shape[0] != 2U ||
        feedback->tensor.metadata.shape[1] != 1U ||
        feedback->tensor.metadata.shape[2] != 4U) {
        return false;
    }
    data = m3_storage_const_data(feedback->storage);
    return data != NULL &&
           memcmp(data + feedback->tensor.byte_offset, expected,
                  sizeof(expected)) == 0;
}

static void m3_generation_state_set(m3_qwen_state *state,
                                    const m3_qwen_forward_result *result)
{
    state->hidden = result->hidden;
    state->eos_logits = result->eos_logits;
    state->semantic_logits = result->semantic_logits;
}

void m3_test_metal_generation_foundations(m3_test_context *test)
{
    static const int32_t prompt[] = {0, 1, 2, 3};
    static const uint16_t semantic_bits[] = {
        0xbf00U, 0x3fc0U, 0x3f00U, 0x3f80U
    };
    m3_qwen_test_fixture qwen;
    m3_rvq_test_fixture rvq;
    m3_qwen_runtime lookup;
    m3_qwen_forward_result result = {0};
    m3_qwen_state state = {0};
    m3_rvq_frame frame;
    m3_rvq_feedback feedback;
    m3_tensor_view prompt_ids;
    m3_tensor_view semantic;
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    m3_backend *backend = NULL;
    float uniforms[M3_RVQ_RESIDUAL_COUNT];
    uint32_t expected_codes[M3_RVQ_RESIDUAL_COUNT];
    m3_error error;
    m3_status status;
    bool ready;

    (void)memset(&qwen, 0, sizeof(qwen));
    (void)memset(&rvq, 0, sizeof(rvq));
    (void)memset(&lookup, 0, sizeof(lookup));
    m3_rvq_frame_init(&frame);
    m3_rvq_feedback_init(&feedback);
    m3_tensor_view_init(&prompt_ids);
    m3_tensor_view_init(&semantic);

    status = m3_backend_create_metal(&backend, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "require a real Metal generation backend");
    if (status != M3_STATUS_OK) {
        m3_backend_free(backend);
        return;
    }
    ready = m3_qwen_test_fixture_init_backend(
        &qwen, 3U, backend, false);
    M3_TEST_EXPECT(test, ready,
                   "build reduced Qwen state on the Metal backend");
    if (!ready) {
        goto cleanup;
    }
    ready = m3_rvq_test_fixture_init_backend(
        &rvq, M3_DTYPE_BF16,
        (uint32_t)qwen.dimensions.hidden_size,
        qwen.tensors.backend, false);
    M3_TEST_EXPECT(test, ready,
                   "attach reduced RVQ weights to the same backend");
    if (!ready) {
        goto cleanup;
    }
    ready = m3_backend_get_info(qwen.tensors.backend, &info, &error) ==
                M3_STATUS_OK &&
            info.kind == M3_BACKEND_METAL &&
            rvq.fixture.backend == qwen.tensors.backend;
    M3_TEST_EXPECT(test, ready,
                   "use one Metal backend without a Host executor");
    if (!ready) {
        goto cleanup;
    }

    ready = m3_qwen_test_ids(&qwen, &prompt_ids, 2U, prompt) &&
            m3_qwen_forward_execute(
                &qwen.forward, M3_QWEN_FORWARD_PREFILL, &prompt_ids,
                NULL, NULL, &result, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "run production Qwen prefill commands on Metal");
    if (!ready) {
        goto cleanup;
    }
    m3_generation_state_set(&state, &result);
    M3_TEST_EXPECT(
        test,
        qwen.forward.token_count == 2U && state.hidden != NULL &&
            state.hidden->storage != NULL &&
            m3_storage_backend(state.hidden->storage) ==
                qwen.tensors.backend,
        "publish conditional and unconditional Qwen state on Metal");

    lookup.dimensions = qwen.dimensions;
    lookup.weights = qwen.weights;
    ready = m3_qwen_runtime_semantic_embedding(
                &lookup, 1U, &semantic, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ready && semantic.storage == qwen.weights.embedding->storage &&
            semantic.metadata.dtype == M3_DTYPE_BF16 &&
            semantic.metadata.rank == 1U &&
            semantic.metadata.shape[0] == qwen.dimensions.hidden_size &&
            memcmp((const uint8_t *)m3_storage_const_data(
                       semantic.storage) + semantic.byte_offset,
                   semantic_bits, sizeof(semantic_bits)) == 0,
        "map one semantic code through the production Qwen table view");
    if (!ready) {
        goto cleanup;
    }

    m3_rvq_test_uniforms(uniforms, expected_codes);
    ready = m3_rvq_decode_frame_core(
                rvq.fixture.backend, &rvq.config, &rvq.weights,
                state.hidden, &semantic, uniforms,
                M3_RVQ_RESIDUAL_COUNT, NULL, NULL, &frame,
                &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ready && frame.storage != NULL &&
            m3_storage_backend(frame.storage) == qwen.tensors.backend &&
            memcmp(frame.codes, expected_codes,
                   sizeof(expected_codes)) == 0,
        "decode all seven RVQ residual codes through Metal commands");
    if (!ready) {
        goto cleanup;
    }

    ready = m3_rvq_feedback_build_core(
                rvq.fixture.backend, &rvq.config, &rvq.weights,
                &semantic, &frame, &feedback, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ready &&
            m3_storage_backend(feedback.storage) == qwen.tensors.backend &&
            m3_generation_feedback_exact(&feedback),
        "execute the complete five-command RVQ feedback list on Metal");
    if (!ready) {
        goto cleanup;
    }

    ready = m3_qwen_forward_execute(
                &qwen.forward, M3_QWEN_FORWARD_ADVANCE,
                &feedback.tensor, NULL, NULL, &result,
                &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "advance Qwen with complete repeated audio feedback");
    if (!ready) {
        goto cleanup;
    }
    m3_generation_state_set(&state, &result);
    M3_TEST_EXPECT(
        test,
        qwen.forward.token_count == 3U && state.hidden != NULL &&
            state.eos_logits != NULL && state.semantic_logits != NULL &&
            m3_storage_backend(state.hidden->storage) ==
                qwen.tensors.backend,
        "publish exactly one complete-frame Qwen advance on Metal");

cleanup:
    m3_rvq_feedback_dispose(&feedback);
    m3_rvq_frame_dispose(&frame);
    m3_rvq_test_fixture_dispose(&rvq);
    m3_qwen_test_fixture_dispose(&qwen);
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_storage_count == 0U &&
            stats.live_allocated_bytes == 0U,
        "release every generation allocation before Metal teardown");
    m3_backend_free(backend);
}
