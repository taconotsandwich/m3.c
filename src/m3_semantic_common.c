/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_semantic_internal.h"

#include "m3_tokenizer.h"

#include <string.h>

void m3_semantic_output_init(m3_semantic_output *output)
{
    if (output != NULL) {
        (void)memset(output, 0, sizeof(*output));
    }
}

void m3_semantic_output_dispose(m3_semantic_output *output)
{
    if (output == NULL) {
        return;
    }
    m3_storage_free(output->storage);
    m3_semantic_output_init(output);
}

static bool m3_semantic_output_valid(const m3_semantic_output *output)
{
    const m3_tensor_view *view;

    if (output == NULL) {
        return false;
    }
    view = &output->frame_hiddens;
    if (output->storage == NULL) {
        return view->storage == NULL;
    }
    return view->storage == output->storage &&
           view->metadata.dtype == M3_DTYPE_BF16 &&
           view->metadata.rank == 4U &&
           view->metadata.shape[0] == 1U &&
           view->metadata.shape[1] != 0U &&
           view->metadata.shape[1] <= M3_SEMANTIC_MAX_FRAMES &&
           view->metadata.shape[2] == M3_RVQ_CODEBOOK_COUNT &&
           view->metadata.shape[3] == M3_QWEN_HIDDEN_SIZE;
}

static m3_status m3_semantic_prompt_contract(
    const m3_tensor_view *prompt_ids, m3_error *error)
{
    const int32_t *ids;
    const void *raw = NULL;
    size_t tokens = (size_t)prompt_ids->metadata.shape[1];
    size_t index;
    m3_status status = m3_tensor_const_data(prompt_ids, &raw, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (raw == NULL) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "semantic prompt IDs are not host-readable");
    }
    ids = raw;
    if (ids[0] != (int32_t)M3_TOKEN_IM_START ||
        ids[tokens - 2U] != (int32_t)M3_TOKEN_IM_END ||
        ids[tokens - 1U] != (int32_t)M3_TOKEN_AUDIO_START) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "semantic prompt has invalid boundary tokens");
    }
    for (index = 0U; index < tokens; ++index) {
        if (ids[index] < 0 ||
            (uint32_t)ids[index] >= M3_TOKENIZER_ID_COUNT) {
            return m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "semantic conditional prompt ID is outside the tokenizer");
        }
    }
    if (ids[tokens] != ids[0] ||
        ids[tokens * 2U - 2U] != ids[tokens - 2U] ||
        ids[tokens * 2U - 1U] != ids[tokens - 1U]) {
        return m3_error_set(
            error, M3_STATUS_INVALID_FORMAT,
            "semantic unconditional prompt boundaries do not match");
    }
    for (index = 1U; index + 2U < tokens; ++index) {
        if (ids[tokens + index] != (int32_t)M3_TOKEN_AUDIO_CFG) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "semantic unconditional prompt is not the CFG row");
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_semantic_validate_request(
    m3_backend *backend, const m3_tensor_view *prompt_ids,
    uint64_t frame_limit, const m3_rng *rng,
    const m3_semantic_output *output, m3_semantic_plan *plan,
    m3_error *error)
{
    uint64_t tokens;
    m3_status status;

    if (backend == NULL || prompt_ids == NULL || rng == NULL ||
        plan == NULL || !m3_semantic_output_valid(output) ||
        (rng->increment & UINT64_C(1)) == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "semantic generation arguments are invalid");
    }
    if (prompt_ids->storage == NULL ||
        m3_storage_backend(prompt_ids->storage) != backend ||
        prompt_ids->metadata.dtype != M3_DTYPE_I32 ||
        prompt_ids->metadata.rank != 2U ||
        prompt_ids->metadata.shape[0] != 2U ||
        !m3_tensor_is_contiguous(prompt_ids)) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "semantic prompt must be contiguous I32 [2,T] on the backend");
    }
    tokens = prompt_ids->metadata.shape[1];
    if (tokens < 3U || tokens > M3_TOKENIZER_MAX_PROMPT_IDS ||
        frame_limit == 0U || frame_limit > M3_SEMANTIC_MAX_FRAMES) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "semantic prompt or frame limit is out of range");
    }
    status = m3_semantic_prompt_contract(prompt_ids, error);
    if (status == M3_STATUS_OK) {
        status = m3_semantic_plan_build(tokens, frame_limit, plan, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_preflight(backend, plan, error);
    }
    return status;
}
