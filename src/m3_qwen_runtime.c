/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_qwen_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void m3_qwen_runtime_init(m3_qwen_runtime *runtime)
{
    (void)memset(runtime, 0, sizeof(*runtime));
    m3_qwen_cache_init(&runtime->cache);
    m3_runtime_workspace_init(&runtime->rope);
    m3_qwen_forward_state_init(&runtime->forward);
}

void m3_qwen_runtime_free(m3_qwen_runtime *runtime)
{
    if (runtime == NULL) {
        return;
    }
    m3_qwen_forward_state_dispose(&runtime->forward);
    m3_command_executor_dispose(&runtime->executor);
    m3_runtime_workspace_dispose(&runtime->rope);
    m3_qwen_cache_dispose(&runtime->cache);
    free(runtime);
}

static void m3_qwen_runtime_connect(m3_qwen_runtime *runtime)
{
    runtime->forward.dimensions = &runtime->dimensions;
    runtime->forward.weights = &runtime->weights;
    runtime->forward.backend = runtime->stage->backend;
    runtime->forward.executor = &runtime->executor;
    runtime->forward.cache_views = runtime->cache.workspace.views;
    runtime->forward.cosines = &runtime->rope.views[0];
    runtime->forward.sines = &runtime->rope.views[1];
    runtime->forward.cache_capacity = runtime->cache.capacity;
}

m3_status m3_qwen_runtime_create(m3_qwen_runtime **runtime,
                                 const m3_weight_stage *language_model,
                                 uint64_t cache_capacity,
                                 m3_error *error)
{
    m3_qwen_runtime *built;
    m3_status status;

    if (runtime == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen runtime output is null");
    }
    *runtime = NULL;
    if (language_model == NULL || language_model->backend == NULL ||
        cache_capacity == 0U ||
        cache_capacity > M3_QWEN_MAX_CAPACITY) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "staged Qwen model and cache capacity 1..14000 are required");
    }
    built = calloc(1U, sizeof(*built));
    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Qwen runtime state");
    }
    m3_qwen_runtime_init(built);
    built->stage = language_model;
    built->dimensions = *m3_qwen_official_dimensions();
    m3_command_executor_init(&built->executor, language_model->backend);
    status = m3_qwen_weights_bind(&built->weights, language_model, error);
    if (status == M3_STATUS_OK) {
        status = m3_qwen_cache_build(
            &built->cache, language_model->backend, &built->dimensions,
            cache_capacity, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_rope_build(
            &built->rope, language_model->backend, &built->dimensions,
            cache_capacity, error);
    }
    if (status != M3_STATUS_OK) {
        m3_qwen_runtime_free(built);
        return status;
    }
    m3_qwen_runtime_connect(built);
    *runtime = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_qwen_runtime_prefill(m3_qwen_runtime *runtime,
                                  const m3_tensor_view *prompt_ids,
                                  m3_progress_callback progress,
                                  void *progress_context,
                                  m3_qwen_logits *logits,
                                  m3_error *error)
{
    m3_qwen_forward_result forward_result;
    m3_status status;

    if (runtime == NULL || logits == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen runtime and prefill output are required");
    }
    (void)memset(&forward_result, 0, sizeof(forward_result));
    status = m3_qwen_forward_execute(
        &runtime->forward, M3_QWEN_FORWARD_PREFILL, prompt_ids, progress,
        progress_context, &forward_result, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    logits->eos_logits = forward_result.eos_logits;
    logits->semantic_logits = forward_result.semantic_logits;
    return M3_STATUS_OK;
}

static m3_status m3_qwen_step_ids(m3_qwen_runtime *runtime,
                                  uint32_t token_id,
                                  m3_runtime_workspace *ids,
                                  m3_error *error)
{
    m3_runtime_tensor_spec spec;
    uint64_t shape[] = {2U, 1U};
    int32_t values[2];
    m3_status status;

    (void)memset(&spec, 0, sizeof(spec));
    spec.dtype = M3_DTYPE_I32;
    spec.rank = 2U;
    spec.shape[0] = shape[0];
    spec.shape[1] = shape[1];
    spec.alignment = 64U;
    status = m3_runtime_workspace_build(
        ids, runtime->stage->backend, &spec, 1U, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    values[0] = (int32_t)token_id;
    values[1] = (int32_t)token_id;
    return m3_storage_write(ids->storages[0], 0U, values, sizeof(values),
                            error);
}

m3_status m3_qwen_runtime_step(m3_qwen_runtime *runtime,
                               uint32_t token_id,
                               m3_progress_callback progress,
                               void *progress_context,
                               m3_qwen_step_result *result,
                               m3_error *error)
{
    m3_runtime_workspace ids;
    m3_qwen_forward_result forward_result;
    uint64_t semantic_end = (uint64_t)M3_QWEN_SEMANTIC_TOKEN_START +
                            M3_QWEN_SEMANTIC_TOKEN_COUNT;
    m3_status status;

    if (runtime == NULL || result == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen runtime and step output are required");
    }
    if ((uint64_t)token_id < M3_QWEN_SEMANTIC_TOKEN_START ||
        (uint64_t)token_id >= semantic_end) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Qwen step token is not an official semantic token");
    }
    m3_runtime_workspace_init(&ids);
    (void)memset(&forward_result, 0, sizeof(forward_result));
    status = m3_qwen_step_ids(runtime, token_id, &ids, error);
    if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_execute(
            &runtime->forward, M3_QWEN_FORWARD_STEP, &ids.views[0],
            progress, progress_context, &forward_result, error);
    }
    m3_runtime_workspace_dispose(&ids);
    if (status != M3_STATUS_OK) {
        return status;
    }
    result->token_embedding = forward_result.token_embedding;
    result->hidden = forward_result.hidden;
    result->eos_logits = forward_result.eos_logits;
    result->semantic_logits = forward_result.semantic_logits;
    return M3_STATUS_OK;
}

uint64_t m3_qwen_runtime_token_count(const m3_qwen_runtime *runtime)
{
    return runtime == NULL ? 0U : runtime->forward.token_count;
}

uint64_t m3_qwen_runtime_cache_capacity(const m3_qwen_runtime *runtime)
{
    return runtime == NULL ? 0U : runtime->cache.capacity;
}

size_t m3_qwen_runtime_cache_bytes(const m3_qwen_runtime *runtime)
{
    return runtime == NULL ? 0U : runtime->cache.byte_count;
}
