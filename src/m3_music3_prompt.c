/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include <limits.h>
#include <stdlib.h>

void m3_music3_prepared_prompt_init(m3_music3_prepared_prompt *prompt)
{
    if (prompt != NULL) {
        prompt->storage = NULL;
        m3_tensor_view_init(&prompt->ids);
        prompt->semantic_plan = (m3_semantic_plan){0};
    }
}

void m3_music3_prepared_prompt_dispose(m3_music3_prepared_prompt *prompt)
{
    if (prompt != NULL) {
        m3_storage_free(prompt->storage);
        m3_music3_prepared_prompt_init(prompt);
    }
}

static m3_status m3_music3_prompt_rows(
    const m3_token_ids *conditional, const m3_token_ids *cfg,
    int32_t **rows, size_t *byte_count, m3_error *error)
{
    size_t count;
    int32_t *built;
    size_t index;

    *rows = NULL;
    *byte_count = 0U;
    if (conditional->count != cfg->count || conditional->count < 3U ||
        conditional->count > M3_TOKENIZER_MAX_PROMPT_IDS ||
        conditional->count > SIZE_MAX / (2U * sizeof(*built))) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Music3 prompt token count is invalid");
    }
    count = conditional->count;
    built = malloc(2U * count * sizeof(*built));
    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Music3 prompt rows");
    }
    for (index = 0U; index < count; ++index) {
        if (conditional->data[index] > INT32_MAX ||
            cfg->data[index] > INT32_MAX) {
            free(built);
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "Music3 prompt token does not fit I32");
        }
        built[index] = (int32_t)conditional->data[index];
        built[count + index] = (int32_t)cfg->data[index];
    }
    *rows = built;
    *byte_count = 2U * count * sizeof(*built);
    return M3_STATUS_OK;
}

static m3_status m3_music3_prompt_upload(
    m3_music3_engine *engine, const int32_t *rows, size_t byte_count,
    size_t token_count, m3_music3_prepared_prompt *prompt,
    m3_error *error)
{
    uint64_t shape[2] = {2U, (uint64_t)token_count};
    m3_status status = m3_storage_allocate(
        engine->backend, byte_count, 64U, &prompt->storage, error);

    if (status == M3_STATUS_OK) {
        status = m3_storage_write(prompt->storage, 0U, rows, byte_count,
                                  error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &prompt->ids, prompt->storage, M3_DTYPE_I32, 2U, shape, 0U,
            error);
    }
    return status;
}

m3_status m3_music3_prepare_prompt(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_prepared_prompt *prompt, m3_error *error)
{
    m3_prompt_text text;
    m3_token_ids conditional;
    m3_token_ids cfg;
    m3_music3_prepared_prompt built;
    int32_t *rows = NULL;
    size_t row_bytes = 0U;
    m3_status status;

    if (engine == NULL || engine->backend == NULL || request == NULL ||
        prompt == NULL || request->maximum_frames == 0U ||
        request->maximum_frames > M3_SEMANTIC_MAX_FRAMES) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Music3 request is invalid");
    }
    m3_prompt_text_init(&text);
    m3_token_ids_init(&conditional);
    m3_token_ids_init(&cfg);
    m3_music3_prepared_prompt_init(&built);
    status = m3_prompt_build(
        &text, request->caption, request->lyrics, error);
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_encode(
            &engine->tokenizer, m3_prompt_tokenizer_bytes(&text),
            &conditional, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_build_cfg_row(
            conditional.data, conditional.count, &cfg, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_prompt_rows(
            &conditional, &cfg, &rows, &row_bytes, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_plan_build(
            (uint64_t)conditional.count, request->maximum_frames,
            &built.semantic_plan, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_prompt_upload(
            engine, rows, row_bytes, conditional.count, &built, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_preflight_semantic(
            engine, &built.semantic_plan, error);
    }
    free(rows);
    m3_token_ids_dispose(&cfg);
    m3_token_ids_dispose(&conditional);
    m3_prompt_text_dispose(&text);
    if (status != M3_STATUS_OK) {
        m3_music3_prepared_prompt_dispose(&built);
        return status;
    }
    m3_music3_prepared_prompt_dispose(prompt);
    *prompt = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
