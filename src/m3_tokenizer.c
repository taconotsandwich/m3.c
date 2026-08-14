/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

#include "m3_file.h"

#include <stdlib.h>
#include <string.h>

static const m3_tokenizer_contract m3_official_tokenizer_contract = {
    M3_TOKENIZER_MODEL_VOCAB_COUNT,
    M3_TOKENIZER_MERGE_COUNT,
};

void m3_tokenizer_init(m3_tokenizer *tokenizer)
{
    if (tokenizer != NULL) {
        tokenizer->state = NULL;
    }
}

void m3_tokenizer_state_dispose(m3_tokenizer_state *state)
{
    if (state != NULL) {
        free(state->model_tokens);
        free(state->model_bytes);
        free(state->added_bytes);
        free(state->merge_slots);
        free(state);
    }
}

void m3_tokenizer_dispose(m3_tokenizer *tokenizer)
{
    if (tokenizer != NULL) {
        m3_tokenizer_state_dispose(tokenizer->state);
        tokenizer->state = NULL;
    }
}

void m3_token_ids_init(m3_token_ids *ids)
{
    if (ids != NULL) {
        ids->data = NULL;
        ids->count = 0U;
    }
}

void m3_token_ids_dispose(m3_token_ids *ids)
{
    if (ids != NULL) {
        free(ids->data);
        m3_token_ids_init(ids);
    }
}

void m3_tokenizer_text_init(m3_tokenizer_text *text)
{
    if (text != NULL) {
        text->data = NULL;
        text->length = 0U;
    }
}

void m3_tokenizer_text_dispose(m3_tokenizer_text *text)
{
    if (text != NULL) {
        free(text->data);
        m3_tokenizer_text_init(text);
    }
}

m3_status m3_tokenizer_load_data(m3_tokenizer *tokenizer,
                                  const uint8_t *data, size_t size,
                                  const m3_tokenizer_contract *contract,
                                  m3_error *error)
{
    m3_tokenizer_state *built = NULL;
    m3_status status;

    if (tokenizer == NULL || contract == NULL || data == NULL || size == 0U ||
        contract->model_vocab_count < 256U ||
        contract->merge_count != contract->model_vocab_count - 256U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer load contract is invalid");
    }
    status = m3_tokenizer_state_build(data, size, contract, &built, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_tokenizer_state_dispose(tokenizer->state);
    tokenizer->state = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_load(m3_tokenizer *tokenizer, const char *path,
                            m3_error *error)
{
    uint8_t *data = NULL;
    size_t size = 0U;
    m3_status status;

    if (tokenizer == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer is null");
    }
    status = m3_file_read_bounded(path, M3_TOKENIZER_JSON_MAX_BYTES,
                                  &data, &size, error);
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_load_data(tokenizer, data, size,
                                        &m3_official_tokenizer_contract,
                                        error);
    }
    free(data);
    return status;
}

m3_status m3_tokenizer_build_cfg_row(const uint32_t *prompt_ids,
                                     size_t prompt_count,
                                     m3_token_ids *cfg_ids,
                                     m3_error *error)
{
    uint32_t *built;
    size_t index;

    if (cfg_ids == NULL || prompt_ids == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "CFG row argument is null");
    }
    if (prompt_count < 3U || prompt_count > M3_TOKENIZER_MAX_PROMPT_IDS ||
        prompt_count > SIZE_MAX / sizeof(*built)) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "CFG prompt count %zu is invalid", prompt_count);
    }
    if (prompt_ids[0] != M3_TOKEN_IM_START ||
        prompt_ids[prompt_count - 2U] != M3_TOKEN_IM_END ||
        prompt_ids[prompt_count - 1U] != M3_TOKEN_AUDIO_START) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "CFG row does not have Music3 boundary tokens");
    }
    for (index = 0U; index < prompt_count; ++index) {
        if (prompt_ids[index] >= M3_TOKENIZER_ID_COUNT) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "CFG prompt ID %u is not a tokenizer ID",
                                prompt_ids[index]);
        }
    }
    built = malloc(prompt_count * sizeof(*built));
    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate CFG token row");
    }
    built[0] = prompt_ids[0];
    for (index = 1U; index + 2U < prompt_count; ++index) {
        built[index] = M3_TOKEN_AUDIO_CFG;
    }
    built[prompt_count - 2U] = prompt_ids[prompt_count - 2U];
    built[prompt_count - 1U] = prompt_ids[prompt_count - 1U];
    m3_token_ids_dispose(cfg_ids);
    cfg_ids->data = built;
    cfg_ids->count = prompt_count;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
