/* SPDX-License-Identifier: GPL-2.0-only */

#include "semantic_runtime_test.h"

#include "m3_tokenizer.h"

#include <stdlib.h>
#include <string.h>

static bool m3_semantic_test_storage(
    m3_backend *backend, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, m3_storage **storage, m3_tensor_view *view)
{
    m3_tensor_metadata metadata;
    m3_error error;

    m3_error_reset(&error);
    m3_tensor_view_init(view);
    return m3_tensor_metadata_init(
               &metadata, dtype, rank, shape, &error) == M3_STATUS_OK &&
           m3_storage_allocate(backend, metadata.byte_count, 64U, storage,
                               &error) == M3_STATUS_OK &&
           m3_tensor_view_contiguous(view, *storage, dtype, rank, shape,
                                     0U, &error) == M3_STATUS_OK;
}

static bool m3_semantic_test_write(m3_storage *storage,
                                   const void *source, size_t bytes)
{
    m3_error error;

    m3_error_reset(&error);
    return m3_storage_write(storage, 0U, source, bytes, &error) ==
           M3_STATUS_OK;
}

static m3_status m3_semantic_test_fail(m3_error *error,
                                       const char *phase)
{
    return m3_error_set(error, M3_STATUS_INTERNAL,
                        "injected semantic %s failure", phase);
}

static m3_status m3_semantic_test_child_progress(
    m3_progress_callback progress, void *context, uint64_t total,
    m3_error *error)
{
    uint64_t completed;

    for (completed = 0U; completed <= total; ++completed) {
        if (progress != NULL && !progress(context, completed, total)) {
            return m3_error_set(error, M3_STATUS_CANCELLED,
                                "scripted semantic child was cancelled");
        }
    }
    return M3_STATUS_OK;
}

static bool m3_semantic_test_fill_state(
    m3_semantic_test_fixture *fixture)
{
    size_t semantic_count =
        2U * (size_t)M3_QWEN_SEMANTIC_TOKEN_COUNT;
    uint16_t *hidden = malloc(2U * M3_QWEN_HIDDEN_SIZE * sizeof(*hidden));
    float *logits = malloc(semantic_count * sizeof(*logits));
    float eos[2];
    uint16_t hidden_value =
        (uint16_t)(UINT16_C(0x3f80) + fixture->state_index);
    bool is_eos = fixture->eos_states[fixture->state_index];
    uint32_t code = fixture->semantic_codes[fixture->state_index];
    uint32_t conditional_code = fixture->swap_guidance_rows
                                    ? fixture->guidance_code_b
                                    : fixture->guidance_code_a;
    uint32_t unconditional_code = fixture->swap_guidance_rows
                                      ? fixture->guidance_code_a
                                      : fixture->guidance_code_b;
    size_t index;
    bool success;

    if (hidden == NULL || logits == NULL ||
        code >= M3_QWEN_SEMANTIC_TOKEN_COUNT ||
        (fixture->adversarial_guidance &&
         (conditional_code >= M3_QWEN_SEMANTIC_TOKEN_COUNT ||
          unconditional_code >= M3_QWEN_SEMANTIC_TOKEN_COUNT))) {
        free(hidden);
        free(logits);
        return false;
    }
    for (index = 0U; index < 2U * M3_QWEN_HIDDEN_SIZE; ++index) {
        hidden[index] = hidden_value;
    }
    for (index = 0U; index < semantic_count; ++index) {
        logits[index] = -100.0F;
    }
    eos[0] = is_eos ? 100.0F : -100.0F;
    eos[1] = eos[0];
    if (!is_eos) {
        if (fixture->adversarial_guidance) {
            logits[conditional_code] = 100.0F;
            logits[M3_QWEN_SEMANTIC_TOKEN_COUNT +
                   unconditional_code] = 100.0F;
        } else {
            logits[code] = 100.0F;
            logits[M3_QWEN_SEMANTIC_TOKEN_COUNT + code] = 100.0F;
        }
    }
    success = m3_semantic_test_write(
                  fixture->hidden_storage, hidden,
                  2U * M3_QWEN_HIDDEN_SIZE * sizeof(*hidden)) &&
              m3_semantic_test_write(
                  fixture->semantic_storage, logits,
                  semantic_count * sizeof(*logits)) &&
              m3_semantic_test_write(
                  fixture->eos_storage, eos, sizeof(eos));
    free(hidden);
    free(logits);
    return success;
}

static m3_status m3_semantic_test_start(void *context,
                                        uint64_t cache_capacity,
                                        m3_error *error)
{
    m3_semantic_test_fixture *fixture = context;

    fixture->start_calls += 1U;
    fixture->cache_capacity = cache_capacity;
    if (fixture->failure == M3_SEMANTIC_TEST_FAIL_START) {
        return m3_semantic_test_fail(error, "start");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static void m3_semantic_test_finish(void *context)
{
    m3_semantic_test_fixture *fixture = context;

    fixture->finish_calls += 1U;
}

static void m3_semantic_test_publish_state(
    m3_semantic_test_fixture *fixture, m3_qwen_state *state)
{
    state->hidden = &fixture->hidden;
    state->eos_logits = &fixture->eos;
    state->semantic_logits = &fixture->semantic;
}

static m3_status m3_semantic_test_prefill(
    void *context, const m3_tensor_view *prompt_ids,
    m3_progress_callback progress, void *progress_context,
    m3_qwen_state *state, m3_error *error)
{
    m3_semantic_test_fixture *fixture = context;
    m3_status status;

    (void)prompt_ids;
    fixture->prefill_calls += 1U;
    if (fixture->failure == M3_SEMANTIC_TEST_FAIL_PREFILL) {
        return m3_semantic_test_fail(error, "prefill");
    }
    status = m3_semantic_test_child_progress(
        progress, progress_context, M3_SEMANTIC_QWEN_PROGRESS, error);
    if (status == M3_STATUS_OK && !m3_semantic_test_fill_state(fixture)) {
        status = m3_semantic_test_fail(error, "state fill");
    }
    if (status == M3_STATUS_OK) {
        m3_semantic_test_publish_state(fixture, state);
        m3_error_reset(error);
    }
    return status;
}

static m3_status m3_semantic_test_embedding(
    void *context, uint32_t code, m3_tensor_view *embedding,
    m3_error *error)
{
    m3_semantic_test_fixture *fixture = context;
    uint16_t values[M3_QWEN_HIDDEN_SIZE];
    size_t index;

    if (fixture->embedding_calls < M3_SEMANTIC_TEST_STATE_CAPACITY) {
        fixture->embedded_codes[fixture->embedding_calls] = code;
    } else {
        fixture->contract_valid = false;
    }
    fixture->embedding_calls += 1U;
    if (fixture->failure == M3_SEMANTIC_TEST_FAIL_EMBEDDING) {
        return m3_semantic_test_fail(error, "embedding");
    }
    for (index = 0U; index < M3_QWEN_HIDDEN_SIZE; ++index) {
        values[index] = (uint16_t)(UINT16_C(0x4100) + code);
    }
    if (!m3_semantic_test_write(
            fixture->embedding_storage, values, sizeof(values))) {
        return m3_semantic_test_fail(error, "embedding write");
    }
    *embedding = fixture->embedding;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_semantic_test_decode(
    void *context, const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, m3_progress_callback progress,
    void *progress_context, m3_rvq_frame *frame, m3_error *error)
{
    m3_semantic_test_fixture *fixture = context;
    const uint64_t shape[] = {
        1U, M3_RVQ_CODEBOOK_COUNT, M3_QWEN_HIDDEN_SIZE
    };
    uint16_t *values;
    size_t call = fixture->decode_calls;
    size_t slot;
    size_t index;
    m3_status status;

    if (last_hidden != &fixture->hidden ||
        semantic_embedding->storage != fixture->embedding_storage) {
        fixture->contract_valid = false;
    }
    fixture->decode_calls += 1U;
    if (fixture->failure == M3_SEMANTIC_TEST_FAIL_DECODE) {
        return m3_semantic_test_fail(error, "decode");
    }
    if (uniform_count != M3_RVQ_RESIDUAL_COUNT ||
        fixture->residual_uniform_count + uniform_count >
            M3_SEMANTIC_TEST_UNIFORM_CAPACITY) {
        return m3_semantic_test_fail(error, "uniform contract");
    }
    for (index = 0U; index < uniform_count; ++index) {
        fixture->residual_uniforms[fixture->residual_uniform_count++] =
            uniforms[index];
    }
    status = m3_semantic_test_child_progress(
        progress, progress_context, M3_SEMANTIC_RVQ_PROGRESS, error);
    values = malloc(
        M3_RVQ_CODEBOOK_COUNT * M3_QWEN_HIDDEN_SIZE * sizeof(*values));
    if (status != M3_STATUS_OK || values == NULL) {
        free(values);
        return status == M3_STATUS_OK
                   ? m3_semantic_test_fail(error, "frame allocation")
                   : status;
    }
    for (slot = 0U; slot < M3_RVQ_CODEBOOK_COUNT; ++slot) {
        uint16_t value = slot == 0U
                             ? (uint16_t)(UINT16_C(0x3f80) + call)
                             : (uint16_t)(UINT16_C(0x4200) + call * 16U +
                                          slot);
        for (index = 0U; index < M3_QWEN_HIDDEN_SIZE; ++index) {
            values[slot * M3_QWEN_HIDDEN_SIZE + index] = value;
        }
    }
    status = m3_storage_allocate(
        fixture->backend,
        M3_RVQ_CODEBOOK_COUNT * M3_QWEN_HIDDEN_SIZE * sizeof(*values),
        64U, &frame->storage, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &frame->conditioning, frame->storage, M3_DTYPE_BF16, 3U,
            shape, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_write(
            frame->storage, 0U, values,
            M3_RVQ_CODEBOOK_COUNT * M3_QWEN_HIDDEN_SIZE *
                sizeof(*values),
            error);
    }
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
        frame->codes[index] = (uint32_t)(call * 10U + index);
    }
    free(values);
    return status;
}

static m3_status m3_semantic_test_feedback(
    void *context, const m3_tensor_view *semantic_embedding,
    const m3_rvq_frame *frame, m3_rvq_feedback *feedback,
    m3_error *error)
{
    m3_semantic_test_fixture *fixture = context;
    const uint64_t shape[] = {2U, 1U, M3_QWEN_HIDDEN_SIZE};
    uint16_t values[2U * M3_QWEN_HIDDEN_SIZE];
    size_t index;
    m3_status status;

    if (semantic_embedding->storage != fixture->embedding_storage ||
        frame->storage == NULL ||
        frame->conditioning.storage != frame->storage) {
        fixture->contract_valid = false;
    }
    fixture->feedback_calls += 1U;
    if (fixture->failure == M3_SEMANTIC_TEST_FAIL_FEEDBACK) {
        return m3_semantic_test_fail(error, "feedback");
    }
    for (index = 0U; index < 2U * M3_QWEN_HIDDEN_SIZE; ++index) {
        values[index] = UINT16_C(0x3eb5);
    }
    status = m3_storage_allocate(
        fixture->backend, sizeof(values), 64U, &feedback->storage, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &feedback->tensor, feedback->storage, M3_DTYPE_BF16, 3U,
            shape, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_write(
            feedback->storage, 0U, values, sizeof(values), error);
    }
    return status;
}

static m3_status m3_semantic_test_advance(
    void *context, const m3_tensor_view *feedback,
    m3_progress_callback progress, void *progress_context,
    m3_qwen_state *state, m3_error *error)
{
    m3_semantic_test_fixture *fixture = context;
    m3_status status;

    if (feedback == NULL || feedback->metadata.dtype != M3_DTYPE_BF16 ||
        feedback->metadata.rank != 3U ||
        feedback->metadata.shape[0] != 2U ||
        feedback->metadata.shape[1] != 1U ||
        feedback->metadata.shape[2] != M3_QWEN_HIDDEN_SIZE) {
        fixture->contract_valid = false;
    }
    fixture->advance_calls += 1U;
    if (fixture->failure == M3_SEMANTIC_TEST_FAIL_ADVANCE) {
        return m3_semantic_test_fail(error, "advance");
    }
    status = m3_semantic_test_child_progress(
        progress, progress_context, M3_SEMANTIC_QWEN_PROGRESS, error);
    if (status == M3_STATUS_OK) {
        fixture->state_index += 1U;
        if (fixture->state_index >= fixture->state_count ||
            !m3_semantic_test_fill_state(fixture)) {
            status = m3_semantic_test_fail(error, "advance state");
        }
    }
    if (status == M3_STATUS_OK) {
        m3_semantic_test_publish_state(fixture, state);
        m3_error_reset(error);
    }
    return status;
}

bool m3_semantic_test_fixture_init(m3_semantic_test_fixture *fixture,
                                   m3_backend *backend)
{
    const uint64_t prompt_shape[] = {2U, 5U};
    const uint64_t hidden_shape[] = {2U, M3_QWEN_HIDDEN_SIZE};
    const uint64_t eos_shape[] = {2U, 1U};
    const uint64_t semantic_shape[] = {
        2U, M3_QWEN_SEMANTIC_TOKEN_COUNT
    };
    const uint64_t embedding_shape[] = {M3_QWEN_HIDDEN_SIZE};
    const int32_t prompt[] = {
        M3_TOKEN_IM_START, M3_TOKEN_CAPTION_START, M3_TOKEN_CAPTION_END,
        M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START,
        M3_TOKEN_IM_START, M3_TOKEN_AUDIO_CFG, M3_TOKEN_AUDIO_CFG,
        M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START
    };

    if (fixture == NULL || backend == NULL) {
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    fixture->backend = backend;
    if (!m3_semantic_test_storage(
            backend, M3_DTYPE_I32, 2U, prompt_shape,
            &fixture->prompt_storage, &fixture->prompt) ||
        !m3_semantic_test_storage(
            backend, M3_DTYPE_BF16, 2U, hidden_shape,
            &fixture->hidden_storage, &fixture->hidden) ||
        !m3_semantic_test_storage(
            backend, M3_DTYPE_F32, 2U, eos_shape,
            &fixture->eos_storage, &fixture->eos) ||
        !m3_semantic_test_storage(
            backend, M3_DTYPE_F32, 2U, semantic_shape,
            &fixture->semantic_storage, &fixture->semantic) ||
        !m3_semantic_test_storage(
            backend, M3_DTYPE_BF16, 1U, embedding_shape,
            &fixture->embedding_storage, &fixture->embedding) ||
        !m3_semantic_test_write(
            fixture->prompt_storage, prompt, sizeof(prompt))) {
        m3_semantic_test_fixture_dispose(fixture);
        return false;
    }
    fixture->operations = (m3_semantic_operations){
        fixture,
        backend,
        m3_semantic_test_start,
        m3_semantic_test_finish,
        m3_semantic_test_prefill,
        m3_semantic_test_embedding,
        m3_semantic_test_decode,
        m3_semantic_test_feedback,
        m3_semantic_test_advance
    };
    fixture->contract_valid = true;
    m3_semantic_test_outcomes(fixture, 3U, SIZE_MAX);
    return true;
}

void m3_semantic_test_fixture_dispose(
    m3_semantic_test_fixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    m3_storage_free(fixture->embedding_storage);
    m3_storage_free(fixture->semantic_storage);
    m3_storage_free(fixture->eos_storage);
    m3_storage_free(fixture->hidden_storage);
    m3_storage_free(fixture->prompt_storage);
    (void)memset(fixture, 0, sizeof(*fixture));
}

void m3_semantic_test_outcomes(m3_semantic_test_fixture *fixture,
                               size_t state_count, size_t eos_index)
{
    size_t index;

    fixture->state_count = state_count;
    fixture->state_index = 0U;
    for (index = 0U; index < M3_SEMANTIC_TEST_STATE_CAPACITY; ++index) {
        fixture->eos_states[index] = index == eos_index;
        fixture->semantic_codes[index] = (uint32_t)(index + 3U);
    }
}

void m3_semantic_test_guidance_rows(
    m3_semantic_test_fixture *fixture, uint32_t code_a,
    uint32_t code_b, bool swapped)
{
    fixture->guidance_code_a = code_a;
    fixture->guidance_code_b = code_b;
    fixture->adversarial_guidance = true;
    fixture->swap_guidance_rows = swapped;
}

bool m3_semantic_test_progress_callback(void *context,
                                        uint64_t completed,
                                        uint64_t total)
{
    m3_semantic_test_progress *progress = context;

    if (progress->count == 0U) {
        progress->total = total;
        progress->total_stable = true;
    } else if (progress->total != total) {
        progress->total_stable = false;
    }
    if (progress->count < M3_SEMANTIC_TEST_PROGRESS_CAPACITY) {
        progress->values[progress->count++] = completed;
    }
    return completed < progress->cancel_at;
}

bool m3_semantic_test_expected_rng(const m3_rng *initial,
                                   size_t uniform_count,
                                   m3_rng *expected)
{
    m3_error error;
    float unused;
    size_t index;

    *expected = *initial;
    m3_error_reset(&error);
    for (index = 0U; index < uniform_count; ++index) {
        if (m3_rng_uniform_f32(expected, &unused, &error) !=
            M3_STATUS_OK) {
            return false;
        }
    }
    return true;
}

bool m3_semantic_test_output_bits(const m3_semantic_output *output,
                                  uint16_t *bits, size_t count)
{
    m3_error error;

    m3_error_reset(&error);
    return output != NULL && output->storage != NULL && bits != NULL &&
           count <= m3_storage_size(output->storage) / sizeof(*bits) &&
           m3_storage_read(output->storage, 0U, bits,
                           count * sizeof(*bits), &error) == M3_STATUS_OK;
}

bool m3_semantic_test_seed_output(m3_backend *backend,
                                  uint16_t value,
                                  m3_semantic_output *output)
{
    const uint64_t shape[] = {
        1U, 1U, M3_RVQ_CODEBOOK_COUNT, M3_QWEN_HIDDEN_SIZE
    };
    size_t count = M3_RVQ_CODEBOOK_COUNT * M3_QWEN_HIDDEN_SIZE;
    uint16_t *values = malloc(count * sizeof(*values));
    size_t index;
    m3_error error;
    bool success;

    if (values == NULL) {
        return false;
    }
    for (index = 0U; index < count; ++index) {
        values[index] = value;
    }
    m3_error_reset(&error);
    m3_semantic_output_init(output);
    success = m3_storage_allocate(
                  backend, count * sizeof(*values), 64U,
                  &output->storage, &error) == M3_STATUS_OK &&
              m3_tensor_view_contiguous(
                  &output->frame_hiddens, output->storage,
                  M3_DTYPE_BF16, 4U, shape, 0U, &error) == M3_STATUS_OK &&
              m3_storage_write(
                  output->storage, 0U, values,
                  count * sizeof(*values), &error) == M3_STATUS_OK;
    free(values);
    if (!success) {
        m3_semantic_output_dispose(output);
    }
    return success;
}
