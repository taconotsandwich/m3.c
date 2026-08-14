/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_semantic_internal.h"

#include "m3_runtime_workspace.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    m3_progress_callback callback;
    void *context;
    uint64_t total;
    uint64_t reported;
    bool has_reported;
    bool cancelled;
    bool invalid_child;
} m3_semantic_progress;

typedef struct {
    m3_semantic_progress *progress;
    uint64_t base;
    uint64_t expected_total;
} m3_semantic_progress_bridge;

typedef struct {
    const m3_weight_stage *language_model;
    m3_backend *backend;
    m3_rvq_weights rvq_weights;
    m3_qwen_runtime *qwen;
} m3_semantic_production;

static m3_status m3_semantic_report(m3_semantic_progress *progress,
                                    uint64_t completed,
                                    m3_error *error)
{
    if (completed > progress->total) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "semantic progress exceeds its total");
    }
    if (progress->has_reported && completed <= progress->reported) {
        return M3_STATUS_OK;
    }
    progress->has_reported = true;
    progress->reported = completed;
    if (progress->callback != NULL &&
        !progress->callback(progress->context, completed,
                            progress->total)) {
        progress->cancelled = true;
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "semantic generation was cancelled");
    }
    return M3_STATUS_OK;
}

static bool m3_semantic_child_progress(void *context, uint64_t completed,
                                       uint64_t total)
{
    m3_semantic_progress_bridge *bridge = context;
    m3_semantic_progress *progress = bridge->progress;
    uint64_t mapped;

    if (total != bridge->expected_total || completed > total ||
        completed > UINT64_MAX - bridge->base) {
        progress->invalid_child = true;
        return false;
    }
    mapped = bridge->base + completed;
    if (mapped > progress->total) {
        progress->invalid_child = true;
        return false;
    }
    if (progress->has_reported && mapped <= progress->reported) {
        return !progress->cancelled;
    }
    progress->has_reported = true;
    progress->reported = mapped;
    if (progress->callback != NULL &&
        !progress->callback(progress->context, mapped, progress->total)) {
        progress->cancelled = true;
        return false;
    }
    return true;
}

static m3_status m3_semantic_child_status(
    m3_semantic_progress *progress, m3_status status, m3_error *error)
{
    if (progress->invalid_child) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "semantic child progress contract is invalid");
    }
    if (progress->cancelled && status == M3_STATUS_OK) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "semantic generation was cancelled");
    }
    return status;
}

static m3_status m3_semantic_check_view(
    m3_backend *backend, const m3_tensor_view *view, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, const char *name,
    m3_error *error)
{
    const void *unused = NULL;
    uint8_t axis;
    m3_status status;

    if (backend == NULL || view == NULL || view->storage == NULL ||
        m3_storage_backend(view->storage) != backend) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s is not on the semantic backend", name);
    }
    status = m3_tensor_const_data(view, &unused, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (view->metadata.dtype != dtype || view->metadata.rank != rank) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s has the wrong dtype or rank", name);
    }
    for (axis = 0U; axis < rank; ++axis) {
        if (view->metadata.shape[axis] != shape[axis]) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "%s has the wrong shape", name);
        }
    }
    (void)unused;
    return M3_STATUS_OK;
}

static m3_status m3_semantic_check_state(m3_backend *backend,
                                         const m3_qwen_state *state,
                                         m3_error *error)
{
    const uint64_t hidden[] = {2U, M3_QWEN_HIDDEN_SIZE};
    const uint64_t eos[] = {2U, 1U};
    const uint64_t semantic[] = {
        2U, M3_QWEN_SEMANTIC_TOKEN_COUNT
    };
    m3_status status;

    if (state == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "semantic Qwen state is null");
    }
    status = m3_semantic_check_view(
        backend, state->hidden, M3_DTYPE_BF16, 2U, hidden,
        "semantic Qwen hidden", error);
    if (status == M3_STATUS_OK) {
        status = m3_semantic_check_view(
            backend, state->eos_logits, M3_DTYPE_F32, 2U, eos,
            "semantic EOS logits", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_check_view(
            backend, state->semantic_logits, M3_DTYPE_F32, 2U,
            semantic, "semantic code logits", error);
    }
    return status;
}

static m3_status m3_semantic_check_frame(m3_backend *backend,
                                         const m3_rvq_frame *frame,
                                         m3_error *error)
{
    const uint64_t shape[] = {
        1U, M3_RVQ_CODEBOOK_COUNT, M3_QWEN_HIDDEN_SIZE
    };
    size_t index;
    m3_status status;

    if (frame == NULL || frame->storage == NULL ||
        frame->conditioning.storage != frame->storage) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "semantic RVQ frame is not owned");
    }
    status = m3_semantic_check_view(
        backend, &frame->conditioning, M3_DTYPE_BF16, 3U, shape,
        "semantic RVQ frame", error);
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT &&
                     status == M3_STATUS_OK; ++index) {
        if (frame->codes[index] >= M3_RVQ_CODEBOOK_SIZE) {
            status = m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "semantic RVQ code is outside the codebook");
        }
    }
    return status;
}

static m3_status m3_semantic_allocate_output(
    m3_backend *backend, const m3_semantic_plan *plan,
    m3_semantic_output *output, m3_error *error)
{
    const uint64_t shape[] = {
        1U, plan->frame_limit, M3_RVQ_CODEBOOK_COUNT,
        M3_QWEN_HIDDEN_SIZE
    };
    m3_status status = m3_storage_allocate(
        backend, plan->output_bytes, 64U, &output->storage, error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &output->frame_hiddens, output->storage, M3_DTYPE_BF16, 4U,
            shape, 0U, error);
    }
    return status;
}

static m3_status m3_semantic_copy_frame(
    m3_command_executor *executor, m3_semantic_output *output,
    uint64_t frame_index, const m3_rvq_frame *frame, m3_error *error)
{
    const uint64_t shape[] = {
        1U, M3_RVQ_CODEBOOK_COUNT, M3_QWEN_HIDDEN_SIZE
    };
    m3_tensor_view destination_slice;
    m3_tensor_view destination;
    m3_command command = {0};
    m3_status status;

    m3_tensor_view_init(&destination_slice);
    m3_tensor_view_init(&destination);
    status = m3_tensor_slice(&output->frame_hiddens, 1U, frame_index,
                             1U, &destination_slice, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(&destination_slice, 3U, shape,
                                   &destination, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    command.kind = M3_OP_COPY;
    command.descriptor.copy = (m3_op_unary){
        &frame->conditioning, &destination
    };
    return m3_command_executor_execute(executor, &command, 1U, error);
}

static m3_status m3_semantic_publish_shape(m3_semantic_output *output,
                                           uint64_t frames,
                                           m3_error *error)
{
    const uint64_t shape[] = {
        1U, frames, M3_RVQ_CODEBOOK_COUNT, M3_QWEN_HIDDEN_SIZE
    };
    m3_tensor_view published;
    m3_status status;

    m3_tensor_view_init(&published);
    status = m3_tensor_view_contiguous(
        &published, output->storage, M3_DTYPE_BF16, 4U, shape, 0U,
        error);
    if (status == M3_STATUS_OK) {
        output->frame_hiddens = published;
    }
    return status;
}

static m3_status m3_semantic_draw_residuals(
    m3_rng *rng, float uniforms[M3_RVQ_RESIDUAL_COUNT],
    m3_error *error)
{
    size_t index;
    m3_status status = M3_STATUS_OK;

    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT &&
                     status == M3_STATUS_OK; ++index) {
        status = m3_rng_uniform_f32(rng, &uniforms[index], error);
    }
    return status;
}

static m3_status m3_semantic_attempt(
    const m3_semantic_operations *operations,
    const m3_semantic_plan *plan, m3_qwen_state *state,
    m3_rng *rng, uint64_t attempt, uint64_t *emitted,
    m3_command_executor *executor, m3_semantic_output *output,
    m3_semantic_progress *progress, bool *finished,
    m3_error *error)
{
    const uint64_t embedding_shape[] = {M3_QWEN_HIDDEN_SIZE};
    uint64_t base = M3_SEMANTIC_QWEN_PROGRESS +
                    attempt * M3_SEMANTIC_FULL_ATTEMPT_PROGRESS;
    float residual_uniforms[M3_RVQ_RESIDUAL_COUNT];
    float semantic_uniform = 0.0F;
    m3_guided_semantic_sample sample = {0};
    m3_rvq_feedback feedback;
    m3_rvq_frame frame;
    m3_tensor_view embedding;
    m3_semantic_progress_bridge bridge;
    m3_status status;

    *finished = false;
    m3_rvq_feedback_init(&feedback);
    m3_rvq_frame_init(&frame);
    m3_tensor_view_init(&embedding);
    status = m3_rng_uniform_f32(rng, &semantic_uniform, error);
    if (status == M3_STATUS_OK) {
        status = m3_guided_sample_semantic(
            operations->backend, state->eos_logits,
            state->semantic_logits, semantic_uniform, &sample, error);
    }
    if (status == M3_STATUS_OK && !sample.eos) {
        status = operations->semantic_embedding(
            operations->context, sample.code, &embedding, error);
    }
    if (status == M3_STATUS_OK && !sample.eos) {
        status = m3_semantic_check_view(
            operations->backend, &embedding, M3_DTYPE_BF16, 1U,
            embedding_shape, "semantic code embedding", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_report(progress, base + 1U, error);
    }
    if (status == M3_STATUS_OK && sample.eos) {
        if (*emitted == 0U) {
            status = m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "MiniMax Music 3 generated zero semantic frames");
        } else {
            status = m3_semantic_report(
                progress, plan->progress_total, error);
            *finished = status == M3_STATUS_OK;
        }
    }
    if (status == M3_STATUS_OK && !sample.eos) {
        status = m3_semantic_draw_residuals(
            rng, residual_uniforms, error);
    }
    bridge = (m3_semantic_progress_bridge){
        progress, base + 1U, M3_SEMANTIC_RVQ_PROGRESS
    };
    if (status == M3_STATUS_OK && !sample.eos) {
        status = operations->decode_frame(
            operations->context, state->hidden, &embedding,
            residual_uniforms, M3_RVQ_RESIDUAL_COUNT,
            m3_semantic_child_progress, &bridge, &frame, error);
        status = m3_semantic_child_status(progress, status, error);
    }
    if (status == M3_STATUS_OK && !sample.eos) {
        status = m3_semantic_check_frame(
            operations->backend, &frame, error);
    }
    if (status == M3_STATUS_OK && !sample.eos && attempt != 0U) {
        status = m3_semantic_copy_frame(
            executor, output, *emitted, &frame, error);
        if (status == M3_STATUS_OK) {
            *emitted += 1U;
        }
    }
    if (status == M3_STATUS_OK && !sample.eos) {
        status = m3_semantic_report(progress, base + 9U, error);
    }
    if (status == M3_STATUS_OK && !sample.eos &&
        *emitted == plan->frame_limit) {
        *finished = true;
    }
    if (status == M3_STATUS_OK && !sample.eos && !*finished) {
        status = operations->build_feedback(
            operations->context, &embedding, &frame, &feedback, error);
    }
    m3_rvq_frame_dispose(&frame);
    if (status == M3_STATUS_OK && !sample.eos && !*finished) {
        status = m3_semantic_report(progress, base + 10U, error);
    }
    bridge = (m3_semantic_progress_bridge){
        progress, base + 10U, M3_SEMANTIC_QWEN_PROGRESS
    };
    if (status == M3_STATUS_OK && !sample.eos && !*finished) {
        status = operations->advance(
            operations->context, &feedback.tensor,
            m3_semantic_child_progress, &bridge, state, error);
        status = m3_semantic_child_status(progress, status, error);
    }
    m3_rvq_feedback_dispose(&feedback);
    if (status == M3_STATUS_OK && !sample.eos && !*finished) {
        status = m3_semantic_check_state(
            operations->backend, state, error);
    }
    return status;
}

static m3_status m3_semantic_schedule(
    const m3_semantic_operations *operations,
    const m3_tensor_view *prompt_ids, const m3_semantic_plan *plan,
    m3_rng *rng, m3_semantic_progress *progress,
    m3_semantic_output *output, m3_error *error)
{
    m3_semantic_progress_bridge bridge = {
        progress, 0U, M3_SEMANTIC_QWEN_PROGRESS
    };
    m3_command_executor executor;
    m3_qwen_state state = {0};
    uint64_t emitted = 0U;
    uint64_t attempt;
    bool finished = false;
    m3_status status;

    m3_command_executor_init(&executor, operations->backend);
    status = operations->prefill(
        operations->context, prompt_ids, m3_semantic_child_progress,
        &bridge, &state, error);
    status = m3_semantic_child_status(progress, status, error);
    if (status == M3_STATUS_OK) {
        status = m3_semantic_check_state(
            operations->backend, &state, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_allocate_output(
            operations->backend, plan, output, error);
    }
    for (attempt = 0U; attempt <= plan->frame_limit &&
                       status == M3_STATUS_OK && !finished; ++attempt) {
        status = m3_semantic_attempt(
            operations, plan, &state, rng, attempt, &emitted, &executor,
            output, progress, &finished, error);
    }
    if (status == M3_STATUS_OK && !finished) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "semantic schedule did not terminate");
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_publish_shape(output, emitted, error);
    }
    m3_command_executor_dispose(&executor);
    return status;
}

m3_status m3_semantic_generate_core(
    const m3_semantic_operations *operations,
    const m3_tensor_view *prompt_ids, uint64_t frame_limit, m3_rng *rng,
    m3_progress_callback progress_callback, void *progress_context,
    m3_semantic_output *output, m3_error *error)
{
    m3_semantic_output built;
    m3_semantic_progress progress;
    m3_semantic_plan plan;
    m3_rng local_rng;
    m3_status status;

    if (operations == NULL || operations->backend == NULL ||
        operations->start == NULL || operations->finish == NULL ||
        operations->prefill == NULL ||
        operations->semantic_embedding == NULL ||
        operations->decode_frame == NULL ||
        operations->build_feedback == NULL ||
        operations->advance == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "semantic operation driver is incomplete");
    }
    status = m3_semantic_validate_request(
        operations->backend, prompt_ids, frame_limit, rng, output, &plan,
        error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    local_rng = *rng;
    m3_semantic_output_init(&built);
    progress = (m3_semantic_progress){
        progress_callback, progress_context, plan.progress_total,
        0U, false, false, false
    };
    status = m3_semantic_report(&progress, 0U, error);
    if (status == M3_STATUS_OK) {
        status = operations->start(
            operations->context, plan.cache_capacity, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_semantic_schedule(
            operations, prompt_ids, &plan, &local_rng, &progress, &built,
            error);
    }
    operations->finish(operations->context);
    if (status != M3_STATUS_OK) {
        m3_semantic_output_dispose(&built);
        return status;
    }
    m3_semantic_output_dispose(output);
    *output = built;
    *rng = local_rng;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_semantic_production_start(void *context,
                                              uint64_t cache_capacity,
                                              m3_error *error)
{
    m3_semantic_production *production = context;

    return m3_qwen_runtime_create(
        &production->qwen, production->language_model, cache_capacity,
        error);
}

static void m3_semantic_production_finish(void *context)
{
    m3_semantic_production *production = context;

    m3_qwen_runtime_free(production->qwen);
    production->qwen = NULL;
}

static m3_status m3_semantic_production_prefill(
    void *context, const m3_tensor_view *prompt_ids,
    m3_progress_callback progress, void *progress_context,
    m3_qwen_state *state, m3_error *error)
{
    m3_semantic_production *production = context;

    return m3_qwen_runtime_prefill(
        production->qwen, prompt_ids, progress, progress_context, state,
        error);
}

static m3_status m3_semantic_production_embedding(
    void *context, uint32_t code, m3_tensor_view *embedding,
    m3_error *error)
{
    m3_semantic_production *production = context;

    return m3_qwen_runtime_semantic_embedding(
        production->qwen, code, embedding, error);
}

static m3_status m3_semantic_production_decode(
    void *context, const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, m3_progress_callback progress,
    void *progress_context, m3_rvq_frame *frame, m3_error *error)
{
    m3_semantic_production *production = context;

    return m3_rvq_decode_frame(
        production->backend, &production->rvq_weights, last_hidden,
        semantic_embedding, uniforms, uniform_count, progress,
        progress_context, frame, error);
}

static m3_status m3_semantic_production_feedback(
    void *context, const m3_tensor_view *semantic_embedding,
    const m3_rvq_frame *frame, m3_rvq_feedback *feedback,
    m3_error *error)
{
    m3_semantic_production *production = context;

    return m3_rvq_feedback_build(
        production->backend, &production->rvq_weights,
        semantic_embedding, frame, feedback, error);
}

static m3_status m3_semantic_production_advance(
    void *context, const m3_tensor_view *feedback,
    m3_progress_callback progress, void *progress_context,
    m3_qwen_state *state, m3_error *error)
{
    m3_semantic_production *production = context;

    return m3_qwen_runtime_advance(
        production->qwen, feedback, progress, progress_context, state,
        error);
}

m3_status m3_semantic_generate(
    const m3_weight_stage *language_model,
    const m3_weight_stage *rvq_depth_decoder,
    const m3_tensor_view *prompt_ids, uint64_t frame_limit, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_semantic_output *output, m3_error *error)
{
    m3_semantic_production production;
    m3_semantic_operations operations;
    m3_status status;

    if (language_model == NULL || rvq_depth_decoder == NULL ||
        language_model->backend == NULL ||
        rvq_depth_decoder->backend != language_model->backend) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "semantic language and RVQ stages require one backend");
    }
    (void)memset(&production, 0, sizeof(production));
    production.language_model = language_model;
    production.backend = language_model->backend;
    status = m3_rvq_weights_bind(
        rvq_depth_decoder, &production.rvq_weights, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    operations = (m3_semantic_operations){
        &production,
        production.backend,
        m3_semantic_production_start,
        m3_semantic_production_finish,
        m3_semantic_production_prefill,
        m3_semantic_production_embedding,
        m3_semantic_production_decode,
        m3_semantic_production_feedback,
        m3_semantic_production_advance
    };
    return m3_semantic_generate_core(
        &operations, prompt_ids, frame_limit, rng, progress,
        progress_context, output, error);
}
