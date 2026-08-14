/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include <stdint.h>
#include <string.h>

typedef enum {
    M3_RVQ_FEEDBACK_IDS = 0,
    M3_RVQ_FEEDBACK_EMBEDDINGS,
    M3_RVQ_FEEDBACK_ONES,
    M3_RVQ_FEEDBACK_SUM,
    M3_RVQ_FEEDBACK_SCALE,
    M3_RVQ_FEEDBACK_WORKSPACE_COUNT
} m3_rvq_feedback_workspace_index;

static void m3_rvq_feedback_spec(m3_runtime_tensor_spec *spec,
                                 m3_dtype dtype, uint8_t rank,
                                 const uint64_t *shape)
{
    uint8_t axis;

    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = dtype;
    spec->rank = rank;
    spec->alignment = 64U;
    for (axis = 0U; axis < rank; ++axis) {
        spec->shape[axis] = shape[axis];
    }
}

static void m3_rvq_feedback_specs(
    uint32_t hidden_size, m3_runtime_tensor_spec *specs)
{
    const uint64_t ids[] = {M3_RVQ_RESIDUAL_COUNT};
    const uint64_t embeddings[] = {
        M3_RVQ_RESIDUAL_COUNT, hidden_size
    };
    const uint64_t ones[] = {1U, M3_RVQ_RESIDUAL_COUNT};
    const uint64_t vector[] = {1U, hidden_size};
    const uint64_t scale[] = {1U};

    m3_rvq_feedback_spec(&specs[M3_RVQ_FEEDBACK_IDS], M3_DTYPE_I32,
                         1U, ids);
    m3_rvq_feedback_spec(&specs[M3_RVQ_FEEDBACK_EMBEDDINGS],
                         M3_DTYPE_BF16, 2U, embeddings);
    m3_rvq_feedback_spec(&specs[M3_RVQ_FEEDBACK_ONES], M3_DTYPE_BF16,
                         2U, ones);
    m3_rvq_feedback_spec(&specs[M3_RVQ_FEEDBACK_SUM], M3_DTYPE_BF16,
                         2U, vector);
    m3_rvq_feedback_spec(&specs[M3_RVQ_FEEDBACK_SCALE], M3_DTYPE_BF16,
                         1U, scale);
}

static m3_status m3_rvq_feedback_validate(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights,
    const m3_tensor_view *semantic_embedding, const m3_rvq_frame *frame,
    const m3_rvq_feedback *feedback, size_t *output_bytes,
    m3_error *error)
{
    uint64_t audio_shape[2];
    uint64_t semantic_shape[1];
    uint64_t conditioning_shape[3];
    const uint64_t output_shape[] = {2U, 1U, config != NULL
                                                    ? config->hidden_size
                                                    : 0U};
    m3_tensor_metadata output_metadata;
    size_t index;
    m3_status status;

    if (backend == NULL || config == NULL || weights == NULL ||
        frame == NULL || feedback == NULL || output_bytes == NULL ||
        config->hidden_size == 0U || config->dtype != M3_DTYPE_BF16) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ feedback arguments are invalid");
    }
    audio_shape[0] = M3_RVQ_RESIDUAL_COUNT * M3_RVQ_CODEBOOK_SIZE;
    audio_shape[1] = config->hidden_size;
    semantic_shape[0] = config->hidden_size;
    conditioning_shape[0] = 1U;
    conditioning_shape[1] = M3_RVQ_CODEBOOK_COUNT;
    conditioning_shape[2] = config->hidden_size;
    status = m3_rvq_condition_check_view(
        backend, weights->audio_embeddings, M3_DTYPE_BF16, 2U,
        audio_shape, "RVQ feedback embeddings", error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_check_view(
            backend, semantic_embedding, M3_DTYPE_BF16, 1U,
            semantic_shape, "RVQ semantic embedding", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_check_view(
            backend, &frame->conditioning, M3_DTYPE_BF16, 3U,
            conditioning_shape, "RVQ decoded frame", error);
    }
    if (status == M3_STATUS_OK &&
        frame->conditioning.storage != frame->storage) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "RVQ decoded frame does not own its storage");
    }
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT &&
                     status == M3_STATUS_OK; ++index) {
        if (frame->codes[index] >= M3_RVQ_CODEBOOK_SIZE) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                  "RVQ feedback code is outside 0..1023");
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_metadata_init(
            &output_metadata, M3_DTYPE_BF16, 3U, output_shape, error);
    }
    if (status == M3_STATUS_OK) {
        *output_bytes = output_metadata.byte_count;
    }
    return status;
}

static m3_status m3_rvq_feedback_constants(
    m3_runtime_workspace *workspace, const m3_rvq_frame *frame,
    m3_error *error)
{
    static const uint16_t ones[M3_RVQ_RESIDUAL_COUNT] = {
        0x3f80U, 0x3f80U, 0x3f80U, 0x3f80U,
        0x3f80U, 0x3f80U, 0x3f80U
    };
    static const uint16_t inverse_sqrt_eight = 0x3eb5U;
    int32_t ids[M3_RVQ_RESIDUAL_COUNT];
    size_t index;
    m3_status status;

    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
        ids[index] = (int32_t)(index * M3_RVQ_CODEBOOK_SIZE +
                               frame->codes[index]);
    }
    status = m3_storage_write(
        workspace->storages[M3_RVQ_FEEDBACK_IDS], 0U, ids, sizeof(ids),
        error);
    if (status == M3_STATUS_OK) {
        status = m3_storage_write(
            workspace->storages[M3_RVQ_FEEDBACK_ONES], 0U, ones,
            sizeof(ones), error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_write(
            workspace->storages[M3_RVQ_FEEDBACK_SCALE], 0U,
            &inverse_sqrt_eight, sizeof(inverse_sqrt_eight), error);
    }
    return status;
}

static m3_status m3_rvq_feedback_execute(
    m3_backend *backend, const m3_rvq_weights *weights,
    const m3_tensor_view *semantic_embedding,
    m3_runtime_workspace *workspace, m3_rvq_feedback *feedback,
    m3_error *error)
{
    const uint64_t vector_shape[] = {
        1U, semantic_embedding->metadata.shape[0]
    };
    m3_tensor_view first_row;
    m3_tensor_view first_vector;
    m3_tensor_view second_row;
    m3_tensor_view second_vector;
    m3_command commands[5] = {0};
    m3_command_executor executor;
    m3_status status;

    m3_tensor_view_init(&first_row);
    m3_tensor_view_init(&first_vector);
    m3_tensor_view_init(&second_row);
    m3_tensor_view_init(&second_vector);
    status = m3_tensor_slice(
        &feedback->tensor, 0U, 0U, 1U, &first_row, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &first_row, 2U, vector_shape, &first_vector, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &feedback->tensor, 0U, 1U, 1U, &second_row, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &second_row, 2U, vector_shape, &second_vector, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_EMBEDDING;
    commands[0].descriptor.embedding = (m3_op_embedding){
        &workspace->views[M3_RVQ_FEEDBACK_IDS],
        weights->audio_embeddings,
        &workspace->views[M3_RVQ_FEEDBACK_EMBEDDINGS]
    };
    commands[1].kind = M3_OP_MATMUL;
    commands[1].descriptor.matmul = (m3_op_matmul){
        &workspace->views[M3_RVQ_FEEDBACK_ONES],
        &workspace->views[M3_RVQ_FEEDBACK_EMBEDDINGS],
        &workspace->views[M3_RVQ_FEEDBACK_SUM]
    };
    commands[2].kind = M3_OP_ADD;
    commands[2].descriptor.add = (m3_op_binary){
        semantic_embedding, &workspace->views[M3_RVQ_FEEDBACK_SUM],
        &first_vector
    };
    commands[3].kind = M3_OP_MUL;
    commands[3].descriptor.mul = (m3_op_binary){
        &first_vector, &workspace->views[M3_RVQ_FEEDBACK_SCALE],
        &first_vector
    };
    commands[4].kind = M3_OP_COPY;
    commands[4].descriptor.copy = (m3_op_unary){
        &first_vector, &second_vector
    };
    m3_command_executor_init(&executor, backend);
    status = m3_command_executor_execute(
        &executor, commands, sizeof(commands) / sizeof(commands[0]),
        error);
    m3_command_executor_dispose(&executor);
    return status;
}

m3_status m3_rvq_feedback_build_core(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights,
    const m3_tensor_view *semantic_embedding, const m3_rvq_frame *frame,
    m3_rvq_feedback *feedback, m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_RVQ_FEEDBACK_WORKSPACE_COUNT];
    m3_runtime_workspace workspace;
    m3_rvq_feedback built;
    uint64_t output_shape[3];
    size_t output_bytes = 0U;
    m3_status status;

    m3_runtime_workspace_init(&workspace);
    m3_rvq_feedback_init(&built);
    status = m3_rvq_feedback_validate(
        backend, config, weights, semantic_embedding, frame, feedback,
        &output_bytes, error);
    if (status == M3_STATUS_OK) {
        m3_rvq_feedback_specs(config->hidden_size, specs);
        status = m3_rvq_condition_preflight(
            backend, output_bytes, specs,
            M3_RVQ_FEEDBACK_WORKSPACE_COUNT, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_allocate(
            backend, output_bytes, 64U, &built.storage, error);
    }
    output_shape[0] = 2U;
    output_shape[1] = 1U;
    output_shape[2] = config != NULL ? config->hidden_size : 0U;
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &built.tensor, built.storage, M3_DTYPE_BF16, 3U,
            output_shape, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_build(
            &workspace, backend, specs,
            M3_RVQ_FEEDBACK_WORKSPACE_COUNT, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_feedback_constants(&workspace, frame, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_feedback_execute(
            backend, weights, semantic_embedding, &workspace, &built,
            error);
    }
    m3_runtime_workspace_dispose(&workspace);
    if (status != M3_STATUS_OK) {
        m3_rvq_feedback_dispose(&built);
        return status;
    }
    m3_rvq_feedback_dispose(feedback);
    *feedback = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_rvq_feedback_build(
    m3_backend *backend, const m3_rvq_weights *weights,
    const m3_tensor_view *semantic_embedding, const m3_rvq_frame *frame,
    m3_rvq_feedback *feedback, m3_error *error)
{
    static const m3_rvq_config config = {
        4096U, 4U, 16U, 6144U, 16U, M3_DTYPE_BF16
    };

    return m3_rvq_feedback_build_core(
        backend, &config, weights, semantic_embedding, frame, feedback,
        error);
}
