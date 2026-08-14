/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include "m3_guided_sampling.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    m3_backend *backend;
    const m3_rvq_config *config;
    const m3_rvq_weights *weights;
    m3_runtime_workspace workspace;
    m3_command_executor executor;
    m3_rvq_frame built;
} m3_rvq_run;

typedef struct {
    m3_tensor_view sequence;
    m3_tensor_view position_ids;
    m3_tensor_view positions;
    m3_tensor_view hidden;
    m3_tensor_view hidden_temp;
    m3_tensor_view norm;
    m3_tensor_view query;
    m3_tensor_view key;
    m3_tensor_view value;
    m3_tensor_view attention;
    m3_tensor_view reorder;
    m3_tensor_view gate;
    m3_tensor_view up;
    m3_tensor_view logits;
} m3_rvq_step_views;

static m3_status m3_rvq_view(m3_rvq_run *run, size_t workspace_index,
                             m3_dtype dtype, uint8_t rank,
                             const uint64_t *shape, m3_tensor_view *view,
                             m3_error *error)
{
    m3_tensor_view_init(view);
    return m3_tensor_view_contiguous(
        view, run->workspace.storages[workspace_index], dtype, rank, shape,
        0U, error);
}

static m3_status m3_rvq_sequence_position(
    m3_rvq_run *run, size_t position, m3_tensor_view *view,
    m3_error *error)
{
    m3_tensor_view slice;
    const uint64_t shape[] = {2U, run->config->hidden_size};
    m3_status status;

    m3_tensor_view_init(&slice);
    m3_tensor_view_init(view);
    status = m3_tensor_slice(
        m3_runtime_workspace_view(&run->workspace, M3_RVQ_WS_SEQUENCE),
        0U, (uint64_t)position, 1U, &slice, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(&slice, 2U, shape, view, error);
    }
    return status;
}

static m3_status m3_rvq_row(const m3_tensor_view *source, size_t row,
                            m3_tensor_view *view, m3_error *error)
{
    uint64_t shape[1];
    size_t strides[1];
    size_t offset;

    m3_tensor_view_init(view);
    if (source == NULL || source->metadata.rank != 2U ||
        row >= source->metadata.shape[0] ||
        (source->byte_strides[0] != 0U &&
         row > (SIZE_MAX - source->byte_offset) /
                   source->byte_strides[0])) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ row view is invalid");
    }
    shape[0] = source->metadata.shape[1];
    strides[0] = source->byte_strides[1];
    offset = source->byte_offset + row * source->byte_strides[0];
    return m3_tensor_view_strided(
        view, source->storage, source->metadata.dtype, 1U, shape, strides,
        offset, error);
}

static m3_status m3_rvq_condition_vector(
    m3_rvq_run *run, size_t index, m3_tensor_view *view, m3_error *error)
{
    m3_tensor_view slice;
    const uint64_t shape[] = {run->config->hidden_size};
    m3_status status;

    m3_tensor_view_init(&slice);
    m3_tensor_view_init(view);
    status = m3_tensor_slice(&run->built.conditioning, 1U,
                             (uint64_t)index, 1U, &slice, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(&slice, 1U, shape, view, error);
    }
    return status;
}

static void m3_rvq_linear(m3_command *command,
                          const m3_tensor_view *input,
                          const m3_tensor_view *weight,
                          m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_LINEAR;
    command->descriptor.linear.input = input;
    command->descriptor.linear.weight = weight;
    command->descriptor.linear.output = output;
}

static void m3_rvq_copy(m3_command *command,
                        const m3_tensor_view *input,
                        m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_COPY;
    command->descriptor.copy.input = input;
    command->descriptor.copy.output = output;
}

static m3_status m3_rvq_setup(m3_rvq_run *run,
                              const m3_tensor_view *last_hidden,
                              const m3_tensor_view *semantic_embedding,
                              m3_error *error)
{
    const uint64_t semantic_shape[] = {run->config->hidden_size};
    m3_tensor_view condition_zero;
    m3_tensor_view last_zero;
    m3_tensor_view sequence_zero;
    m3_tensor_view sequence_one;
    m3_tensor_view sequence_one_zero;
    m3_tensor_view sequence_one_one;
    m3_tensor_view semantic_projected;
    m3_command commands[5];
    int32_t positions[M3_RVQ_CODEBOOK_COUNT];
    size_t index;
    m3_status status;

    for (index = 0U; index < M3_RVQ_CODEBOOK_COUNT; ++index) {
        positions[index] = (int32_t)index;
    }
    status = m3_storage_write(
        run->workspace.storages[M3_RVQ_WS_POSITION_IDS], 0U, positions,
        sizeof(positions), error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_row(last_hidden, 0U, &last_zero, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_vector(run, 0U, &condition_zero, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_sequence_position(run, 0U, &sequence_zero, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_sequence_position(run, 1U, &sequence_one, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_row(&sequence_one, 0U, &sequence_one_zero, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_row(&sequence_one, 1U, &sequence_one_one, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_view(
            run, M3_RVQ_WS_TOKEN, run->config->dtype, 1U,
            semantic_shape, &semantic_projected, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_rvq_copy(&commands[0], &last_zero, &condition_zero);
    m3_rvq_linear(&commands[1], last_hidden, run->weights->projection,
                  &sequence_zero);
    m3_rvq_linear(&commands[2], semantic_embedding,
                  run->weights->projection, &semantic_projected);
    m3_rvq_copy(&commands[3], &semantic_projected, &sequence_one_zero);
    m3_rvq_copy(&commands[4], &semantic_projected, &sequence_one_one);
    return m3_command_executor_execute(&run->executor, commands, 5U,
                                       error);
}

static m3_status m3_rvq_build_step_views(
    m3_rvq_run *run, size_t sequence_length, m3_rvq_step_views *views,
    m3_error *error)
{
    const uint32_t hidden = run->config->hidden_size;
    const uint32_t heads = run->config->attention_head_count;
    const uint32_t depth = hidden / heads;
    const uint64_t position_ids_shape[] = {sequence_length};
    const uint64_t positions_shape[] = {sequence_length, hidden};
    const uint64_t hidden_shape[] = {2U, sequence_length, hidden};
    const uint64_t attention_shape[] = {
        2U, heads, sequence_length, depth
    };
    const uint64_t reorder_shape[] = {
        2U, sequence_length, heads, depth
    };
    const uint64_t expanded_shape[] = {
        2U, sequence_length, run->config->intermediate_size
    };
    const uint64_t logits_shape[] = {2U, M3_RVQ_CODEBOOK_SIZE};
    const uint8_t permutation[] = {1U, 0U, 2U};
    m3_tensor_view sequence_slice;
    m3_status status;

    (void)memset(views, 0, sizeof(*views));
    m3_tensor_view_init(&sequence_slice);
    status = m3_tensor_slice(
        m3_runtime_workspace_view(&run->workspace, M3_RVQ_WS_SEQUENCE),
        0U, 0U, (uint64_t)sequence_length, &sequence_slice, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&sequence_slice, permutation,
                                   &views->sequence, error);
    }
#define M3_RVQ_BUILD_VIEW(member, slot, dtype, rank, shape)                 \
    if (status == M3_STATUS_OK) {                                           \
        status = m3_rvq_view(run, (slot), (dtype), (rank), (shape),          \
                             &views->member, error);                         \
    }
    M3_RVQ_BUILD_VIEW(position_ids, M3_RVQ_WS_POSITION_IDS, M3_DTYPE_I32,
                      1U, position_ids_shape)
    M3_RVQ_BUILD_VIEW(positions, M3_RVQ_WS_POSITIONS, run->config->dtype,
                      2U, positions_shape)
    M3_RVQ_BUILD_VIEW(hidden, M3_RVQ_WS_HIDDEN, run->config->dtype, 3U,
                      hidden_shape)
    M3_RVQ_BUILD_VIEW(hidden_temp, M3_RVQ_WS_HIDDEN_TEMP,
                      run->config->dtype, 3U, hidden_shape)
    M3_RVQ_BUILD_VIEW(norm, M3_RVQ_WS_NORM, run->config->dtype, 3U,
                      hidden_shape)
    M3_RVQ_BUILD_VIEW(query, M3_RVQ_WS_QUERY, run->config->dtype, 3U,
                      hidden_shape)
    M3_RVQ_BUILD_VIEW(key, M3_RVQ_WS_KEY, run->config->dtype, 3U,
                      hidden_shape)
    M3_RVQ_BUILD_VIEW(value, M3_RVQ_WS_VALUE, run->config->dtype, 3U,
                      hidden_shape)
    M3_RVQ_BUILD_VIEW(attention, M3_RVQ_WS_ATTENTION,
                      run->config->dtype, 4U, attention_shape)
    M3_RVQ_BUILD_VIEW(reorder, M3_RVQ_WS_REORDER, run->config->dtype, 4U,
                      reorder_shape)
    M3_RVQ_BUILD_VIEW(gate, M3_RVQ_WS_GATE, run->config->dtype, 3U,
                      expanded_shape)
    M3_RVQ_BUILD_VIEW(up, M3_RVQ_WS_UP, run->config->dtype, 3U,
                      expanded_shape)
    M3_RVQ_BUILD_VIEW(logits, M3_RVQ_WS_LOGITS, run->config->dtype, 2U,
                      logits_shape)
#undef M3_RVQ_BUILD_VIEW
    return status;
}

static m3_status m3_rvq_begin_step(m3_rvq_run *run,
                                   m3_rvq_step_views *views,
                                   m3_error *error)
{
    m3_command commands[2] = {0};

    commands[0].kind = M3_OP_EMBEDDING;
    commands[0].descriptor.embedding.ids = &views->position_ids;
    commands[0].descriptor.embedding.table =
        run->weights->position_embeddings;
    commands[0].descriptor.embedding.output = &views->positions;
    commands[1].kind = M3_OP_ADD;
    commands[1].descriptor.add.left = &views->sequence;
    commands[1].descriptor.add.right = &views->positions;
    commands[1].descriptor.add.output = &views->hidden;
    return m3_command_executor_execute(&run->executor, commands, 2U,
                                       error);
}

static m3_status m3_rvq_layer(m3_rvq_run *run,
                              m3_rvq_step_views *views,
                              const m3_rvq_layer_weights *weights,
                              size_t sequence_length, m3_error *error)
{
    const uint64_t projected_shape[] = {
        2U, sequence_length, run->config->attention_head_count,
        run->config->hidden_size / run->config->attention_head_count
    };
    const uint64_t merged_shape[] = {
        2U, sequence_length, run->config->hidden_size
    };
    const uint8_t to_heads[] = {0U, 2U, 1U, 3U};
    const uint8_t to_sequence[] = {0U, 2U, 1U, 3U};
    m3_tensor_view attention_sequence;
    m3_tensor_view key_heads;
    m3_tensor_view merged;
    m3_tensor_view query_heads;
    m3_tensor_view query_split;
    m3_tensor_view key_split;
    m3_tensor_view value_heads;
    m3_tensor_view value_split;
    m3_command commands[14] = {0};
    m3_status status;

    m3_tensor_view_init(&query_split);
    m3_tensor_view_init(&key_split);
    m3_tensor_view_init(&value_split);
    m3_tensor_view_init(&query_heads);
    m3_tensor_view_init(&key_heads);
    m3_tensor_view_init(&value_heads);
    m3_tensor_view_init(&attention_sequence);
    m3_tensor_view_init(&merged);
    status = m3_tensor_reshape(&views->query, 4U, projected_shape,
                               &query_split, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(&views->key, 4U, projected_shape,
                                   &key_split, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(&views->value, 4U, projected_shape,
                                   &value_split, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&query_split, to_heads, &query_heads,
                                   error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&key_split, to_heads, &key_heads, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&value_split, to_heads, &value_heads,
                                   error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&views->attention, to_sequence,
                                   &attention_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(&views->reorder, 3U, merged_shape,
                                   &merged, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_RMS_NORM;
    commands[0].descriptor.rms_norm = (m3_op_rms_norm){
        &views->hidden, weights->input_norm, &views->norm, 1.0e-6F
    };
    m3_rvq_linear(&commands[1], &views->norm, weights->query,
                  &views->query);
    m3_rvq_linear(&commands[2], &views->norm, weights->key, &views->key);
    m3_rvq_linear(&commands[3], &views->norm, weights->value,
                  &views->value);
    commands[4].kind = M3_OP_ATTENTION;
    commands[4].descriptor.attention = (m3_op_attention){
        &query_heads, &key_heads, &value_heads, NULL, &views->attention,
        1.0F / sqrtf((float)(run->config->hidden_size /
                            run->config->attention_head_count)),
        0, true
    };
    m3_rvq_copy(&commands[5], &attention_sequence, &views->reorder);
    m3_rvq_linear(&commands[6], &merged, weights->attention_out,
                  &views->hidden_temp);
    commands[7].kind = M3_OP_ADD;
    commands[7].descriptor.add = (m3_op_binary){
        &views->hidden, &views->hidden_temp, &views->hidden
    };
    commands[8].kind = M3_OP_RMS_NORM;
    commands[8].descriptor.rms_norm = (m3_op_rms_norm){
        &views->hidden, weights->post_attention_norm, &views->norm,
        1.0e-6F
    };
    m3_rvq_linear(&commands[9], &views->norm, weights->gate,
                  &views->gate);
    m3_rvq_linear(&commands[10], &views->norm, weights->up, &views->up);
    commands[11].kind = M3_OP_GATED_SILU;
    commands[11].descriptor.gated_silu = (m3_op_binary){
        &views->gate, &views->up, &views->gate
    };
    m3_rvq_linear(&commands[12], &views->gate, weights->down,
                  &views->hidden_temp);
    commands[13].kind = M3_OP_ADD;
    commands[13].descriptor.add = (m3_op_binary){
        &views->hidden, &views->hidden_temp, &views->hidden
    };
    return m3_command_executor_execute(&run->executor, commands, 14U,
                                       error);
}

static m3_status m3_rvq_finish_step(
    m3_rvq_run *run, m3_rvq_step_views *views, size_t head_index,
    size_t sequence_length, float uniform, m3_error *error)
{
    const uint64_t last_shape[] = {2U, run->config->hidden_size};
    const size_t element_size = m3_dtype_size(run->config->dtype);
    const size_t strides[] = {
        views->norm.byte_strides[0], element_size
    };
    size_t last_offset = views->norm.byte_offset +
                         (sequence_length - 1U) *
                             views->norm.byte_strides[1];
    m3_tensor_view condition;
    m3_tensor_view conditional_last;
    m3_tensor_view last;
    m3_command commands[3] = {0};
    uint32_t code = UINT32_MAX;
    m3_status status;

    m3_tensor_view_init(&last);
    status = m3_tensor_view_strided(
        &last, views->norm.storage, run->config->dtype, 2U, last_shape,
        strides, last_offset, error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_row(&last, 0U, &conditional_last, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_vector(
            run, head_index + 1U, &condition, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_RMS_NORM;
    commands[0].descriptor.rms_norm = (m3_op_rms_norm){
        &views->hidden, run->weights->norm, &views->norm, 1.0e-6F
    };
    m3_rvq_linear(&commands[1], &last, run->weights->heads[head_index],
                  &views->logits);
    m3_rvq_copy(&commands[2], &conditional_last, &condition);
    status = m3_command_executor_execute(&run->executor, commands, 3U,
                                         error);
    if (status == M3_STATUS_OK) {
        status = m3_guided_sample_residual(
            run->backend, &views->logits, uniform, &code, error);
    }
    if (status == M3_STATUS_OK) {
        run->built.codes[head_index] = code;
    }
    return status;
}

static m3_status m3_rvq_append_code(m3_rvq_run *run, size_t head_index,
                                    m3_error *error)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t token_shape[] = {2U, run->config->hidden_size};
    m3_tensor_view ids;
    m3_tensor_view next;
    m3_tensor_view token;
    m3_command commands[2] = {0};
    int32_t embedding_id;
    int32_t repeated[2];
    m3_status status;

    if (!m3_rvq_next_embedding_id(
            head_index, run->built.codes[head_index], &embedding_id)) {
        return M3_STATUS_OK;
    }
    repeated[0] = embedding_id;
    repeated[1] = embedding_id;
    status = m3_storage_write(
        run->workspace.storages[M3_RVQ_WS_CODE_IDS], 0U, repeated,
        sizeof(repeated), error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_view(run, M3_RVQ_WS_CODE_IDS, M3_DTYPE_I32, 1U,
                             ids_shape, &ids, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_view(run, M3_RVQ_WS_TOKEN, run->config->dtype, 2U,
                             token_shape, &token, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_sequence_position(
            run, head_index + 2U, &next, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_EMBEDDING;
    commands[0].descriptor.embedding = (m3_op_embedding){
        &ids, run->weights->audio_embeddings, &token
    };
    m3_rvq_linear(&commands[1], &token, run->weights->projection, &next);
    return m3_command_executor_execute(&run->executor, commands, 2U,
                                       error);
}

static m3_status m3_rvq_run_steps(m3_rvq_run *run, const float *uniforms,
                                  m3_progress_callback progress,
                                  void *progress_context, m3_error *error)
{
    size_t head;
    m3_status status = M3_STATUS_OK;

    for (head = 0U; head < M3_RVQ_RESIDUAL_COUNT &&
                    status == M3_STATUS_OK; ++head) {
        m3_rvq_step_views views;
        size_t sequence_length = m3_rvq_sequence_length(head);
        size_t layer;

        status = m3_rvq_build_step_views(
            run, sequence_length, &views, error);
        if (status == M3_STATUS_OK) {
            status = m3_rvq_begin_step(run, &views, error);
        }
        for (layer = 0U; layer < run->config->layer_count &&
                         status == M3_STATUS_OK; ++layer) {
            status = m3_rvq_layer(
                run, &views, &run->weights->layers[layer],
                sequence_length, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_rvq_finish_step(
                run, &views, head, sequence_length, uniforms[head], error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_rvq_append_code(run, head, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_rvq_condition_cancel(
                progress, progress_context, (uint64_t)(head + 1U),
                M3_RVQ_RESIDUAL_COUNT, "RVQ frame decode", error);
        }
    }
    return status;
}

m3_status m3_rvq_decode_frame_core(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights, const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, m3_progress_callback progress,
    void *progress_context, m3_rvq_frame *frame, m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_RVQ_WORKSPACE_COUNT];
    m3_rvq_run run;
    uint64_t output_shape[3];
    size_t output_bytes = 0U;
    m3_status status;

    (void)memset(&run, 0, sizeof(run));
    run.backend = backend;
    run.config = config;
    run.weights = weights;
    m3_runtime_workspace_init(&run.workspace);
    m3_command_executor_init(&run.executor, backend);
    m3_rvq_frame_init(&run.built);
    status = m3_rvq_validate(
        backend, config, weights, last_hidden, semantic_embedding,
        uniforms, uniform_count, frame, &output_bytes, error);
    if (status == M3_STATUS_OK) {
        m3_rvq_workspace_specs(config, specs);
        status = m3_rvq_condition_preflight(
            backend, output_bytes, specs, M3_RVQ_WORKSPACE_COUNT, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_cancel(
            progress, progress_context, 0U, M3_RVQ_RESIDUAL_COUNT,
            "RVQ frame decode", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_allocate(
            backend, output_bytes, 64U, &run.built.storage, error);
    }
    output_shape[0] = 1U;
    output_shape[1] = M3_RVQ_CODEBOOK_COUNT;
    output_shape[2] = config != NULL ? config->hidden_size : 0U;
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &run.built.conditioning, run.built.storage, config->dtype, 3U,
            output_shape, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_build(
            &run.workspace, backend, specs, M3_RVQ_WORKSPACE_COUNT, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_setup(
            &run, last_hidden, semantic_embedding, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_run_steps(
            &run, uniforms, progress, progress_context, error);
    }
    m3_runtime_workspace_dispose(&run.workspace);
    m3_command_executor_dispose(&run.executor);
    if (status != M3_STATUS_OK) {
        m3_rvq_frame_dispose(&run.built);
        return status;
    }
    m3_rvq_frame_dispose(frame);
    *frame = run.built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_rvq_decode_frame(
    m3_backend *backend, const m3_rvq_weights *weights,
    const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, m3_progress_callback progress,
    void *progress_context, m3_rvq_frame *frame, m3_error *error)
{
    static const m3_rvq_config config = {
        4096U, 4U, 16U, 6144U, 16U, M3_DTYPE_BF16
    };

    return m3_rvq_decode_frame_core(
        backend, &config, weights, last_hidden, semantic_embedding,
        uniforms, uniform_count, progress, progress_context, frame, error);
}
