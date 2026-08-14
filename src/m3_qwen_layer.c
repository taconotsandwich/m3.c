/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_qwen_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static m3_status m3_qwen_run(m3_command_executor *executor,
                             m3_command *command, m3_error *error)
{
    return m3_command_executor_execute(executor, command, 1U, error);
}

static m3_status m3_qwen_rms(m3_command_executor *executor,
                             const m3_tensor_view *input,
                             const m3_tensor_view *scale,
                             m3_tensor_view *output, float epsilon,
                             m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_RMS_NORM;
    command.descriptor.rms_norm.input = input;
    command.descriptor.rms_norm.scale = scale;
    command.descriptor.rms_norm.output = output;
    command.descriptor.rms_norm.epsilon = epsilon;
    return m3_qwen_run(executor, &command, error);
}

static m3_status m3_qwen_linear(m3_command_executor *executor,
                                const m3_tensor_view *input,
                                const m3_tensor_view *weight,
                                m3_tensor_view *output,
                                m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = input;
    command.descriptor.linear.weight = weight;
    command.descriptor.linear.output = output;
    return m3_qwen_run(executor, &command, error);
}

static m3_status m3_qwen_copy(m3_command_executor *executor,
                              const m3_tensor_view *input,
                              m3_tensor_view *output, m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = input;
    command.descriptor.copy.output = output;
    return m3_qwen_run(executor, &command, error);
}

static m3_status m3_qwen_rope(m3_command_executor *executor,
                              const m3_tensor_view *input,
                              const m3_tensor_view *cosines,
                              const m3_tensor_view *sines,
                              m3_tensor_view *output, uint64_t position,
                              uint64_t head_dimension,
                              m3_error *error)
{
    m3_command command;

    if (head_dimension > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Qwen RoPE dimension exceeds uint32_t");
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ROPE;
    command.descriptor.rope.input = input;
    command.descriptor.rope.cosines = cosines;
    command.descriptor.rope.sines = sines;
    command.descriptor.rope.output = output;
    command.descriptor.rope.position_offset = position;
    command.descriptor.rope.rotary_dimension = (uint32_t)head_dimension;
    command.descriptor.rope.mode = M3_ROPE_HALF_SPLIT;
    return m3_qwen_run(executor, &command, error);
}

static m3_status m3_qwen_add(m3_command_executor *executor,
                             m3_tensor_view *hidden,
                             const m3_tensor_view *addition,
                             m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ADD;
    command.descriptor.add.left = hidden;
    command.descriptor.add.right = addition;
    command.descriptor.add.output = hidden;
    return m3_qwen_run(executor, &command, error);
}

static m3_status m3_qwen_gated_silu(m3_command_executor *executor,
                                    m3_tensor_view *gate,
                                    const m3_tensor_view *up,
                                    m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_GATED_SILU;
    command.descriptor.gated_silu.left = gate;
    command.descriptor.gated_silu.right = up;
    command.descriptor.gated_silu.output = gate;
    return m3_qwen_run(executor, &command, error);
}

static m3_status m3_qwen_attention(
    m3_command_executor *executor, const m3_tensor_view *query,
    const m3_tensor_view *key, const m3_tensor_view *value,
    m3_tensor_view *output, uint64_t position, uint64_t head_dimension,
    m3_error *error)
{
    m3_command command;

    if (position > (uint64_t)INT64_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Qwen causal offset exceeds int64_t");
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ATTENTION;
    command.descriptor.attention.query = query;
    command.descriptor.attention.key = key;
    command.descriptor.attention.value = value;
    command.descriptor.attention.output = output;
    command.descriptor.attention.scale =
        1.0F / sqrtf((float)head_dimension);
    command.descriptor.attention.causal_offset = (int64_t)position;
    command.descriptor.attention.causal = true;
    return m3_qwen_run(executor, &command, error);
}

typedef struct {
    m3_tensor_view query_reshaped;
    m3_tensor_view key_reshaped;
    m3_tensor_view value_reshaped;
    m3_tensor_view query_heads;
    m3_tensor_view key_heads;
    m3_tensor_view key_write;
    m3_tensor_view value_write;
    m3_tensor_view key_target;
    m3_tensor_view value_target;
    m3_tensor_view cached_key_physical;
    m3_tensor_view cached_value_physical;
    m3_tensor_view cached_key;
    m3_tensor_view cached_value;
    m3_tensor_view attention_input;
    m3_tensor_view attention_linear_input;
} m3_qwen_layer_views;

static m3_status m3_qwen_layer_views_build(
    const m3_qwen_dimensions *dimensions,
    m3_qwen_layer_workspace *workspace, m3_tensor_view *key_cache,
    m3_tensor_view *value_cache, uint64_t position,
    m3_qwen_layer_views *views, m3_error *error)
{
    const uint8_t heads_permutation[] = {0U, 2U, 1U, 3U};
    const uint8_t cache_permutation[] = {2U, 0U, 1U, 3U};
    const uint8_t value_cache_permutation[] = {1U, 0U, 2U, 3U};
    const uint8_t cached_permutation[] = {1U, 2U, 0U, 3U};
    const uint8_t attention_permutation[] = {0U, 2U, 1U, 3U};
    uint64_t sequence = workspace->hidden->metadata.shape[1];
    uint64_t query_shape[] = {2U, sequence, dimensions->query_heads,
                              dimensions->head_dimension};
    uint64_t key_shape[] = {2U, sequence,
                            dimensions->key_value_heads,
                            dimensions->head_dimension};
    uint64_t hidden_shape[] = {2U, sequence, dimensions->hidden_size};
    uint64_t cache_length;
    m3_status status;

    if (sequence > UINT64_MAX - position) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Qwen cache position overflows");
    }
    cache_length = position + sequence;
    if (key_cache->metadata.rank != 4U ||
        value_cache->metadata.rank != 4U ||
        key_cache->metadata.shape[0] < cache_length ||
        value_cache->metadata.shape[0] < cache_length) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Qwen layer exceeds cache capacity");
    }
    (void)memset(views, 0, sizeof(*views));
    status = m3_tensor_reshape(workspace->query, 4U, query_shape,
                               &views->query_reshaped, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(workspace->key, 4U, key_shape,
                                   &views->key_reshaped, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(workspace->value, 4U, key_shape,
                                   &views->value_reshaped, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&views->query_reshaped,
                                   heads_permutation,
                                   &views->query_heads, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&views->key_reshaped, heads_permutation,
                                   &views->key_heads, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&views->value_reshaped,
                                   value_cache_permutation,
                                   &views->value_write, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(workspace->rotated_key,
                                   cache_permutation,
                                   &views->key_write, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(key_cache, 0U, position, sequence,
                                 &views->key_target, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(value_cache, 0U, position, sequence,
                                 &views->value_target, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(key_cache, 0U, 0U, cache_length,
                                 &views->cached_key_physical, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(value_cache, 0U, 0U, cache_length,
                                 &views->cached_value_physical, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&views->cached_key_physical,
                                   cached_permutation,
                                   &views->cached_key, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(&views->cached_value_physical,
                                   cached_permutation,
                                   &views->cached_value, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(workspace->attention,
                                   attention_permutation,
                                   &views->attention_input, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(workspace->attention_reordered, 3U,
                                   hidden_shape,
                                   &views->attention_linear_input, error);
    }
    return status;
}

static m3_status m3_qwen_layer_project(
    m3_command_executor *executor, const m3_qwen_dimensions *dimensions,
    const m3_qwen_layer_weights *weights,
    m3_qwen_layer_workspace *workspace, uint64_t position,
    const m3_tensor_view *cosines, const m3_tensor_view *sines,
    m3_qwen_layer_views *views, m3_error *error)
{
    m3_status status = m3_qwen_rms(
        executor, workspace->hidden, weights->input_norm,
        workspace->normalized, dimensions->rms_epsilon, error);

    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, workspace->normalized,
                                weights->query_weight, workspace->query,
                                error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, workspace->normalized,
                                weights->key_weight, workspace->key,
                                error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, workspace->normalized,
                                weights->value_weight, workspace->value,
                                error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_rms(executor, &views->query_reshaped,
                             weights->query_norm,
                             &views->query_reshaped,
                             dimensions->rms_epsilon, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_rms(executor, &views->key_reshaped,
                             weights->key_norm, &views->key_reshaped,
                             dimensions->rms_epsilon, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_rope(executor, &views->query_heads, cosines,
                              sines, workspace->rotated_query, position,
                              dimensions->head_dimension, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_rope(executor, &views->key_heads, cosines, sines,
                              workspace->rotated_key, position,
                              dimensions->head_dimension, error);
    }
    return status;
}

m3_status m3_qwen_layer_execute(
    m3_command_executor *executor, const m3_qwen_dimensions *dimensions,
    const m3_qwen_layer_weights *weights,
    m3_qwen_layer_workspace *workspace, m3_tensor_view *key_cache,
    m3_tensor_view *value_cache, const m3_tensor_view *cosines,
    const m3_tensor_view *sines, uint64_t position,
    m3_error *error)
{
    m3_qwen_layer_views views;
    m3_status status;

    if (executor == NULL || dimensions == NULL || weights == NULL ||
        workspace == NULL || workspace->hidden == NULL ||
        key_cache == NULL || value_cache == NULL || cosines == NULL ||
        sines == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen layer execution inputs are required");
    }
    status = m3_qwen_layer_views_build(dimensions, workspace, key_cache,
                                       value_cache, position, &views,
                                       error);
    if (status == M3_STATUS_OK) {
        status = m3_qwen_layer_project(
            executor, dimensions, weights, workspace, position, cosines,
            sines, &views, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_copy(executor, &views.key_write,
                              &views.key_target, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_copy(executor, &views.value_write,
                              &views.value_target, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_attention(
            executor, workspace->rotated_query, &views.cached_key,
            &views.cached_value, workspace->attention, position,
            dimensions->head_dimension, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_copy(executor, &views.attention_input,
                              workspace->attention_reordered, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, &views.attention_linear_input,
                                weights->output_weight,
                                workspace->projection, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_add(executor, workspace->hidden,
                             workspace->projection, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_rms(
            executor, workspace->hidden, weights->post_attention_norm,
            workspace->normalized, dimensions->rms_epsilon, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, workspace->normalized,
                                weights->gate_weight, workspace->gate,
                                error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, workspace->normalized,
                                weights->up_weight, workspace->up, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_gated_silu(executor, workspace->gate,
                                    workspace->up, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_linear(executor, workspace->gate,
                                weights->down_weight, workspace->down,
                                error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_add(executor, workspace->hidden,
                             workspace->down, error);
    }
    return status;
}
