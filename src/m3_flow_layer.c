/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <math.h>
#include <string.h>

typedef struct {
    m3_tensor_view hidden;
    m3_tensor_view hidden_temp;
    m3_tensor_view normalized;
    m3_tensor_view query;
    m3_tensor_view key;
    m3_tensor_view value;
    m3_tensor_view query_heads;
    m3_tensor_view key_heads;
    m3_tensor_view value_heads;
    m3_tensor_view query_rotary;
    m3_tensor_view key_rotary;
    m3_tensor_view attention;
    m3_tensor_view attention_sequence;
    m3_tensor_view reorder;
    m3_tensor_view merged;
    m3_tensor_view feed_forward;
    m3_tensor_view feed_forward_first;
    m3_tensor_view feed_forward_gate;
    m3_tensor_view gated;
    m3_tensor_view cosines;
    m3_tensor_view sines;
} m3_flow_layer_views;

static void m3_flow_linear(m3_command *command,
                           const m3_tensor_view *input,
                           const m3_tensor_view *weight,
                           const m3_tensor_view *bias,
                           m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_LINEAR;
    command->descriptor.linear = (m3_op_linear){
        input, weight, bias, output
    };
}

static m3_status m3_flow_layer_workspace_views(
    m3_flow_run *run, uint64_t length, m3_flow_layer_views *views,
    m3_error *error)
{
    const uint64_t inner = (uint64_t)run->config->attention_heads *
                           run->config->head_dimension;
    const uint64_t sequence = length + 1U;
    const uint64_t hidden_shape[] = {2U, sequence, inner};
    const uint64_t heads_shape[] = {
        2U, run->config->attention_heads, sequence,
        run->config->head_dimension
    };
    const uint64_t reorder_shape[] = {
        2U, sequence, run->config->attention_heads,
        run->config->head_dimension
    };
    const uint64_t feed_forward_shape[] = {
        2U, sequence,
        (uint64_t)run->config->feed_forward_dimension * 2U
    };
    const uint64_t gated_shape[] = {
        2U, sequence, run->config->feed_forward_dimension
    };
    const uint64_t rotary_shape[] = {
        sequence, run->config->rotary_dimension / 2U
    };
    m3_status status;

    (void)memset(views, 0, sizeof(*views));
#define M3_FLOW_VIEW(member, slot, rank, shape)                              \
    status = m3_flow_view(run, slot, M3_DTYPE_F32, rank, shape,              \
                          &views->member, error);                             \
    if (status != M3_STATUS_OK) {                                            \
        return status;                                                       \
    }
    M3_FLOW_VIEW(hidden, M3_FLOW_WS_HIDDEN, 3U, hidden_shape)
    M3_FLOW_VIEW(hidden_temp, M3_FLOW_WS_HIDDEN_TEMP, 3U, hidden_shape)
    M3_FLOW_VIEW(normalized, M3_FLOW_WS_NORMALIZED, 3U, hidden_shape)
    M3_FLOW_VIEW(query, M3_FLOW_WS_QUERY, 3U, hidden_shape)
    M3_FLOW_VIEW(key, M3_FLOW_WS_KEY, 3U, hidden_shape)
    M3_FLOW_VIEW(value, M3_FLOW_WS_VALUE, 3U, hidden_shape)
    M3_FLOW_VIEW(query_rotary, M3_FLOW_WS_QUERY_ROTARY, 4U, heads_shape)
    M3_FLOW_VIEW(key_rotary, M3_FLOW_WS_KEY_ROTARY, 4U, heads_shape)
    M3_FLOW_VIEW(attention, M3_FLOW_WS_ATTENTION, 4U, heads_shape)
    M3_FLOW_VIEW(reorder, M3_FLOW_WS_REORDER, 4U, reorder_shape)
    M3_FLOW_VIEW(feed_forward, M3_FLOW_WS_FEED_FORWARD, 3U,
                 feed_forward_shape)
    M3_FLOW_VIEW(gated, M3_FLOW_WS_GATED, 3U, gated_shape)
    M3_FLOW_VIEW(cosines, M3_FLOW_WS_COSINES, 2U, rotary_shape)
    M3_FLOW_VIEW(sines, M3_FLOW_WS_SINES, 2U, rotary_shape)
#undef M3_FLOW_VIEW
    return M3_STATUS_OK;
}

static m3_status m3_flow_layer_transform_views(
    m3_flow_run *run, m3_flow_layer_views *views, uint64_t length,
    m3_error *error)
{
    const uint64_t sequence = length + 1U;
    const uint64_t split_shape[] = {
        2U, sequence, run->config->attention_heads,
        run->config->head_dimension
    };
    const uint64_t merged_shape[] = {
        2U, sequence,
        (uint64_t)run->config->attention_heads *
            run->config->head_dimension
    };
    const uint8_t to_heads[] = {0U, 2U, 1U, 3U};
    const uint8_t to_sequence[] = {0U, 2U, 1U, 3U};
    m3_tensor_view query_split;
    m3_tensor_view key_split;
    m3_tensor_view value_split;
    m3_status status;

    m3_tensor_view_init(&query_split);
    m3_tensor_view_init(&key_split);
    m3_tensor_view_init(&value_split);
    status = m3_tensor_reshape(
        &views->query, 4U, split_shape, &query_split, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &views->key, 4U, split_shape, &key_split, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &views->value, 4U, split_shape, &value_split, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &query_split, to_heads, &views->query_heads, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &key_split, to_heads, &views->key_heads, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &value_split, to_heads, &views->value_heads, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &views->attention, to_sequence,
            &views->attention_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &views->reorder, 3U, merged_shape, &views->merged, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &views->feed_forward, 2U, 0U,
            run->config->feed_forward_dimension,
            &views->feed_forward_first, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &views->feed_forward, 2U,
            run->config->feed_forward_dimension,
            run->config->feed_forward_dimension,
            &views->feed_forward_gate, error);
    }
    return status;
}

static m3_status m3_flow_layer_attention(
    m3_flow_run *run, const m3_flow_layer_weights *weights,
    m3_flow_layer_views *views, m3_error *error)
{
    m3_command commands[3] = {0};
    m3_status status;

    commands[0].kind = M3_OP_LAYER_NORM;
    commands[0].descriptor.layer_norm = (m3_op_layer_norm){
        &views->hidden, weights->norm1_scale, weights->norm1_bias,
        &views->normalized, run->config->layer_norm_epsilon
    };
    status = m3_command_executor_execute(
        &run->executor, commands, 1U, error);
    m3_flow_linear(&commands[0], &views->normalized, weights->query,
                   NULL, &views->query);
    m3_flow_linear(&commands[1], &views->normalized, weights->key,
                   NULL, &views->key);
    m3_flow_linear(&commands[2], &views->normalized, weights->value,
                   NULL, &views->value);
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, commands, 3U, error);
    }
    (void)memset(commands, 0, sizeof(commands));
    commands[0].kind = M3_OP_ROPE;
    commands[0].descriptor.rope = (m3_op_rope){
        &views->query_heads, &views->cosines, &views->sines,
        &views->query_rotary, 0U, run->config->rotary_dimension,
        M3_ROPE_HALF_SPLIT
    };
    commands[1].kind = M3_OP_ROPE;
    commands[1].descriptor.rope = (m3_op_rope){
        &views->key_heads, &views->cosines, &views->sines,
        &views->key_rotary, 0U, run->config->rotary_dimension,
        M3_ROPE_HALF_SPLIT
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, commands, 2U, error);
    }
    (void)memset(commands, 0, sizeof(commands));
    commands[0].kind = M3_OP_ATTENTION;
    commands[0].descriptor.attention = (m3_op_attention){
        &views->query_rotary, &views->key_rotary, &views->value_heads,
        NULL, &views->attention,
        1.0F / sqrtf((float)run->config->head_dimension), 0, false
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, commands, 1U, error);
    }
    return status;
}

static m3_status m3_flow_layer_consume_attention(
    m3_flow_run *run, const m3_flow_layer_weights *weights,
    m3_flow_layer_views *views, m3_error *error)
{
    m3_command command = {0};
    m3_status status;

    command.kind = M3_OP_COPY;
    command.descriptor.copy = (m3_op_unary){
        &views->attention_sequence, &views->reorder
    };
    status = m3_command_executor_execute(
        &run->executor, &command, 1U, error);
    m3_flow_linear(&command, &views->merged, weights->attention_out,
                   NULL, &views->hidden_temp);
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ADD;
    command.descriptor.add = (m3_op_binary){
        &views->hidden, &views->hidden_temp, &views->hidden
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    return status;
}

static m3_status m3_flow_layer_feed_forward(
    m3_flow_run *run, const m3_flow_layer_weights *weights,
    m3_flow_layer_views *views, m3_error *error)
{
    m3_command command = {0};
    m3_status status;

    command.kind = M3_OP_LAYER_NORM;
    command.descriptor.layer_norm = (m3_op_layer_norm){
        &views->hidden, weights->norm2_scale, weights->norm2_bias,
        &views->normalized, run->config->layer_norm_epsilon
    };
    status = m3_command_executor_execute(
        &run->executor, &command, 1U, error);
    m3_flow_linear(&command, &views->normalized,
                   weights->feed_forward_in,
                   weights->feed_forward_in_bias,
                   &views->feed_forward);
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_GATED_SILU;
    command.descriptor.gated_silu = (m3_op_binary){
        &views->feed_forward_gate, &views->feed_forward_first,
        &views->gated
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    m3_flow_linear(&command, &views->gated,
                   weights->feed_forward_out,
                   weights->feed_forward_out_bias,
                   &views->hidden_temp);
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ADD;
    command.descriptor.add = (m3_op_binary){
        &views->hidden, &views->hidden_temp, &views->hidden
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    return status;
}

m3_status m3_flow_transformer_layer(
    m3_flow_run *run, const m3_flow_layer_weights *weights,
    uint64_t length, m3_error *error)
{
    m3_flow_layer_views views;
    m3_status status = m3_flow_layer_workspace_views(
        run, length, &views, error);

    if (status == M3_STATUS_OK) {
        status = m3_flow_layer_transform_views(
            run, &views, length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_layer_attention(
            run, weights, &views, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_layer_consume_attention(
            run, weights, &views, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_layer_feed_forward(
            run, weights, &views, error);
    }
    return status;
}
