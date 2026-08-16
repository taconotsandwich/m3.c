/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_graph_internal.h"

#include <string.h>

static m3_status m3_graph_input(m3_graph_value_id value,
                                 m3_tensor_view *views,
                                 size_t value_count, bool optional,
                                 const m3_tensor_view **view,
                                 m3_error *error)
{
    *view = NULL;
    if (value == M3_GRAPH_VALUE_NONE) {
        if (optional) {
            *view = NULL;
            return M3_STATUS_OK;
        }
        (void)m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                           "required graph node input is absent");
        return M3_STATUS_INVALID_ARGUMENT;
    }
    if ((size_t)value >= value_count) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                           "graph node input is out of range");
        return M3_STATUS_OUT_OF_RANGE;
    }
    *view = &views[value];
    return M3_STATUS_OK;
}

static m3_status m3_graph_output(m3_graph_value_id value,
                                  m3_tensor_view *views,
                                  size_t value_count,
                                  m3_tensor_view **view,
                                  m3_error *error)
{
    *view = NULL;
    if (value == M3_GRAPH_VALUE_NONE || (size_t)value >= value_count) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                           "graph node output is absent or out of range");
        return M3_STATUS_OUT_OF_RANGE;
    }
    *view = &views[value];
    return M3_STATUS_OK;
}

static m3_status m3_graph_bind_unary(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_op_unary *operation, m3_error *error)
{
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false,
        &operation->input, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &operation->output, error);
    }
    return status;
}

static m3_status m3_graph_bind_binary(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_op_binary *operation, m3_error *error)
{
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false,
        &operation->left, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[1], views, value_count,
                                false, &operation->right, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &operation->output, error);
    }
    return status;
}

static m3_status m3_graph_bind_linear(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_command *command, m3_error *error)
{
    m3_op_linear *operation = &command->descriptor.linear;
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false,
        &operation->input, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[1], views, value_count,
                                false, &operation->weight, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[2], views, value_count,
                                true, &operation->bias, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &operation->output, error);
    }
    return status;
}

static m3_status m3_graph_bind_norm(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_command *command, m3_error *error)
{
    const m3_tensor_view *input = NULL;
    const m3_tensor_view *scale = NULL;
    const m3_tensor_view *bias = NULL;
    m3_tensor_view *output = NULL;
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false, &input, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[1], views, value_count,
                                false, &scale, error);
    }
    if (status == M3_STATUS_OK && description->kind == M3_OP_LAYER_NORM) {
        status = m3_graph_input(description->inputs[2], views, value_count,
                                true, &bias, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &output, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (description->kind == M3_OP_RMS_NORM) {
        command->descriptor.rms_norm.input = input;
        command->descriptor.rms_norm.scale = scale;
        command->descriptor.rms_norm.output = output;
        command->descriptor.rms_norm.epsilon =
            description->parameters.norm.epsilon;
    } else {
        command->descriptor.layer_norm.input = input;
        command->descriptor.layer_norm.scale = scale;
        command->descriptor.layer_norm.bias = bias;
        command->descriptor.layer_norm.output = output;
        command->descriptor.layer_norm.epsilon =
            description->parameters.norm.epsilon;
    }
    return M3_STATUS_OK;
}

static m3_status m3_graph_bind_rope(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_command *command, m3_error *error)
{
    m3_op_rope *operation = &command->descriptor.rope;
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false,
        &operation->input, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[1], views, value_count,
                                false, &operation->cosines, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[2], views, value_count,
                                false, &operation->sines, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &operation->output, error);
    }
    operation->position_offset = description->parameters.rope.position_offset;
    operation->rotary_dimension =
        description->parameters.rope.rotary_dimension;
    operation->mode = description->parameters.rope.mode;
    return status;
}

static m3_status m3_graph_bind_attention(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_command *command, m3_error *error)
{
    m3_op_attention *operation = &command->descriptor.attention;
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false,
        &operation->query, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[1], views, value_count,
                                false, &operation->key, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[2], views, value_count,
                                false, &operation->value, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[3], views, value_count,
                                true, &operation->mask, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &operation->output, error);
    }
    operation->scale = description->parameters.attention.scale;
    operation->causal_offset =
        description->parameters.attention.causal_offset;
    operation->causal = description->parameters.attention.causal;
    return status;
}

static m3_status m3_graph_bind_convolution(
    const m3_graph_node_desc *description, m3_tensor_view *views,
    size_t value_count, m3_command *command, m3_error *error)
{
    const m3_tensor_view *input = NULL;
    const m3_tensor_view *weight = NULL;
    const m3_tensor_view *bias = NULL;
    m3_tensor_view *output = NULL;
    m3_status status = m3_graph_input(
        description->inputs[0], views, value_count, false, &input, error);

    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[1], views, value_count,
                                false, &weight, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_input(description->inputs[2], views, value_count,
                                true, &bias, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_output(description->outputs[0], views,
                                 value_count, &output, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (description->kind == M3_OP_CONV1D) {
        m3_op_conv1d *operation = &command->descriptor.conv1d;

        operation->input = input;
        operation->weight = weight;
        operation->bias = bias;
        operation->output = output;
        operation->groups = description->parameters.convolution.groups;
        operation->stride = description->parameters.convolution.stride;
        operation->dilation = description->parameters.convolution.dilation;
        operation->pad_left = description->parameters.convolution.pad_left;
        operation->pad_right = description->parameters.convolution.pad_right;
    } else {
        m3_op_conv_transpose1d *operation =
            &command->descriptor.conv_transpose1d;

        operation->input = input;
        operation->weight = weight;
        operation->bias = bias;
        operation->output = output;
        operation->groups = description->parameters.convolution.groups;
        operation->stride = description->parameters.convolution.stride;
        operation->dilation = description->parameters.convolution.dilation;
        operation->pad_left = description->parameters.convolution.pad_left;
        operation->pad_right = description->parameters.convolution.pad_right;
        operation->output_padding =
            description->parameters.convolution.output_padding;
    }
    return M3_STATUS_OK;
}

m3_status m3_graph_bind_command(const m3_graph_node_desc *description,
                                m3_tensor_view *views, size_t value_count,
                                m3_command *command, m3_error *error)
{
    m3_status status;

    if (description == NULL || views == NULL || command == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "graph command binding is incomplete");
    }
    (void)memset(command, 0, sizeof(*command));
    command->kind = description->kind;
    switch (description->kind) {
    case M3_OP_COPY:
        status = m3_graph_bind_unary(description, views, value_count,
                                     &command->descriptor.copy, error);
        break;
    case M3_OP_CAST:
        status = m3_graph_bind_unary(description, views, value_count,
                                     &command->descriptor.cast, error);
        break;
    case M3_OP_SOFTMAX:
        status = m3_graph_bind_unary(description, views, value_count,
                                     &command->descriptor.softmax, error);
        break;
    case M3_OP_NEAREST_RESIZE1D:
        status = m3_graph_bind_unary(
            description, views, value_count,
            &command->descriptor.nearest_resize1d, error);
        break;
    case M3_OP_TANH:
        status = m3_graph_bind_unary(description, views, value_count,
                                     &command->descriptor.tanh, error);
        break;
    case M3_OP_ADD:
        status = m3_graph_bind_binary(description, views, value_count,
                                      &command->descriptor.add, error);
        break;
    case M3_OP_MUL:
        status = m3_graph_bind_binary(description, views, value_count,
                                      &command->descriptor.mul, error);
        break;
    case M3_OP_MATMUL:
        status = m3_graph_input(description->inputs[0], views, value_count,
                                false, &command->descriptor.matmul.left,
                                error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_input(
                description->inputs[1], views, value_count, false,
                &command->descriptor.matmul.right, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_graph_output(
                description->outputs[0], views, value_count,
                &command->descriptor.matmul.output, error);
        }
        break;
    case M3_OP_GATED_SILU:
        status = m3_graph_bind_binary(description, views, value_count,
                                      &command->descriptor.gated_silu,
                                      error);
        break;
    case M3_OP_EMBEDDING:
        status = m3_graph_input(description->inputs[0], views, value_count,
                                false, &command->descriptor.embedding.ids,
                                error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_input(
                description->inputs[1], views, value_count, false,
                &command->descriptor.embedding.table, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_graph_output(
                description->outputs[0], views, value_count,
                &command->descriptor.embedding.output, error);
        }
        break;
    case M3_OP_LINEAR:
        status = m3_graph_bind_linear(description, views, value_count,
                                      command, error);
        break;
    case M3_OP_RMS_NORM:
    case M3_OP_LAYER_NORM:
        status = m3_graph_bind_norm(description, views, value_count, command,
                                    error);
        break;
    case M3_OP_ROPE:
        status = m3_graph_bind_rope(description, views, value_count, command,
                                    error);
        break;
    case M3_OP_ATTENTION:
        status = m3_graph_bind_attention(description, views, value_count,
                                         command, error);
        break;
    case M3_OP_TOP_K:
        status = m3_graph_input(description->inputs[0], views, value_count,
                                false, &command->descriptor.top_k.logits,
                                error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_output(
                description->outputs[0], views, value_count,
                &command->descriptor.top_k.values, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_graph_output(
                description->outputs[1], views, value_count,
                &command->descriptor.top_k.indices, error);
        }
        command->descriptor.top_k.k = description->parameters.top_k.k;
        break;
    case M3_OP_CATEGORICAL:
        status = m3_graph_input(
            description->inputs[0], views, value_count, false,
            &command->descriptor.categorical.probabilities, error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_input(
                description->inputs[1], views, value_count, false,
                &command->descriptor.categorical.uniforms, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_graph_output(
                description->outputs[0], views, value_count,
                &command->descriptor.categorical.output, error);
        }
        break;
    case M3_OP_CONV1D:
    case M3_OP_CONV_TRANSPOSE1D:
        status = m3_graph_bind_convolution(
            description, views, value_count, command, error);
        break;
    case M3_OP_SNAKE1D:
        status = m3_graph_input(description->inputs[0], views, value_count,
                                false, &command->descriptor.snake1d.input,
                                error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_input(
                description->inputs[1], views, value_count, false,
                &command->descriptor.snake1d.alpha, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_graph_output(
                description->outputs[0], views, value_count,
                &command->descriptor.snake1d.output, error);
        }
        break;
    default:
        status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                              "unsupported graph operation kind %d",
                              (int)description->kind);
        break;
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}
