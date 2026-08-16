/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_graph_internal.h"

#include <string.h>

static m3_status m3_graph_infer_input(
    const m3_graph *graph, m3_graph_value_id value,
    const m3_tensor_metadata **metadata, m3_error *error)
{
    *metadata = NULL;
    if (value == M3_GRAPH_VALUE_NONE || (size_t)value >= graph->value_count) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                           "inferred graph input is absent or out of range");
        return M3_STATUS_OUT_OF_RANGE;
    }
    *metadata = &graph->values[value].metadata;
    return M3_STATUS_OK;
}

static bool m3_graph_u64_add(uint64_t left, uint64_t right,
                             uint64_t *result)
{
    if (left > UINT64_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool m3_graph_u64_multiply(uint64_t left, uint64_t right,
                                  uint64_t *result)
{
    if (left != 0U && right > UINT64_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static m3_status m3_graph_infer_broadcast(
    const m3_tensor_metadata *left, const m3_tensor_metadata *right,
    m3_graph_value_desc *output, m3_error *error)
{
    uint8_t rank = left->rank > right->rank ? left->rank : right->rank;
    uint8_t axis;

    output->rank = rank;
    for (axis = 0U; axis < rank; ++axis) {
        int left_axis = (int)axis - (int)(rank - left->rank);
        int right_axis = (int)axis - (int)(rank - right->rank);
        uint64_t left_size = left_axis < 0 ? 1U : left->shape[left_axis];
        uint64_t right_size = right_axis < 0 ? 1U : right->shape[right_axis];

        if (left_size != right_size && left_size != 1U &&
            right_size != 1U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "inferred broadcast dimensions disagree");
        }
        output->shape[axis] = left_size == 1U ? right_size : left_size;
    }
    return M3_STATUS_OK;
}

static m3_status m3_graph_infer_convolution(
    const m3_graph_node_desc *node, const m3_tensor_metadata *input,
    const m3_tensor_metadata *weight, m3_graph_value_desc *output,
    m3_error *error)
{
    uint64_t effective;
    uint64_t span;
    uint64_t positive;
    uint64_t padding;
    uint64_t output_channels;

    if (input->rank != 3U || weight->rank != 3U ||
        node->parameters.convolution.groups == 0U ||
        node->parameters.convolution.stride == 0U ||
        node->parameters.convolution.dilation == 0U ||
        input->shape[2] == 0U || weight->shape[2] == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "cannot infer invalid convolution dimensions");
    }
    output->rank = 3U;
    output->shape[0] = input->shape[0];
    if (node->kind == M3_OP_CONV1D) {
        output_channels = weight->shape[0];
        if (!m3_graph_u64_multiply(
                weight->shape[2] - 1U,
                node->parameters.convolution.dilation, &span) ||
            !m3_graph_u64_add(span, 1U, &effective) ||
            !m3_graph_u64_add(
                input->shape[2], node->parameters.convolution.pad_left,
                &positive) ||
            !m3_graph_u64_add(
                positive, node->parameters.convolution.pad_right,
                &positive) || effective > positive) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "Conv1d inferred length is invalid");
        }
        output->shape[2] =
            (positive - effective) /
                node->parameters.convolution.stride + 1U;
    } else {
        if (!m3_graph_u64_multiply(
                weight->shape[1], node->parameters.convolution.groups,
                &output_channels) ||
            !m3_graph_u64_multiply(
                input->shape[2] - 1U,
                node->parameters.convolution.stride, &positive) ||
            !m3_graph_u64_multiply(
                weight->shape[2] - 1U,
                node->parameters.convolution.dilation, &span) ||
            !m3_graph_u64_add(positive, span, &positive) ||
            !m3_graph_u64_add(
                positive, node->parameters.convolution.output_padding,
                &positive) ||
            !m3_graph_u64_add(positive, 1U, &positive) ||
            !m3_graph_u64_add(
                node->parameters.convolution.pad_left,
                node->parameters.convolution.pad_right, &padding) ||
            positive <= padding) {
            return m3_error_set(
                error, M3_STATUS_OVERFLOW,
                "ConvTranspose1d inferred length is invalid");
        }
        output->shape[2] = positive - padding;
    }
    output->shape[1] = output_channels;
    return M3_STATUS_OK;
}

static m3_status m3_graph_infer_shape(
    const m3_graph *graph, const m3_graph_node_desc *node,
    size_t output_index, m3_graph_value_desc *output, m3_error *error)
{
    const m3_tensor_metadata *first = NULL;
    const m3_tensor_metadata *second = NULL;
    m3_status status = m3_graph_infer_input(
        graph, node->inputs[0], &first, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (output_index != 0U && node->kind != M3_OP_TOP_K) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "operation has no requested inferred output");
    }
    switch (node->kind) {
    case M3_OP_COPY:
    case M3_OP_CAST:
    case M3_OP_RMS_NORM:
    case M3_OP_LAYER_NORM:
    case M3_OP_ROPE:
    case M3_OP_ATTENTION:
    case M3_OP_SOFTMAX:
    case M3_OP_SNAKE1D:
    case M3_OP_TANH:
        output->rank = first->rank;
        (void)memcpy(output->shape, first->shape, sizeof(output->shape));
        if (node->kind == M3_OP_COPY) {
            output->dtype = first->dtype;
        } else if (node->kind == M3_OP_SNAKE1D ||
                   node->kind == M3_OP_TANH) {
            output->dtype = M3_DTYPE_F32;
        }
        break;
    case M3_OP_ADD:
    case M3_OP_MUL:
        status = m3_graph_infer_input(
            graph, node->inputs[1], &second, error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_infer_broadcast(first, second, output, error);
        }
        break;
    case M3_OP_GATED_SILU:
        output->rank = first->rank;
        (void)memcpy(output->shape, first->shape, sizeof(output->shape));
        break;
    case M3_OP_EMBEDDING:
        status = m3_graph_infer_input(
            graph, node->inputs[1], &second, error);
        if (status == M3_STATUS_OK && first->rank < M3_TENSOR_MAX_RANK &&
            second->rank == 2U) {
            output->rank = first->rank + 1U;
            (void)memcpy(output->shape, first->shape,
                         sizeof(output->shape));
            output->shape[first->rank] = second->shape[1];
        } else if (status == M3_STATUS_OK) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "cannot infer embedding shape");
        }
        break;
    case M3_OP_MATMUL:
        status = m3_graph_infer_input(
            graph, node->inputs[1], &second, error);
        if (status == M3_STATUS_OK && first->rank == 2U &&
            second->rank == 2U) {
            output->rank = 2U;
            output->shape[0] = first->shape[0];
            output->shape[1] = second->shape[1];
        } else if (status == M3_STATUS_OK) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "cannot infer matmul shape");
        }
        break;
    case M3_OP_LINEAR:
        status = m3_graph_infer_input(
            graph, node->inputs[1], &second, error);
        if (status == M3_STATUS_OK && first->rank != 0U &&
            second->rank == 2U) {
            output->rank = first->rank;
            (void)memcpy(output->shape, first->shape,
                         sizeof(output->shape));
            output->shape[first->rank - 1U] = second->shape[0];
        } else if (status == M3_STATUS_OK) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "cannot infer linear shape");
        }
        break;
    case M3_OP_TOP_K:
        if (first->rank == 0U || output_index > 1U) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "cannot infer top-k shape");
        } else {
            output->rank = first->rank;
            (void)memcpy(output->shape, first->shape,
                         sizeof(output->shape));
            output->shape[first->rank - 1U] = node->parameters.top_k.k;
            output->dtype = output_index == 0U ? M3_DTYPE_F32 : M3_DTYPE_I32;
        }
        break;
    case M3_OP_CATEGORICAL:
        if (first->rank == 0U) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "cannot infer categorical shape");
        } else {
            output->rank = first->rank - 1U;
            (void)memcpy(output->shape, first->shape,
                         sizeof(output->shape));
            output->dtype = M3_DTYPE_I32;
        }
        break;
    case M3_OP_CONV1D:
    case M3_OP_CONV_TRANSPOSE1D:
        status = m3_graph_infer_input(
            graph, node->inputs[1], &second, error);
        if (status == M3_STATUS_OK) {
            status = m3_graph_infer_convolution(
                node, first, second, output, error);
        }
        break;
    case M3_OP_NEAREST_RESIZE1D:
        if (first->rank != 3U ||
            node->parameters.resize.output_length == 0U) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "cannot infer nearest resize shape");
        } else {
            output->rank = 3U;
            output->shape[0] = first->shape[0];
            output->shape[1] = first->shape[1];
            output->shape[2] = node->parameters.resize.output_length;
        }
        break;
    default:
        status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                              "cannot infer unsupported operation kind %d",
                              (int)node->kind);
        break;
    }
    return status;
}

m3_status m3_graph_add_inferred_value(
    m3_graph *graph, const m3_graph_node_desc *node,
    size_t output_index, m3_graph_value_role role, m3_dtype dtype,
    m3_graph_value_id *value, m3_error *error)
{
    m3_graph_value_desc output;
    m3_status status;

    if (graph == NULL || node == NULL || value == NULL ||
        (role != M3_GRAPH_TEMPORARY && role != M3_GRAPH_OUTPUT)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "inferred graph value request is invalid");
    }
    (void)memset(&output, 0, sizeof(output));
    output.role = role;
    output.dtype = dtype;
    status = m3_graph_infer_shape(
        graph, node, output_index, &output, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    return m3_graph_add_value(graph, &output, value, error);
}
