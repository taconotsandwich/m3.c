/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <math.h>

static m3_status m3_op_nn_view(const m3_backend *backend,
                               const m3_tensor_view *view, bool output,
                               const char *name, m3_error *error)
{
    m3_status status = m3_op_check_view(backend, view, output, name, error);

    if (status == M3_STATUS_OK && !m3_op_dtype_float(view->metadata.dtype)) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "%s must have a floating-point dtype", name);
    }
    return status;
}

static m3_status m3_op_validate_matmul(const m3_backend *backend,
                                       const m3_op_matmul *op,
                                       m3_error *error)
{
    m3_status status = m3_op_nn_view(backend, op->left, false,
                                    "matmul left input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->right, false,
                               "matmul right input", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->output, true,
                               "matmul output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->left->metadata.rank != 2U ||
        op->right->metadata.rank != 2U ||
        op->output->metadata.rank != 2U ||
        op->left->metadata.shape[1] != op->right->metadata.shape[0] ||
        op->output->metadata.shape[0] != op->left->metadata.shape[0] ||
        op->output->metadata.shape[1] != op->right->metadata.shape[1]) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "matmul shapes must be [M,K]x[K,N]->[M,N]");
    }
    status = m3_op_check_alias(op->output, op->left, false,
                               "matmul left input", error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->right, false,
                                   "matmul right input", error);
    }
    return status;
}

static m3_status m3_op_validate_linear(const m3_backend *backend,
                                       const m3_op_linear *op,
                                       m3_error *error)
{
    uint8_t axis;
    uint8_t last;
    m3_status status = m3_op_nn_view(backend, op->input, false,
                                    "linear input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->weight, false,
                               "linear weight", error);
    }
    if (status == M3_STATUS_OK && op->bias != NULL) {
        status = m3_op_nn_view(backend, op->bias, false,
                               "linear bias", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->output, true,
                               "linear output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->input->metadata.rank == 0U ||
        op->weight->metadata.rank != 2U ||
        op->output->metadata.rank != op->input->metadata.rank) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "linear ranks are invalid");
    }
    last = op->input->metadata.rank - 1U;
    for (axis = 0U; axis < last; ++axis) {
        if (op->input->metadata.shape[axis] !=
            op->output->metadata.shape[axis]) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "linear leading dimensions differ");
        }
    }
    if (op->input->metadata.shape[last] !=
            op->weight->metadata.shape[1] ||
        op->output->metadata.shape[last] !=
            op->weight->metadata.shape[0] ||
        op->weight->metadata.shape[0] == 0U ||
        op->weight->metadata.shape[1] == 0U ||
        (op->bias != NULL &&
         (op->bias->metadata.rank != 1U ||
          op->bias->metadata.shape[0] !=
              op->weight->metadata.shape[0]))) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "linear feature dimensions differ");
    }
    status = m3_op_check_alias(op->output, op->input, false,
                               "linear input", error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->weight, false,
                                   "linear weight", error);
    }
    if (status == M3_STATUS_OK && op->bias != NULL) {
        status = m3_op_check_alias(op->output, op->bias, false,
                                   "linear bias", error);
    }
    return status;
}

static m3_status m3_op_validate_norm_common(
    const m3_backend *backend, const m3_tensor_view *input,
    const m3_tensor_view *scale, const m3_tensor_view *bias,
    m3_tensor_view *output, float epsilon, const char *name,
    m3_error *error)
{
    uint64_t channels;
    m3_status status = m3_op_nn_view(backend, input, false, name, error);

    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, scale, false, "norm scale", error);
    }
    if (status == M3_STATUS_OK && bias != NULL) {
        status = m3_op_nn_view(backend, bias, false, "norm bias", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, output, true, "norm output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (input->metadata.rank == 0U || !m3_op_shape_equal(input, output) ||
        !isfinite(epsilon) || epsilon <= 0.0F) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "norm shape or epsilon is invalid");
    }
    channels = input->metadata.shape[input->metadata.rank - 1U];
    if (channels == 0U || scale->metadata.rank != 1U ||
        scale->metadata.shape[0] != channels ||
        (bias != NULL &&
         (bias->metadata.rank != 1U ||
          bias->metadata.shape[0] != channels))) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "norm channel shape is invalid");
    }
    status = m3_op_check_alias(output, input, true, name, error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(output, scale, false, "norm scale",
                                   error);
    }
    if (status == M3_STATUS_OK && bias != NULL) {
        status = m3_op_check_alias(output, bias, false, "norm bias", error);
    }
    return status;
}

static m3_status m3_op_validate_rope(const m3_backend *backend,
                                     const m3_op_rope *op,
                                     m3_error *error)
{
    uint64_t positions;
    uint64_t sequence;
    uint64_t rotary = op->rotary_dimension;
    m3_status status = m3_op_nn_view(backend, op->input, false,
                                    "RoPE input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->cosines, false,
                               "RoPE cosines", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->sines, false,
                               "RoPE sines", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->output, true,
                               "RoPE output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->input->metadata.rank != 4U ||
        !m3_op_shape_equal(op->input, op->output) ||
        op->cosines->metadata.rank != 2U ||
        !m3_op_shape_equal(op->cosines, op->sines) ||
        rotary == 0U || (rotary & 1U) != 0U ||
        rotary > op->input->metadata.shape[3] ||
        op->cosines->metadata.shape[1] != rotary / 2U ||
        (op->mode != M3_ROPE_HALF_SPLIT &&
         op->mode != M3_ROPE_INTERLEAVED)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RoPE shape, dimension, or mode is invalid");
    }
    positions = op->cosines->metadata.shape[0];
    sequence = op->input->metadata.shape[2];
    if (op->position_offset > positions ||
        sequence > positions - op->position_offset) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "RoPE positions exceed cosine table");
    }
    status = m3_op_check_alias(op->output, op->input, true, "RoPE input",
                               error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->cosines, false,
                                   "RoPE cosines", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->sines, false,
                                   "RoPE sines", error);
    }
    return status;
}

static bool m3_op_attention_mask_shape(const m3_tensor_view *mask,
                                       uint64_t batch, uint64_t heads,
                                       uint64_t queries, uint64_t keys)
{
    return mask->metadata.rank == 4U &&
           (mask->metadata.shape[0] == 1U ||
            mask->metadata.shape[0] == batch) &&
           (mask->metadata.shape[1] == 1U ||
            mask->metadata.shape[1] == heads) &&
           (mask->metadata.shape[2] == 1U ||
            mask->metadata.shape[2] == queries) &&
           mask->metadata.shape[3] == keys;
}

static m3_status m3_op_validate_attention(const m3_backend *backend,
                                          const m3_op_attention *op,
                                          m3_error *error)
{
    uint64_t batch;
    uint64_t heads;
    uint64_t kv_heads;
    uint64_t queries;
    uint64_t keys;
    uint64_t depth;
    m3_status status = m3_op_nn_view(backend, op->query, false,
                                    "attention query", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->key, false, "attention key",
                               error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->value, false,
                               "attention value", error);
    }
    if (status == M3_STATUS_OK && op->mask != NULL) {
        status = m3_op_nn_view(backend, op->mask, false, "attention mask",
                               error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_nn_view(backend, op->output, true,
                               "attention output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->query->metadata.rank != 4U || op->key->metadata.rank != 4U ||
        op->value->metadata.rank != 4U ||
        !m3_op_shape_equal(op->key, op->value) ||
        !m3_op_shape_equal(op->query, op->output) ||
        !isfinite(op->scale) || op->scale <= 0.0F) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "attention rank, shape, or scale is invalid");
    }
    batch = op->query->metadata.shape[0];
    heads = op->query->metadata.shape[1];
    queries = op->query->metadata.shape[2];
    depth = op->query->metadata.shape[3];
    kv_heads = op->key->metadata.shape[1];
    keys = op->key->metadata.shape[2];
    if (op->key->metadata.shape[0] != batch ||
        op->key->metadata.shape[3] != depth || heads == 0U ||
        kv_heads == 0U || keys == 0U || depth == 0U ||
        heads % kv_heads != 0U ||
        (op->mask != NULL &&
         !m3_op_attention_mask_shape(op->mask, batch, heads, queries,
                                     keys))) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "attention dimensions are incompatible");
    }
    status = m3_op_check_alias(op->output, op->query, false,
                               "attention query", error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->key, false,
                                   "attention key", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->value, false,
                                   "attention value", error);
    }
    if (status == M3_STATUS_OK && op->mask != NULL) {
        status = m3_op_check_alias(op->output, op->mask, false,
                                   "attention mask", error);
    }
    return status;
}

m3_status m3_op_validate_nn(const m3_backend *backend,
                            const m3_command *command,
                            bool *handled, m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_MATMUL:
        return m3_op_validate_matmul(backend, &command->descriptor.matmul,
                                     error);
    case M3_OP_LINEAR:
        return m3_op_validate_linear(backend, &command->descriptor.linear,
                                     error);
    case M3_OP_RMS_NORM:
        return m3_op_validate_norm_common(
            backend, command->descriptor.rms_norm.input,
            command->descriptor.rms_norm.scale, NULL,
            command->descriptor.rms_norm.output,
            command->descriptor.rms_norm.epsilon, "RMSNorm input", error);
    case M3_OP_LAYER_NORM:
        return m3_op_validate_norm_common(
            backend, command->descriptor.layer_norm.input,
            command->descriptor.layer_norm.scale,
            command->descriptor.layer_norm.bias,
            command->descriptor.layer_norm.output,
            command->descriptor.layer_norm.epsilon, "LayerNorm input",
            error);
    case M3_OP_ROPE:
        return m3_op_validate_rope(backend, &command->descriptor.rope,
                                   error);
    case M3_OP_ATTENTION:
        return m3_op_validate_attention(
            backend, &command->descriptor.attention, error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}
