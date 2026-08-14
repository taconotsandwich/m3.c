/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

static m3_status m3_op_require_float(const m3_tensor_view *view,
                                     const char *name, m3_error *error)
{
    if (!m3_op_dtype_float(view->metadata.dtype)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s must have a floating-point dtype", name);
    }
    return M3_STATUS_OK;
}

static m3_status m3_op_validate_unary_views(
    const m3_backend *backend, const m3_op_unary *op, m3_error *error)
{
    m3_status status = m3_op_check_view(backend, op->input, false,
                                       "unary input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_check_view(backend, op->output, true,
                                  "unary output", error);
    }
    if (status == M3_STATUS_OK && !m3_op_shape_equal(op->input, op->output)) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "unary input and output shapes differ");
    }
    return status;
}

static m3_status m3_op_validate_copy(const m3_backend *backend,
                                     const m3_op_unary *op,
                                     m3_error *error)
{
    m3_status status = m3_op_validate_unary_views(backend, op, error);

    if (status == M3_STATUS_OK &&
        op->input->metadata.dtype != op->output->metadata.dtype) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "copy dtypes differ");
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->input, true, "copy",
                                   error);
    }
    return status;
}

static m3_status m3_op_validate_cast(const m3_backend *backend,
                                     const m3_op_unary *op,
                                     m3_error *error)
{
    bool exact_allowed;
    m3_status status = m3_op_validate_unary_views(backend, op, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    exact_allowed = m3_dtype_size(op->input->metadata.dtype) ==
                    m3_dtype_size(op->output->metadata.dtype);
    return m3_op_check_alias(op->output, op->input, exact_allowed, "cast",
                             error);
}

static m3_status m3_op_validate_binary_views(
    const m3_backend *backend, const m3_op_binary *op, bool broadcast,
    bool exact_alias, m3_error *error)
{
    m3_status status = m3_op_check_view(backend, op->left, false,
                                       "binary left input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_check_view(backend, op->right, false,
                                  "binary right input", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_view(backend, op->output, true,
                                  "binary output", error);
    }
    if (status == M3_STATUS_OK &&
        (!m3_op_dtype_float(op->left->metadata.dtype) ||
         !m3_op_dtype_float(op->right->metadata.dtype) ||
         !m3_op_dtype_float(op->output->metadata.dtype))) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "binary arithmetic requires float tensors");
    }
    if (status == M3_STATUS_OK && broadcast) {
        status = m3_op_broadcast_validate(op->left, op->right, op->output,
                                          error);
    } else if (status == M3_STATUS_OK &&
               (!m3_op_shape_equal(op->left, op->right) ||
                !m3_op_shape_equal(op->left, op->output))) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "binary tensor shapes differ");
    }
    if (status == M3_STATUS_OK) {
        bool left_alias = exact_alias &&
                          m3_op_shape_equal(op->left, op->output);
        status = m3_op_check_alias(op->output, op->left, left_alias,
                                   "binary left input", error);
    }
    if (status == M3_STATUS_OK) {
        bool right_alias = exact_alias &&
                           m3_op_shape_equal(op->right, op->output);
        status = m3_op_check_alias(op->output, op->right, right_alias,
                                   "binary right input", error);
    }
    return status;
}

static m3_status m3_op_validate_embedding(const m3_backend *backend,
                                          const m3_op_embedding *op,
                                          m3_error *error)
{
    uint8_t axis;
    m3_status status = m3_op_check_view(backend, op->ids, false,
                                       "embedding IDs", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_check_view(backend, op->table, false,
                                  "embedding table", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_view(backend, op->output, true,
                                  "embedding output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->ids->metadata.dtype != M3_DTYPE_I32 ||
        op->table->metadata.rank != 2U ||
        !m3_op_dtype_float(op->table->metadata.dtype) ||
        !m3_op_dtype_float(op->output->metadata.dtype) ||
        op->output->metadata.rank != op->ids->metadata.rank + 1U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "embedding dtype or rank contract is invalid");
    }
    for (axis = 0U; axis < op->ids->metadata.rank; ++axis) {
        if (op->ids->metadata.shape[axis] !=
            op->output->metadata.shape[axis]) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "embedding leading shapes differ");
        }
    }
    if (op->table->metadata.shape[0] == 0U ||
        op->output->metadata.shape[op->ids->metadata.rank] !=
            op->table->metadata.shape[1]) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "embedding channel shape is invalid");
    }
    status = m3_op_check_alias(op->output, op->ids, false,
                               "embedding IDs", error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->table, false,
                                   "embedding table", error);
    }
    return status;
}

static m3_status m3_op_validate_softmax(const m3_backend *backend,
                                        const m3_op_unary *op,
                                        m3_error *error)
{
    m3_status status = m3_op_validate_unary_views(backend, op, error);

    if (status == M3_STATUS_OK) {
        status = m3_op_require_float(op->input, "softmax input", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_require_float(op->output, "softmax output", error);
    }
    if (status == M3_STATUS_OK &&
        (op->input->metadata.rank == 0U ||
         op->input->metadata.shape[op->input->metadata.rank - 1U] == 0U)) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "softmax last axis must be nonempty");
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->input, true, "softmax",
                                   error);
    }
    return status;
}

m3_status m3_op_validate_basic(const m3_backend *backend,
                               const m3_command *command,
                               bool *handled, m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_COPY:
        return m3_op_validate_copy(backend, &command->descriptor.copy,
                                   error);
    case M3_OP_CAST:
        return m3_op_validate_cast(backend, &command->descriptor.cast,
                                   error);
    case M3_OP_ADD:
        return m3_op_validate_binary_views(backend,
                                           &command->descriptor.add, true,
                                           true, error);
    case M3_OP_MUL:
        return m3_op_validate_binary_views(backend,
                                           &command->descriptor.mul, true,
                                           true, error);
    case M3_OP_EMBEDDING:
        return m3_op_validate_embedding(backend,
                                        &command->descriptor.embedding,
                                        error);
    case M3_OP_GATED_SILU:
        return m3_op_validate_binary_views(
            backend, &command->descriptor.gated_silu, false, true, error);
    case M3_OP_SOFTMAX:
        return m3_op_validate_softmax(backend,
                                      &command->descriptor.softmax, error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}
