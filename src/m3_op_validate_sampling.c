/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

static m3_status m3_op_sampling_view(const m3_backend *backend,
                                     const m3_tensor_view *view,
                                     bool output, const char *name,
                                     m3_error *error)
{
    return m3_op_check_view(backend, view, output, name, error);
}

static bool m3_op_top_k_shape(const m3_op_top_k *op)
{
    uint8_t rank = op->logits->metadata.rank;
    uint8_t axis;

    if (rank == 0U || op->values->metadata.rank != rank ||
        op->indices->metadata.rank != rank) {
        return false;
    }
    for (axis = 0U; axis + 1U < rank; ++axis) {
        if (op->values->metadata.shape[axis] !=
                op->logits->metadata.shape[axis] ||
            op->indices->metadata.shape[axis] !=
                op->logits->metadata.shape[axis]) {
            return false;
        }
    }
    return op->k != 0U &&
           op->k <= op->logits->metadata.shape[rank - 1U] &&
           op->values->metadata.shape[rank - 1U] == op->k &&
           op->indices->metadata.shape[rank - 1U] == op->k;
}

static m3_status m3_op_validate_top_k(const m3_backend *backend,
                                      const m3_op_top_k *op,
                                      m3_error *error)
{
    m3_status status = m3_op_sampling_view(backend, op->logits, false,
                                          "top-k logits", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_sampling_view(backend, op->values, true,
                                     "top-k values", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_sampling_view(backend, op->indices, true,
                                     "top-k indices", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->logits->metadata.rank == 0U ||
        !m3_op_dtype_float(op->logits->metadata.dtype) ||
        op->values->metadata.dtype != M3_DTYPE_F32 ||
        op->indices->metadata.dtype != M3_DTYPE_I32 ||
        op->logits->metadata.shape[op->logits->metadata.rank - 1U] >
            (uint64_t)INT32_MAX ||
        !m3_op_top_k_shape(op)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "top-k dtype, K, or shape is invalid");
    }
    status = m3_op_check_alias(op->values, op->logits, false,
                               "top-k values", error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->indices, op->logits, false,
                                   "top-k indices", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_check_pair_disjoint(op->values, op->indices,
                                           "top-k", error);
    }
    return status;
}

static bool m3_op_categorical_shape(const m3_op_categorical *op)
{
    uint8_t rank = op->probabilities->metadata.rank;
    uint8_t axis;

    if (rank == 0U || op->probabilities->metadata.shape[rank - 1U] == 0U ||
        op->probabilities->metadata.shape[rank - 1U] >
            (uint64_t)INT32_MAX ||
        op->uniforms->metadata.rank + 1U != rank ||
        op->output->metadata.rank + 1U != rank ||
        !m3_op_shape_equal(op->uniforms, op->output)) {
        return false;
    }
    for (axis = 0U; axis + 1U < rank; ++axis) {
        if (op->uniforms->metadata.shape[axis] !=
            op->probabilities->metadata.shape[axis]) {
            return false;
        }
    }
    return true;
}

static m3_status m3_op_validate_categorical(const m3_backend *backend,
                                            const m3_op_categorical *op,
                                            m3_error *error)
{
    m3_status status = m3_op_sampling_view(
        backend, op->probabilities, false, "categorical probabilities",
        error);

    if (status == M3_STATUS_OK) {
        status = m3_op_sampling_view(backend, op->uniforms, false,
                                     "categorical uniforms", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_sampling_view(backend, op->output, true,
                                     "categorical output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!m3_op_dtype_float(op->probabilities->metadata.dtype) ||
        !m3_op_dtype_float(op->uniforms->metadata.dtype) ||
        op->output->metadata.dtype != M3_DTYPE_I32 ||
        !m3_op_categorical_shape(op)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "categorical dtype or shape is invalid");
    }
    status = m3_op_check_alias(op->output, op->probabilities, false,
                               "categorical probabilities", error);
    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(op->output, op->uniforms, false,
                                   "categorical uniforms", error);
    }
    return status;
}

m3_status m3_op_validate_sampling(const m3_backend *backend,
                                  const m3_command *command,
                                  bool *handled, m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_TOP_K:
        return m3_op_validate_top_k(backend, &command->descriptor.top_k,
                                    error);
    case M3_OP_CATEGORICAL:
        return m3_op_validate_categorical(
            backend, &command->descriptor.categorical, error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}
