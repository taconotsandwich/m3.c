/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <stdint.h>

typedef struct {
    size_t begin;
    size_t end;
    bool empty;
} m3_op_span;

static bool m3_op_size_add(size_t left, size_t right, size_t *result)
{
    if (left > SIZE_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool m3_op_size_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0U && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

bool m3_op_dtype_float(m3_dtype dtype)
{
    return dtype == M3_DTYPE_F32 || dtype == M3_DTYPE_F16 ||
           dtype == M3_DTYPE_BF16;
}

bool m3_op_shape_equal(const m3_tensor_view *left,
                       const m3_tensor_view *right)
{
    uint8_t axis;

    if (left->metadata.rank != right->metadata.rank) {
        return false;
    }
    for (axis = 0U; axis < left->metadata.rank; ++axis) {
        if (left->metadata.shape[axis] != right->metadata.shape[axis]) {
            return false;
        }
    }
    return true;
}

m3_status m3_op_check_view(const m3_backend *backend,
                           const m3_tensor_view *view, bool output,
                           const char *name, m3_error *error)
{
    const void *data;
    m3_status status;

    if (backend == NULL || view == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s view is required", name);
    }
    if (m3_storage_backend(view->storage) != backend) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s view belongs to another backend", name);
    }
    status = m3_tensor_const_data(view, &data, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (output && !m3_tensor_is_contiguous(view)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s output must be contiguous", name);
    }
    return M3_STATUS_OK;
}

static m3_status m3_op_view_span(const m3_tensor_view *view,
                                 m3_op_span *span, m3_error *error)
{
    size_t end = view->byte_offset;
    uint8_t axis;

    span->begin = view->byte_offset;
    span->end = view->byte_offset;
    span->empty = view->metadata.element_count == 0U;
    if (span->empty) {
        return M3_STATUS_OK;
    }
    for (axis = 0U; axis < view->metadata.rank; ++axis) {
        size_t delta;

        if (!m3_op_size_multiply(
                (size_t)(view->metadata.shape[axis] - 1U),
                view->byte_strides[axis], &delta) ||
            !m3_op_size_add(end, delta, &end)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tensor storage span overflows");
        }
    }
    if (!m3_op_size_add(end, m3_dtype_size(view->metadata.dtype), &end)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "tensor storage span end overflows");
    }
    span->end = end;
    return M3_STATUS_OK;
}

static bool m3_op_exact_view(const m3_tensor_view *left,
                             const m3_tensor_view *right)
{
    uint8_t axis;

    if (left->storage != right->storage ||
        left->byte_offset != right->byte_offset ||
        m3_dtype_size(left->metadata.dtype) !=
            m3_dtype_size(right->metadata.dtype) ||
        !m3_op_shape_equal(left, right)) {
        return false;
    }
    for (axis = 0U; axis < left->metadata.rank; ++axis) {
        if (left->metadata.shape[axis] > 1U &&
            left->byte_strides[axis] != right->byte_strides[axis]) {
            return false;
        }
    }
    return true;
}

static m3_status m3_op_overlap(const m3_tensor_view *left,
                               const m3_tensor_view *right, bool *overlap,
                               m3_error *error)
{
    m3_op_span left_span;
    m3_op_span right_span;
    m3_status status;

    *overlap = false;
    if (left->storage != right->storage) {
        return M3_STATUS_OK;
    }
    status = m3_op_view_span(left, &left_span, error);
    if (status == M3_STATUS_OK) {
        status = m3_op_view_span(right, &right_span, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    *overlap = !left_span.empty && !right_span.empty &&
               left_span.begin < right_span.end &&
               right_span.begin < left_span.end;
    return M3_STATUS_OK;
}

m3_status m3_op_check_alias(const m3_tensor_view *output,
                            const m3_tensor_view *input,
                            bool exact_allowed, const char *name,
                            m3_error *error)
{
    bool overlap;
    m3_status status = m3_op_overlap(output, input, &overlap, error);

    if (status != M3_STATUS_OK || !overlap) {
        return status;
    }
    if (exact_allowed && m3_op_exact_view(output, input)) {
        return M3_STATUS_OK;
    }
    return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                        "%s has forbidden partial or in-place overlap",
                        name);
}

m3_status m3_op_check_pair_disjoint(const m3_tensor_view *left,
                                    const m3_tensor_view *right,
                                    const char *name, m3_error *error)
{
    bool overlap;
    m3_status status = m3_op_overlap(left, right, &overlap, error);

    if (status != M3_STATUS_OK || !overlap) {
        return status;
    }
    return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                        "%s outputs overlap", name);
}

m3_status m3_op_broadcast_validate(const m3_tensor_view *left,
                                   const m3_tensor_view *right,
                                   const m3_tensor_view *output,
                                   m3_error *error)
{
    uint8_t rank = left->metadata.rank > right->metadata.rank
                       ? left->metadata.rank
                       : right->metadata.rank;
    uint8_t axis;

    if (output->metadata.rank != rank) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "broadcast output rank is incorrect");
    }
    for (axis = 0U; axis < rank; ++axis) {
        int left_axis = (int)axis - (int)(rank - left->metadata.rank);
        int right_axis = (int)axis - (int)(rank - right->metadata.rank);
        uint64_t left_size = left_axis < 0
                                 ? 1U
                                 : left->metadata.shape[left_axis];
        uint64_t right_size = right_axis < 0
                                  ? 1U
                                  : right->metadata.shape[right_axis];
        uint64_t expected;

        if (left_size != right_size && left_size != 1U &&
            right_size != 1U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "broadcast dimensions disagree at axis %u",
                                (unsigned int)axis);
        }
        if (left_size == right_size) {
            expected = left_size;
        } else if (left_size == 1U) {
            expected = right_size;
        } else {
            expected = left_size;
        }
        if (output->metadata.shape[axis] != expected) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "broadcast output shape is incorrect");
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_command_validate(const m3_backend *backend,
                              const m3_command *command,
                              m3_error *error)
{
    bool handled = false;
    m3_status status;

    if (backend == NULL || command == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "backend and command are required");
    }
    status = m3_op_validate_basic(backend, command, &handled, error);
    if (status == M3_STATUS_OK && !handled) {
        status = m3_op_validate_nn(backend, command, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_op_validate_sampling(backend, command, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_op_validate_convolution(backend, command, &handled,
                                            error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                              "unsupported operation kind %d",
                              (int)command->kind);
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_commands_scratch_bytes(const m3_backend *backend,
                                    const m3_command *commands,
                                    size_t command_count,
                                    size_t *scratch_bytes,
                                    m3_error *error)
{
    size_t maximum = 0U;
    size_t index;

    if (backend == NULL || scratch_bytes == NULL ||
        (command_count != 0U && commands == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "command list and scratch output are required");
    }
    *scratch_bytes = 0U;
    for (index = 0U; index < command_count; ++index) {
        size_t current = 0U;
        m3_status status = m3_command_validate(backend, &commands[index],
                                               error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        status = m3_op_command_scratch(&commands[index], &current, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (current > maximum) {
            maximum = current;
        }
    }
    *scratch_bytes = maximum;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_op_command_scratch(const m3_command *command,
                                size_t *byte_count, m3_error *error)
{
    size_t elements = 0U;
    size_t element_size = 0U;
    size_t payload;
    size_t padding;

    if (command->kind == M3_OP_ATTENTION) {
        elements = (size_t)command->descriptor.attention.key->metadata.shape[2];
        element_size = sizeof(float);
    } else if (command->kind == M3_OP_TOP_K) {
        elements = (size_t)command->descriptor.top_k.k;
        element_size = sizeof(m3_top_pair);
    } else {
        *byte_count = 0U;
        return M3_STATUS_OK;
    }
    if (!m3_op_size_multiply(elements, element_size, &payload)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "operation scratch size overflows");
    }
    padding = _Alignof(m3_top_pair) > _Alignof(float)
                  ? _Alignof(m3_top_pair) - 1U
                  : _Alignof(float) - 1U;
    if (!m3_op_size_add(payload, padding, byte_count)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "aligned operation scratch size overflows");
    }
    return M3_STATUS_OK;
}
