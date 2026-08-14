/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tensor.h"

#include "m3_backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static bool m3_size_add(size_t left, size_t right, size_t *result)
{
    if (left > SIZE_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool m3_size_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0U && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static m3_status m3_tensor_require_output(m3_tensor_view *tensor,
                                          m3_error *error)
{
    if (tensor == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor output is null");
    }
    if (tensor->storage != NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor output already describes a view");
    }
    return M3_STATUS_OK;
}

static m3_status m3_tensor_canonical_strides(
    const m3_tensor_metadata *metadata, size_t *strides, m3_error *error)
{
    size_t stride = m3_dtype_size(metadata->dtype);
    uint8_t index = metadata->rank;

    (void)memset(strides, 0,
                 M3_TENSOR_MAX_RANK * sizeof(*strides));
    if (metadata->element_count == 0U) {
        return M3_STATUS_OK;
    }
    while (index > 0U) {
        size_t next;

        --index;
        strides[index] = stride;
        if (!m3_size_multiply(stride, (size_t)metadata->shape[index],
                              &next)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tensor stride overflows at axis %u",
                                (unsigned int)index);
        }
        stride = next;
    }
    return M3_STATUS_OK;
}

static m3_status m3_tensor_validate_metadata(
    const m3_tensor_view *tensor, m3_tensor_metadata *checked,
    m3_error *error)
{
    m3_status status;

    if (tensor == NULL || tensor->storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor view is not initialized");
    }
    status = m3_tensor_metadata_init(checked, tensor->metadata.dtype,
                                     tensor->metadata.rank,
                                     tensor->metadata.shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (checked->element_count != tensor->metadata.element_count ||
        checked->byte_count != tensor->metadata.byte_count) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor metadata counts are inconsistent");
    }
    return M3_STATUS_OK;
}

static m3_status m3_tensor_validate(const m3_tensor_view *tensor,
                                    m3_error *error)
{
    m3_tensor_metadata checked = {0};
    size_t maximum;
    size_t element_size;
    size_t storage_size;
    const void *storage_data;
    uint8_t index;
    m3_status status;

    status = m3_tensor_validate_metadata(tensor, &checked, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    storage_size = m3_storage_size(tensor->storage);
    if (tensor->byte_offset > storage_size) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "tensor byte offset exceeds storage");
    }
    element_size = m3_dtype_size(checked.dtype);
    if (tensor->byte_offset % element_size != 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor byte offset is misaligned");
    }
    storage_data = m3_storage_data(tensor->storage);
    if (storage_data != NULL &&
        (uintptr_t)storage_data % element_size != 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor storage pointer is misaligned");
    }
    for (index = 0U; index < checked.rank; ++index) {
        if (tensor->byte_strides[index] % element_size != 0U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "tensor stride %u is misaligned",
                                (unsigned int)index);
        }
    }
    if (checked.element_count == 0U) {
        return M3_STATUS_OK;
    }
    maximum = tensor->byte_offset;
    for (index = 0U; index < checked.rank; ++index) {
        size_t axis_offset;

        if (!m3_size_multiply((size_t)(checked.shape[index] - 1U),
                              tensor->byte_strides[index], &axis_offset) ||
            !m3_size_add(maximum, axis_offset, &maximum)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tensor view bounds overflow");
        }
    }
    if (!m3_size_add(maximum, element_size, &maximum)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "tensor view end overflows");
    }
    if (maximum > storage_size) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "tensor view exceeds storage");
    }
    return M3_STATUS_OK;
}

void m3_tensor_view_init(m3_tensor_view *tensor)
{
    if (tensor != NULL) {
        (void)memset(tensor, 0, sizeof(*tensor));
    }
}

m3_status m3_tensor_view_strided(m3_tensor_view *tensor,
                                 m3_storage *storage,
                                 m3_dtype dtype, uint8_t rank,
                                 const uint64_t *shape,
                                 const size_t *byte_strides,
                                 size_t byte_offset, m3_error *error)
{
    m3_tensor_metadata metadata;
    m3_status status = m3_tensor_require_output(tensor, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (storage == NULL || (rank != 0U && byte_strides == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor storage and strides are required");
    }
    status = m3_tensor_metadata_init(&metadata, dtype, rank, shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    tensor->metadata = metadata;
    tensor->byte_offset = byte_offset;
    tensor->storage = storage;
    if (rank != 0U) {
        (void)memcpy(tensor->byte_strides, byte_strides,
                     (size_t)rank * sizeof(*byte_strides));
    }
    status = m3_tensor_validate(tensor, error);
    if (status != M3_STATUS_OK) {
        m3_tensor_view_init(tensor);
        return status;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_tensor_view_contiguous(m3_tensor_view *tensor,
                                    m3_storage *storage,
                                    m3_dtype dtype, uint8_t rank,
                                    const uint64_t *shape,
                                    size_t byte_offset, m3_error *error)
{
    m3_tensor_metadata metadata;
    size_t strides[M3_TENSOR_MAX_RANK];
    m3_status status = m3_tensor_metadata_init(&metadata, dtype, rank, shape,
                                               error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_canonical_strides(&metadata, strides, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    return m3_tensor_view_strided(tensor, storage, dtype, rank, shape,
                                  strides, byte_offset, error);
}

bool m3_tensor_is_contiguous(const m3_tensor_view *tensor)
{
    size_t expected;
    uint8_t index;

    if (tensor == NULL || tensor->storage == NULL) {
        return false;
    }
    if (tensor->metadata.element_count == 0U) {
        return true;
    }
    expected = m3_dtype_size(tensor->metadata.dtype);
    index = tensor->metadata.rank;
    while (index > 0U) {
        --index;
        if (tensor->metadata.shape[index] > 1U &&
            tensor->byte_strides[index] != expected) {
            return false;
        }
        if (!m3_size_multiply(expected,
                              (size_t)tensor->metadata.shape[index],
                              &expected)) {
            return false;
        }
    }
    return true;
}

m3_status m3_tensor_const_data(const m3_tensor_view *tensor,
                               const void **data,
                               m3_error *error)
{
    const uint8_t *base;
    m3_status status;

    if (data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor data output is null");
    }
    *data = NULL;
    status = m3_tensor_validate(tensor, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    base = m3_storage_data(tensor->storage);
    if (base != NULL) {
        *data = base + tensor->byte_offset;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_tensor_data(m3_tensor_view *tensor, void **data,
                         m3_error *error)
{
    const void *constant_data = NULL;
    m3_status status;

    if (data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor data output is null");
    }
    *data = NULL;
    status = m3_tensor_const_data(tensor, &constant_data, error);
    if (status == M3_STATUS_OK) {
        *data = (void *)constant_data;
    }
    return status;
}

m3_status m3_tensor_element_offset(const m3_tensor_view *tensor,
                                   const uint64_t *indices,
                                   size_t *byte_offset, m3_error *error)
{
    size_t offset;
    uint8_t index;
    m3_status status;

    if (byte_offset == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor element offset output is null");
    }
    *byte_offset = 0U;
    status = m3_tensor_validate(tensor, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (tensor->metadata.rank != 0U && indices == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor indices are null");
    }
    offset = tensor->byte_offset;
    for (index = 0U; index < tensor->metadata.rank; ++index) {
        size_t axis_offset;

        if (indices[index] >= tensor->metadata.shape[index]) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "tensor index exceeds axis %u",
                                (unsigned int)index);
        }
        if (!m3_size_multiply((size_t)indices[index],
                              tensor->byte_strides[index], &axis_offset) ||
            !m3_size_add(offset, axis_offset, &offset)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tensor element offset overflows");
        }
    }
    *byte_offset = offset;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_tensor_begin_view(const m3_tensor_view *source,
                                      m3_tensor_view *view,
                                      m3_error *error)
{
    m3_status status = m3_tensor_require_output(view, error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_validate(source, error);
    }
    return status;
}

static m3_status m3_tensor_finish_view(m3_tensor_view *view,
                                      m3_error *error)
{
    m3_status status = m3_tensor_validate(view, error);

    if (status != M3_STATUS_OK) {
        m3_tensor_view_init(view);
        return status;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_tensor_slice(const m3_tensor_view *source, uint8_t axis,
                          uint64_t start, uint64_t length,
                          m3_tensor_view *view,
                          m3_error *error)
{
    uint64_t shape[M3_TENSOR_MAX_RANK];
    size_t offset_delta;
    uint8_t index;
    m3_status status = m3_tensor_begin_view(source, view, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (axis >= source->metadata.rank ||
        start > source->metadata.shape[axis] ||
        length > source->metadata.shape[axis] - start) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "tensor slice exceeds axis bounds");
    }
    if (!m3_size_multiply((size_t)start, source->byte_strides[axis],
                          &offset_delta) ||
        !m3_size_add(source->byte_offset, offset_delta, &offset_delta)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "tensor slice offset overflows");
    }
    *view = *source;
    view->byte_offset = offset_delta;
    for (index = 0U; index < source->metadata.rank; ++index) {
        shape[index] = source->metadata.shape[index];
    }
    shape[axis] = length;
    status = m3_tensor_metadata_init(&view->metadata,
                                     source->metadata.dtype,
                                     source->metadata.rank,
                                     shape, error);
    if (status != M3_STATUS_OK) {
        m3_tensor_view_init(view);
        return status;
    }
    return m3_tensor_finish_view(view, error);
}

m3_status m3_tensor_reshape(const m3_tensor_view *source, uint8_t rank,
                            const uint64_t *shape, m3_tensor_view *view,
                            m3_error *error)
{
    m3_tensor_metadata metadata;
    m3_status status = m3_tensor_begin_view(source, view, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!m3_tensor_is_contiguous(source)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "cannot reshape a non-contiguous tensor");
    }
    status = m3_tensor_metadata_init(&metadata, source->metadata.dtype,
                                     rank, shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (metadata.element_count != source->metadata.element_count) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "reshape changes tensor element count");
    }
    status = m3_tensor_canonical_strides(&metadata, view->byte_strides,
                                         error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    view->metadata = metadata;
    view->byte_offset = source->byte_offset;
    view->storage = source->storage;
    return m3_tensor_finish_view(view, error);
}

m3_status m3_tensor_permute(const m3_tensor_view *source,
                            const uint8_t *permutation,
                            m3_tensor_view *view,
                            m3_error *error)
{
    bool seen[M3_TENSOR_MAX_RANK] = {false};
    m3_tensor_metadata metadata;
    uint64_t shape[M3_TENSOR_MAX_RANK];
    uint8_t index;
    m3_status status = m3_tensor_begin_view(source, view, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (source->metadata.rank != 0U && permutation == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor permutation is null");
    }
    for (index = 0U; index < source->metadata.rank; ++index) {
        uint8_t axis = permutation[index];

        if (axis >= source->metadata.rank || seen[axis]) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "tensor permutation axis %u is invalid",
                                (unsigned int)axis);
        }
        seen[axis] = true;
        shape[index] = source->metadata.shape[axis];
    }
    status = m3_tensor_metadata_init(&metadata, source->metadata.dtype,
                                     source->metadata.rank,
                                     shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    *view = *source;
    view->metadata = metadata;
    for (index = 0U; index < source->metadata.rank; ++index) {
        view->byte_strides[index] =
            source->byte_strides[permutation[index]];
    }
    return m3_tensor_finish_view(view, error);
}
