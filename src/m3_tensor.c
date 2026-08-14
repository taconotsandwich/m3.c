/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tensor.h"

#include <stdint.h>
#include <string.h>

const char *m3_dtype_name(m3_dtype dtype)
{
    switch (dtype) {
    case M3_DTYPE_F32:
        return "F32";
    case M3_DTYPE_F16:
        return "F16";
    case M3_DTYPE_BF16:
        return "BF16";
    }

    return "unknown";
}

size_t m3_dtype_size(m3_dtype dtype)
{
    switch (dtype) {
    case M3_DTYPE_F32:
        return 4U;
    case M3_DTYPE_F16:
    case M3_DTYPE_BF16:
        return 2U;
    }

    return 0U;
}

m3_status m3_tensor_metadata_init(m3_tensor_metadata *metadata,
                                  m3_dtype dtype, uint8_t rank,
                                  const uint64_t *shape, m3_error *error)
{
    size_t element_count = 1U;
    size_t byte_count;
    size_t element_size;
    uint8_t index;

    if (metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor metadata output is null");
    }
    (void)memset(metadata, 0, sizeof(*metadata));

    element_size = m3_dtype_size(dtype);
    if (element_size == 0U) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "unsupported tensor dtype value %d", (int)dtype);
    }
    if (rank > M3_TENSOR_MAX_RANK) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "tensor rank %u exceeds maximum rank %u",
                            (unsigned int)rank,
                            (unsigned int)M3_TENSOR_MAX_RANK);
    }
    if (rank != 0U && shape == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tensor shape is null for rank %u",
                            (unsigned int)rank);
    }

    for (index = 0U; index < rank; ++index) {
        uint64_t dimension = shape[index];

        if (dimension > (uint64_t)SIZE_MAX) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tensor dimension %u does not fit size_t",
                                (unsigned int)index);
        }
        if (dimension != 0U &&
            element_count > SIZE_MAX / (size_t)dimension) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tensor element count overflows at dimension %u",
                                (unsigned int)index);
        }
        element_count *= (size_t)dimension;
    }

    if (element_count > SIZE_MAX / element_size) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "tensor byte count overflows size_t");
    }
    byte_count = element_count * element_size;

    metadata->dtype = dtype;
    metadata->rank = rank;
    for (index = 0U; index < rank; ++index) {
        metadata->shape[index] = shape[index];
    }
    metadata->element_count = element_count;
    metadata->byte_count = byte_count;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
