/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TENSOR_H
#define M3_TENSOR_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_TENSOR_MAX_RANK 8

typedef enum {
    M3_DTYPE_F32 = 0,
    M3_DTYPE_F16,
    M3_DTYPE_BF16,
    M3_DTYPE_I32
} m3_dtype;

typedef struct {
    m3_dtype dtype;
    uint8_t rank;
    uint64_t shape[M3_TENSOR_MAX_RANK];
    size_t element_count;
    size_t byte_count;
} m3_tensor_metadata;

typedef struct m3_storage m3_storage;

typedef struct {
    m3_tensor_metadata metadata;
    size_t byte_offset;
    size_t byte_strides[M3_TENSOR_MAX_RANK];
    m3_storage *storage;
} m3_tensor_view;

const char *m3_dtype_name(m3_dtype dtype);
size_t m3_dtype_size(m3_dtype dtype);
m3_status m3_tensor_metadata_init(m3_tensor_metadata *metadata,
                                  m3_dtype dtype, uint8_t rank,
                                  const uint64_t *shape, m3_error *error);

void m3_tensor_view_init(m3_tensor_view *tensor);

/* Views borrow storage and become invalid as soon as that storage is freed. */
m3_status m3_tensor_view_contiguous(m3_tensor_view *tensor,
                                    m3_storage *storage,
                                    m3_dtype dtype, uint8_t rank,
                                    const uint64_t *shape,
                                    size_t byte_offset, m3_error *error);
m3_status m3_tensor_view_strided(m3_tensor_view *tensor,
                                 m3_storage *storage,
                                 m3_dtype dtype, uint8_t rank,
                                 const uint64_t *shape,
                                 const size_t *byte_strides,
                                 size_t byte_offset, m3_error *error);
bool m3_tensor_is_contiguous(const m3_tensor_view *tensor);
m3_status m3_tensor_data(m3_tensor_view *tensor, void **data,
                         m3_error *error);
m3_status m3_tensor_const_data(const m3_tensor_view *tensor,
                               const void **data,
                               m3_error *error);
m3_status m3_tensor_element_offset(const m3_tensor_view *tensor,
                                   const uint64_t *indices,
                                   size_t *byte_offset, m3_error *error);
m3_status m3_tensor_slice(const m3_tensor_view *source, uint8_t axis,
                          uint64_t start, uint64_t length,
                          m3_tensor_view *view,
                          m3_error *error);
m3_status m3_tensor_reshape(const m3_tensor_view *source, uint8_t rank,
                            const uint64_t *shape, m3_tensor_view *view,
                            m3_error *error);
m3_status m3_tensor_permute(const m3_tensor_view *source,
                            const uint8_t *permutation,
                            m3_tensor_view *view,
                            m3_error *error);

#endif
