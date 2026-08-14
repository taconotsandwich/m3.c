/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TENSOR_H
#define M3_TENSOR_H

#include "m3_error.h"

#include <stddef.h>
#include <stdint.h>

#define M3_TENSOR_MAX_RANK 8

typedef enum {
    M3_DTYPE_F32 = 0,
    M3_DTYPE_F16,
    M3_DTYPE_BF16
} m3_dtype;

typedef struct {
    m3_dtype dtype;
    uint8_t rank;
    uint64_t shape[M3_TENSOR_MAX_RANK];
    size_t element_count;
    size_t byte_count;
} m3_tensor_metadata;

const char *m3_dtype_name(m3_dtype dtype);
size_t m3_dtype_size(m3_dtype dtype);
m3_status m3_tensor_metadata_init(m3_tensor_metadata *metadata,
                                  m3_dtype dtype, uint8_t rank,
                                  const uint64_t *shape, m3_error *error);

#endif
