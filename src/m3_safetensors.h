/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_SAFETENSORS_H
#define M3_SAFETENSORS_H

#include "m3_error.h"
#include "m3_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_SAFETENSORS_MAX_HEADER_BYTES (64U * 1024U * 1024U)

typedef struct {
    char *name;
    m3_tensor_metadata tensor;
    uint64_t data_start;
    uint64_t data_end;
} m3_safetensors_tensor;

typedef struct {
    m3_safetensors_tensor *tensors;
    size_t tensor_count;
    size_t tensor_bytes;
    uint64_t data_section_offset;
    bool has_metadata;
} m3_safetensors_metadata;

void m3_safetensors_metadata_init(m3_safetensors_metadata *metadata);
void m3_safetensors_metadata_dispose(m3_safetensors_metadata *metadata);
m3_status m3_safetensors_inspect_file(const char *path,
                                      m3_safetensors_metadata *metadata,
                                      m3_error *error);

#endif
