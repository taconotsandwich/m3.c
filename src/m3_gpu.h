/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_GPU_H
#define M3_GPU_H

#include "m3_error.h"
#include "m3_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_GPU_DEVICE_NAME_CAPACITY 128U

typedef struct m3_gpu m3_gpu;
typedef struct m3_gpu_buffer m3_gpu_buffer;

typedef struct {
    char name[M3_GPU_DEVICE_NAME_CAPACITY];
    bool unified_memory;
    uint64_t recommended_working_set_bytes;
    uint64_t maximum_buffer_bytes;
} m3_gpu_device_info;

typedef struct {
    size_t live_allocated_bytes;
    size_t peak_allocated_bytes;
    size_t live_buffer_count;
    size_t peak_buffer_count;
} m3_gpu_allocation_stats;

m3_status m3_gpu_create(m3_gpu **gpu, m3_error *error);
void m3_gpu_free(m3_gpu *gpu);

m3_status m3_gpu_get_device_info(const m3_gpu *gpu,
                                  m3_gpu_device_info *info,
                                  m3_error *error);
m3_status m3_gpu_get_allocation_stats(const m3_gpu *gpu,
                                      m3_gpu_allocation_stats *stats,
                                      m3_error *error);

m3_status m3_gpu_buffer_create(m3_gpu *gpu,
                                const m3_tensor_metadata *metadata,
                                m3_gpu_buffer **buffer, m3_error *error);
void m3_gpu_buffer_free(m3_gpu_buffer *buffer);
const m3_tensor_metadata *m3_gpu_buffer_metadata(
    const m3_gpu_buffer *buffer);

m3_status m3_gpu_buffer_upload(m3_gpu_buffer *buffer, size_t offset,
                                const void *source, size_t byte_count,
                                m3_error *error);
m3_status m3_gpu_buffer_download(const m3_gpu_buffer *buffer, size_t offset,
                                  void *destination, size_t byte_count,
                                  m3_error *error);

#endif
