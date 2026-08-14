/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_METAL_HELPERS_H
#define M3_BACKEND_METAL_HELPERS_H

#include "m3_op.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t element_count;
    uint64_t byte_offset;
    uint64_t shape[M3_TENSOR_MAX_RANK];
    uint64_t byte_strides[M3_TENSOR_MAX_RANK];
    uint32_t rank;
    uint32_t dtype;
} m3_metal_view_parameters;

void m3_metal_view_parameters_init(
    m3_metal_view_parameters *parameters,
    const m3_tensor_view *view);

bool m3_metal_command_writes_storage(const m3_command *command,
                                     const m3_storage *storage);
bool m3_metal_has_prior_writer(const m3_command *commands,
                               size_t command_index,
                               const m3_storage *storage);

#endif
