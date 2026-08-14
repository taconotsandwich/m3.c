/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_METAL_INTERNAL_H
#define M3_BACKEND_METAL_INTERNAL_H

#include "m3_backend_internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    CFTypeRef device;
    CFTypeRef command_queue;
    CFTypeRef library;
    CFTypeRef copy_pipeline;
    CFTypeRef cast_pipeline;
} m3_metal_context;

typedef struct {
    uint64_t element_count;
    uint64_t input_byte_offset;
    uint64_t output_byte_offset;
    uint64_t input_shape[M3_TENSOR_MAX_RANK];
    uint64_t input_strides[M3_TENSOR_MAX_RANK];
    uint32_t rank;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t element_size;
} m3_metal_parameters;

m3_status m3_metal_compile_library(m3_metal_context *context,
                                    m3_error *error);
m3_status m3_metal_preflight_basic(const m3_command *commands,
                                    size_t command_index,
                                    m3_error *error);
m3_status m3_metal_encode_basic(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error);
m3_status m3_metal_encode_dense(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error);
m3_status m3_metal_encode_attention(const m3_metal_context *context,
                                     id<MTLComputeCommandEncoder> encoder,
                                     const m3_command *command,
                                     bool *handled, m3_error *error);
m3_status m3_metal_encode_convolution(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_command *command,
    bool *handled, m3_error *error);

#endif
