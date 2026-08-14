/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_METAL_INTERNAL_H
#define M3_BACKEND_METAL_INTERNAL_H

#include "m3_backend_internal.h"
#include "m3_backend_metal_helpers.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_METAL_PIPELINE_CAPACITY 128U
#define M3_METAL_PIPELINE_NAME_CAPACITY 64U

typedef struct {
    char name[M3_METAL_PIPELINE_NAME_CAPACITY];
    CFTypeRef state;
} m3_metal_pipeline_entry;

typedef struct {
    CFTypeRef device;
    CFTypeRef command_queue;
    CFTypeRef library;
    m3_metal_pipeline_entry pipelines[M3_METAL_PIPELINE_CAPACITY];
    size_t pipeline_count;
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
m3_status m3_metal_register_pipeline(m3_metal_context *context,
                                      const char *name,
                                      m3_error *error);
id<MTLComputePipelineState> m3_metal_find_pipeline(
    const m3_metal_context *context, const char *name);
m3_status m3_metal_dispatch_1d(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline, size_t work_count,
    m3_error *error);

NSString *m3_metal_source_basic(void);
m3_status m3_metal_prepare_basic(m3_metal_context *context,
                                  m3_error *error);
bool m3_metal_basic_writes_storage(const m3_command *command,
                                    const m3_storage *storage);
m3_status m3_metal_preflight_basic(const m3_command *commands,
                                    size_t command_index,
                                    m3_error *error);
m3_status m3_metal_encode_basic(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error);

NSString *m3_metal_source_dense(void);
m3_status m3_metal_prepare_dense(m3_metal_context *context,
                                  m3_error *error);
bool m3_metal_dense_writes_storage(const m3_command *command,
                                    const m3_storage *storage);
m3_status m3_metal_preflight_dense(const m3_command *commands,
                                    size_t command_index,
                                    m3_error *error);
m3_status m3_metal_encode_dense(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error);

NSString *m3_metal_source_attention(void);
m3_status m3_metal_prepare_attention(m3_metal_context *context,
                                      m3_error *error);
bool m3_metal_attention_writes_storage(const m3_command *command,
                                        const m3_storage *storage);
m3_status m3_metal_preflight_attention(const m3_command *commands,
                                        size_t command_index,
                                        m3_error *error);
m3_status m3_metal_encode_attention(const m3_metal_context *context,
                                     id<MTLComputeCommandEncoder> encoder,
                                     const m3_command *command,
                                     bool *handled, m3_error *error);

NSString *m3_metal_source_sampling(void);
m3_status m3_metal_prepare_sampling(m3_metal_context *context,
                                     m3_error *error);
bool m3_metal_sampling_writes_storage(const m3_command *command,
                                       const m3_storage *storage);
m3_status m3_metal_preflight_sampling(const m3_command *commands,
                                       size_t command_index,
                                       m3_error *error);
m3_status m3_metal_encode_sampling(const m3_metal_context *context,
                                    id<MTLComputeCommandEncoder> encoder,
                                    const m3_command *command,
                                    bool *handled, m3_error *error);

NSString *m3_metal_source_convolution(void);
m3_status m3_metal_prepare_convolution(m3_metal_context *context,
                                        m3_error *error);
bool m3_metal_convolution_writes_storage(const m3_command *command,
                                          const m3_storage *storage);
m3_status m3_metal_preflight_convolution(const m3_command *commands,
                                          size_t command_index,
                                          m3_error *error);
m3_status m3_metal_encode_convolution(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_command *command,
    bool *handled, m3_error *error);

#endif
