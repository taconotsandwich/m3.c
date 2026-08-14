/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void m3_metal_copy_name(char *destination, size_t capacity,
                               NSString *source)
{
    const char *fallback = "Unnamed Metal device";
    size_t index;

    if (source == nil ||
        ![source getCString:destination
                   maxLength:capacity
                    encoding:NSUTF8StringEncoding]) {
        (void)snprintf(destination, capacity, "%s", fallback);
    }
    for (index = 0U; destination[index] != '\0'; ++index) {
        unsigned char value = (unsigned char)destination[index];

        if (value < 0x20U || value == 0x7fU) {
            destination[index] = '?';
        }
    }
}

static void m3_metal_release(CFTypeRef *object)
{
    if (*object != NULL) {
        CFRelease(*object);
        *object = NULL;
    }
}

static void m3_metal_destroy(void *opaque_context)
{
    m3_metal_context *context = opaque_context;

    if (context == NULL) {
        return;
    }
    while (context->pipeline_count != 0U) {
        m3_metal_pipeline_entry *entry;

        --context->pipeline_count;
        entry = &context->pipelines[context->pipeline_count];
        m3_metal_release(&entry->state);
        entry->name[0] = '\0';
    }
    m3_metal_release(&context->library);
    m3_metal_release(&context->command_queue);
    m3_metal_release(&context->device);
    free(context);
}

static m3_status m3_metal_allocate(void *opaque_context, size_t byte_count,
                                   size_t alignment, void **handle,
                                   void **data, m3_error *error)
{
    m3_metal_context *context = opaque_context;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLBuffer> buffer;

    (void)alignment;
    *handle = NULL;
    *data = NULL;
    if (byte_count == 0U) {
        return M3_STATUS_OK;
    }
    buffer = [device newBufferWithLength:byte_count
                                 options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "Metal could not allocate %zu storage bytes",
                            byte_count);
    }
    *handle = (__bridge_retained void *)buffer;
    *data = buffer.contents;
    return M3_STATUS_OK;
}

static void m3_metal_free(void *context, void *handle, void *data)
{
    (void)context;
    (void)data;
    if (handle != NULL) {
        CFRelease((CFTypeRef)handle);
    }
}

bool m3_metal_command_writes_storage(const m3_command *command,
                                      const m3_storage *storage)
{
    return m3_metal_basic_writes_storage(command, storage) ||
           m3_metal_dense_writes_storage(command, storage) ||
           m3_metal_attention_writes_storage(command, storage) ||
           m3_metal_sampling_writes_storage(command, storage) ||
           m3_metal_convolution_writes_storage(command, storage);
}

static m3_status m3_metal_preflight_one(const m3_command *commands,
                                         size_t command_index,
                                         m3_error *error)
{
    m3_status status = m3_metal_preflight_basic(
        commands, command_index, error);

    if (status == M3_STATUS_OK) {
        status = m3_metal_preflight_dense(
            commands, command_index, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_metal_preflight_attention(
            commands, command_index, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_metal_preflight_sampling(
            commands, command_index, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_metal_preflight_convolution(
            commands, command_index, error);
    }
    return status;
}

static m3_status m3_metal_encode_one(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_command *command,
    m3_error *error)
{
    bool handled = false;
    m3_status status = m3_metal_encode_basic(
        context, encoder, command, &handled, error);

    if (status == M3_STATUS_OK && !handled) {
        status = m3_metal_encode_dense(context, encoder, command, &handled,
                                       error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_metal_encode_attention(
            context, encoder, command, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_metal_encode_sampling(
            context, encoder, command, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_metal_encode_convolution(
            context, encoder, command, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                              "Metal does not implement operation %d",
                              (int)command->kind);
    }
    return status;
}

static m3_status m3_metal_command_error(
    id<MTLCommandBuffer> command_buffer, m3_error *error)
{
    NSError *metal_error = command_buffer.error;
    long code = metal_error == nil ? 0L : (long)metal_error.code;

    return m3_error_set(error, M3_STATUS_INTERNAL,
                        "Metal command buffer failed (status %u, error %ld)",
                        (unsigned int)command_buffer.status, code);
}

static m3_status m3_metal_execute_in_pool(
    m3_metal_context *context, const m3_command *commands,
    size_t command_count, m3_error *error)
{
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)context->command_queue;
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    size_t index;
    m3_status status;

    command_buffer = [queue commandBuffer];
    if (command_buffer == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal could not create a command buffer");
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal could not create a compute encoder");
    }
    for (index = 0U; index < command_count; ++index) {
        status = m3_metal_encode_one(context, encoder, &commands[index],
                                     error);
        if (status == M3_STATUS_OK) {
            status = m3_metal_preflight_one(commands, index, error);
        }
        if (status != M3_STATUS_OK) {
            [encoder endEncoding];
            return status;
        }
    }
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return m3_metal_command_error(command_buffer, error);
    }
    return M3_STATUS_OK;
}

static m3_status m3_metal_execute(void *opaque_context,
                                  const m3_command *commands,
                                  size_t command_count,
                                  m3_scratch_arena *scratch,
                                  m3_error *error)
{
    m3_metal_context *context = opaque_context;

    (void)scratch;
    @autoreleasepool {
        return m3_metal_execute_in_pool(context, commands, command_count,
                                        error);
    }
}

static m3_status m3_backend_create_metal_in_pool(m3_backend **backend,
                                                 m3_error *error)
{
    static const m3_backend_vtable vtable = {
        m3_metal_destroy,
        m3_metal_allocate,
        m3_metal_free,
        m3_metal_execute
    };
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    m3_metal_context *context;
    m3_backend_info info;
    m3_status status;

    if (backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Metal backend output is null");
    }
    *backend = NULL;
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "Metal has no default device");
    }
    command_queue = [device newCommandQueue];
    if (command_queue == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal could not create a command queue");
    }
    context = calloc(1U, sizeof(*context));
    if (context == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Metal backend context");
    }
    context->device = CFBridgingRetain(device);
    context->command_queue = CFBridgingRetain(command_queue);
    status = m3_metal_compile_library(context, error);
    if (status != M3_STATUS_OK) {
        m3_metal_destroy(context);
        return status;
    }
    (void)memset(&info, 0, sizeof(info));
    m3_metal_copy_name(info.name, sizeof(info.name), device.name);
    info.kind = M3_BACKEND_METAL;
    info.unified_memory = device.hasUnifiedMemory ? true : false;
    info.recommended_working_set_bytes =
        (uint64_t)device.recommendedMaxWorkingSetSize;
    info.maximum_storage_bytes = (uint64_t)device.maxBufferLength;
    status = m3_backend_create_internal(&vtable, context, &info, backend,
                                        error);
    if (status != M3_STATUS_OK) {
        m3_metal_destroy(context);
    }
    return status;
}

m3_status m3_backend_create_metal(m3_backend **backend, m3_error *error)
{
    @autoreleasepool {
        return m3_backend_create_metal_in_pool(backend, error);
    }
}
