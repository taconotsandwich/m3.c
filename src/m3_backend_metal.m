/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    CFTypeRef device;
    CFTypeRef command_queue;
} m3_metal_context;

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

static void m3_metal_destroy(void *opaque_context)
{
    m3_metal_context *context = opaque_context;

    if (context->command_queue != NULL) {
        CFRelease(context->command_queue);
    }
    if (context->device != NULL) {
        CFRelease(context->device);
    }
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

static m3_status m3_backend_create_metal_in_pool(m3_backend **backend,
                                                 m3_error *error)
{
    static const m3_backend_vtable vtable = {
        m3_metal_destroy,
        m3_metal_allocate,
        m3_metal_free
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
