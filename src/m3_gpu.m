/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct m3_gpu_buffer {
    m3_gpu *owner;
    CFTypeRef resource;
    m3_tensor_metadata metadata;
    struct m3_gpu_buffer *previous;
    struct m3_gpu_buffer *next;
};

struct m3_gpu {
    CFTypeRef device;
    CFTypeRef command_queue;
    m3_gpu_device_info info;
    m3_gpu_allocation_stats stats;
    m3_gpu_buffer *buffers;
};

static void m3_gpu_copy_device_name(char *destination, size_t capacity,
                                    NSString *source)
{
    const char *fallback = "Unnamed Metal device";
    size_t index;

    if (destination == NULL || capacity == 0U) {
        return;
    }
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

static m3_status m3_gpu_validate_metadata(
    const m3_tensor_metadata *metadata, m3_tensor_metadata *validated,
    m3_error *error)
{
    m3_status status;

    if (metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU tensor metadata is null");
    }

    status = m3_tensor_metadata_init(validated, metadata->dtype,
                                     metadata->rank, metadata->shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (validated->element_count != metadata->element_count ||
        validated->byte_count != metadata->byte_count) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "GPU tensor metadata counts do not match its shape and dtype");
    }
    if (validated->byte_count == 0U) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Metal buffers cannot represent an empty tensor");
    }
    return M3_STATUS_OK;
}

static m3_status m3_gpu_validate_range(size_t buffer_size, size_t offset,
                                        size_t byte_count, const void *memory,
                                        const char *operation,
                                        m3_error *error)
{
    if (offset > buffer_size || byte_count > buffer_size - offset) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "%s range at offset %zu with length %zu exceeds "
                            "buffer size %zu",
                            operation, offset, byte_count, buffer_size);
    }
    if (byte_count != 0U && memory == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s memory is null for a nonempty range",
                            operation);
    }
    return M3_STATUS_OK;
}

static void m3_gpu_link_buffer(m3_gpu *gpu, m3_gpu_buffer *buffer)
{
    buffer->owner = gpu;
    buffer->next = gpu->buffers;
    if (gpu->buffers != NULL) {
        gpu->buffers->previous = buffer;
    }
    gpu->buffers = buffer;
}

static void m3_gpu_unlink_buffer(m3_gpu_buffer *buffer)
{
    m3_gpu *gpu = buffer->owner;

    if (buffer->previous != NULL) {
        buffer->previous->next = buffer->next;
    } else {
        gpu->buffers = buffer->next;
    }
    if (buffer->next != NULL) {
        buffer->next->previous = buffer->previous;
    }

    gpu->stats.live_allocated_bytes -= buffer->metadata.byte_count;
    gpu->stats.live_buffer_count -= 1U;
    buffer->owner = NULL;
    buffer->previous = NULL;
    buffer->next = NULL;
}

m3_status m3_gpu_create(m3_gpu **gpu_output, m3_error *error)
{
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    m3_gpu *gpu;

    if (gpu_output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU output is null");
    }
    *gpu_output = NULL;

    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return m3_error_set(
            error, M3_STATUS_UNSUPPORTED,
            "Metal unavailable: MTLCreateSystemDefaultDevice returned no "
            "device");
    }
    command_queue = [device newCommandQueue];
    if (command_queue == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal device could not create a command queue");
    }

    gpu = calloc(1U, sizeof(*gpu));
    if (gpu == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "could not allocate Metal runtime state");
    }

    gpu->device = CFBridgingRetain(device);
    gpu->command_queue = CFBridgingRetain(command_queue);
    m3_gpu_copy_device_name(gpu->info.name, sizeof(gpu->info.name),
                            device.name);
    gpu->info.unified_memory = device.hasUnifiedMemory ? true : false;
    gpu->info.recommended_working_set_bytes =
        (uint64_t)device.recommendedMaxWorkingSetSize;
    gpu->info.maximum_buffer_bytes = (uint64_t)device.maxBufferLength;

    *gpu_output = gpu;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_gpu_free(m3_gpu *gpu)
{
    if (gpu == NULL) {
        return;
    }

    while (gpu->buffers != NULL) {
        m3_gpu_buffer_free(gpu->buffers);
    }
    if (gpu->command_queue != NULL) {
        CFRelease(gpu->command_queue);
    }
    if (gpu->device != NULL) {
        CFRelease(gpu->device);
    }
    free(gpu);
}

m3_status m3_gpu_get_device_info(const m3_gpu *gpu,
                                  m3_gpu_device_info *info,
                                  m3_error *error)
{
    if (gpu == NULL || info == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU and device info output are required");
    }

    *info = gpu->info;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_gpu_get_allocation_stats(const m3_gpu *gpu,
                                      m3_gpu_allocation_stats *stats,
                                      m3_error *error)
{
    if (gpu == NULL || stats == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU and allocation stats output are required");
    }

    *stats = gpu->stats;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_gpu_buffer_create(m3_gpu *gpu,
                                const m3_tensor_metadata *metadata,
                                m3_gpu_buffer **buffer_output,
                                m3_error *error)
{
    m3_tensor_metadata validated;
    id<MTLDevice> device;
    id<MTLBuffer> resource;
    m3_gpu_buffer *buffer;
    m3_status status;

    if (buffer_output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU buffer output is null");
    }
    *buffer_output = NULL;
    if (gpu == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU runtime is null");
    }

    status = m3_gpu_validate_metadata(metadata, &validated, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if ((uint64_t)validated.byte_count > gpu->info.maximum_buffer_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "tensor requires %zu bytes but Metal permits at "
                            "most %llu bytes per buffer",
                            validated.byte_count,
                            (unsigned long long)
                                gpu->info.maximum_buffer_bytes);
    }
    if (gpu->stats.live_allocated_bytes >
            SIZE_MAX - validated.byte_count ||
        gpu->stats.live_buffer_count == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Metal allocation statistics overflow");
    }

    buffer = calloc(1U, sizeof(*buffer));
    if (buffer == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "could not allocate GPU buffer state");
    }

    device = (__bridge id<MTLDevice>)gpu->device;
    resource = [device newBufferWithLength:validated.byte_count
                                   options:MTLResourceStorageModeShared];
    if (resource == nil) {
        free(buffer);
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "Metal could not allocate a %zu-byte shared "
                            "buffer",
                            validated.byte_count);
    }

    buffer->resource = CFBridgingRetain(resource);
    buffer->metadata = validated;
    m3_gpu_link_buffer(gpu, buffer);
    gpu->stats.live_allocated_bytes += validated.byte_count;
    gpu->stats.live_buffer_count += 1U;
    if (gpu->stats.live_allocated_bytes >
        gpu->stats.peak_allocated_bytes) {
        gpu->stats.peak_allocated_bytes =
            gpu->stats.live_allocated_bytes;
    }
    if (gpu->stats.live_buffer_count > gpu->stats.peak_buffer_count) {
        gpu->stats.peak_buffer_count = gpu->stats.live_buffer_count;
    }

    *buffer_output = buffer;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_gpu_buffer_free(m3_gpu_buffer *buffer)
{
    if (buffer == NULL) {
        return;
    }

    if (buffer->owner != NULL) {
        m3_gpu_unlink_buffer(buffer);
    }
    if (buffer->resource != NULL) {
        CFRelease(buffer->resource);
    }
    free(buffer);
}

const m3_tensor_metadata *m3_gpu_buffer_metadata(
    const m3_gpu_buffer *buffer)
{
    if (buffer == NULL) {
        return NULL;
    }
    return &buffer->metadata;
}

m3_status m3_gpu_buffer_upload(m3_gpu_buffer *buffer, size_t offset,
                                const void *source, size_t byte_count,
                                m3_error *error)
{
    id<MTLBuffer> resource;
    m3_status status;

    if (buffer == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU upload buffer is null");
    }
    status = m3_gpu_validate_range(buffer->metadata.byte_count, offset,
                                   byte_count, source, "GPU upload", error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (byte_count != 0U) {
        resource = (__bridge id<MTLBuffer>)buffer->resource;
        (void)memcpy((uint8_t *)resource.contents + offset, source,
                     byte_count);
    }

    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_gpu_buffer_download(const m3_gpu_buffer *buffer, size_t offset,
                                  void *destination, size_t byte_count,
                                  m3_error *error)
{
    id<MTLBuffer> resource;
    m3_status status;

    if (buffer == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "GPU download buffer is null");
    }
    status = m3_gpu_validate_range(buffer->metadata.byte_count, offset,
                                   byte_count, destination, "GPU download",
                                   error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (byte_count != 0U) {
        resource = (__bridge id<MTLBuffer>)buffer->resource;
        (void)memcpy(destination,
                     (const uint8_t *)resource.contents + offset,
                     byte_count);
    }

    m3_error_reset(error);
    return M3_STATUS_OK;
}
