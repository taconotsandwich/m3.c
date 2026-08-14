/* SPDX-License-Identifier: GPL-2.0-only */

#include "music3_sparse_fixture.h"

#include "model_loader_fixture.h"
#include "m3_music3_schema.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} m3_sparse_buffer;

typedef struct {
    m3_sparse_buffer *buffer;
    size_t shard_index;
    size_t shard_count;
    size_t ordinal;
    size_t emitted;
    uint64_t payload_bytes;
} m3_sparse_header_context;

typedef struct {
    m3_sparse_buffer *buffer;
    m3_component_id id;
    size_t shard_count;
    size_t ordinal;
    size_t emitted;
} m3_sparse_index_context;

static void m3_sparse_buffer_dispose(m3_sparse_buffer *buffer)
{
    free(buffer->data);
    buffer->data = NULL;
    buffer->size = 0U;
    buffer->capacity = 0U;
}

static bool m3_sparse_buffer_reserve(m3_sparse_buffer *buffer,
                                     size_t additional)
{
    size_t required;

    if (additional > SIZE_MAX - buffer->size - 1U) {
        return false;
    }
    required = buffer->size + additional + 1U;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity == 0U ? 4096U : buffer->capacity;
        char *data;

        while (capacity < required) {
            if (capacity > SIZE_MAX / 2U) {
                return false;
            }
            capacity *= 2U;
        }
        data = realloc(buffer->data, capacity);
        if (data == NULL) {
            return false;
        }
        buffer->data = data;
        buffer->capacity = capacity;
    }
    return true;
}

static bool m3_sparse_buffer_append(m3_sparse_buffer *buffer,
                                    const char *text)
{
    size_t length = strlen(text);

    if (!m3_sparse_buffer_reserve(buffer, length)) {
        return false;
    }
    (void)memcpy(buffer->data + buffer->size, text, length + 1U);
    buffer->size += length;
    return true;
}

static bool m3_sparse_buffer_format(m3_sparse_buffer *buffer,
                                    const char *format, ...)
{
    va_list arguments;
    va_list copy;
    int count;

    va_start(arguments, format);
    va_copy(copy, arguments);
    count = vsnprintf(NULL, 0U, format, copy);
    va_end(copy);
    if (count < 0 || !m3_sparse_buffer_reserve(buffer, (size_t)count)) {
        va_end(arguments);
        return false;
    }
    count = vsnprintf(buffer->data + buffer->size,
                      buffer->capacity - buffer->size, format, arguments);
    va_end(arguments);
    if (count < 0) {
        return false;
    }
    buffer->size += (size_t)count;
    return true;
}

static bool m3_sparse_shard_name(m3_component_id id, size_t shard_index,
                                 char name[80])
{
    int count;

    if (id == M3_COMPONENT_LANGUAGE_MODEL) {
        count = snprintf(name, 80U, "model-%05zu-of-00004.safetensors",
                         shard_index + 1U);
    } else if (id == M3_COMPONENT_TRANSFORMER) {
        count = snprintf(
            name, 80U,
            "diffusion_pytorch_model-%05zu-of-00002.safetensors",
            shard_index + 1U);
    } else {
        count = snprintf(name, 80U, "diffusion_pytorch_model.safetensors");
    }
    return count >= 0 && count < 80;
}

static m3_status m3_sparse_header_visit(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_sparse_header_context *header = context;
    uint64_t start;
    uint64_t end;
    uint8_t dimension;

    (void)error;
    if (header->ordinal++ % header->shard_count != header->shard_index) {
        return M3_STATUS_OK;
    }
    start = header->payload_bytes;
    if ((uint64_t)tensor->byte_count > UINT64_MAX - start) {
        return M3_STATUS_OVERFLOW;
    }
    end = start + (uint64_t)tensor->byte_count;
    if (!m3_sparse_buffer_format(
            header->buffer,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
            header->emitted == 0U ? "" : ",", name,
            m3_dtype_name(tensor->dtype))) {
        return M3_STATUS_OUT_OF_MEMORY;
    }
    for (dimension = 0U; dimension < tensor->rank; ++dimension) {
        if (!m3_sparse_buffer_format(
                header->buffer, "%s%" PRIu64,
                dimension == 0U ? "" : ",", tensor->shape[dimension])) {
            return M3_STATUS_OUT_OF_MEMORY;
        }
    }
    if (!m3_sparse_buffer_format(
            header->buffer,
            "],\"data_offsets\":[%" PRIu64 ",%" PRIu64 "]}", start,
            end)) {
        return M3_STATUS_OUT_OF_MEMORY;
    }
    header->payload_bytes = end;
    header->emitted += 1U;
    return M3_STATUS_OK;
}

static m3_status m3_sparse_index_visit(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_sparse_index_context *index = context;
    char shard_name[80];
    size_t shard_index = index->ordinal++ % index->shard_count;

    (void)tensor;
    (void)error;
    if (!m3_sparse_shard_name(index->id, shard_index, shard_name) ||
        !m3_sparse_buffer_format(
            index->buffer, "%s\"%s\":\"%s\"",
            index->emitted == 0U ? "" : ",", name, shard_name)) {
        return M3_STATUS_OUT_OF_MEMORY;
    }
    index->emitted += 1U;
    return M3_STATUS_OK;
}

static bool m3_sparse_write_all(int descriptor, const uint8_t *data,
                                size_t size)
{
    size_t total = 0U;

    while (total < size) {
        ssize_t count = write(descriptor, data + total, size - total);

        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        total += (size_t)count;
    }
    return true;
}

static bool m3_sparse_write_file(const char *path,
                                 const m3_sparse_buffer *header,
                                 uint64_t payload_bytes,
                                 uint64_t *logical_bytes,
                                 uint64_t *physical_bytes)
{
    uint8_t length_bytes[8];
    uint64_t header_length = (uint64_t)header->size;
    uint64_t logical_size;
    struct stat file_stat;
    unsigned int index;
    int descriptor;
    bool success;

    if (header_length > UINT64_MAX - payload_bytes - 8U) {
        return false;
    }
    logical_size = 8U + header_length + payload_bytes;
    for (index = 0U; index < 8U; ++index) {
        length_bytes[index] =
            (uint8_t)((header_length >> (index * 8U)) & 0xffU);
    }
    descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (descriptor < 0) {
        return false;
    }
    success = m3_sparse_write_all(descriptor, length_bytes,
                                  sizeof(length_bytes)) &&
              m3_sparse_write_all(descriptor,
                                  (const uint8_t *)header->data,
                                  header->size) &&
              logical_size <= (uint64_t)INT64_MAX &&
              ftruncate(descriptor, (off_t)logical_size) == 0 &&
              fstat(descriptor, &file_stat) == 0;
    if (close(descriptor) != 0) {
        success = false;
    }
    if (!success || (uint64_t)file_stat.st_blocks > UINT64_MAX / 512U) {
        return false;
    }
    if (logical_size > UINT64_MAX - *logical_bytes ||
        (uint64_t)file_stat.st_blocks * 512U >
            UINT64_MAX - *physical_bytes) {
        return false;
    }
    *logical_bytes += logical_size;
    *physical_bytes += (uint64_t)file_stat.st_blocks * 512U;
    return true;
}

static bool m3_sparse_create_shard(const char *component,
                                   m3_component_id id, size_t shard_index,
                                   size_t shard_count,
                                   uint64_t *logical_bytes,
                                   uint64_t *physical_bytes)
{
    m3_sparse_buffer header = {NULL, 0U, 0U};
    m3_sparse_header_context context = {&header, shard_index, shard_count,
                                        0U, 0U, 0U};
    char name[80];
    char path[M3_TEST_PATH_CAPACITY];
    m3_error error;
    bool success = m3_sparse_buffer_append(&header, "{") &&
                   m3_music3_schema_visit(
                       id, m3_sparse_header_visit, &context, &error) ==
                       M3_STATUS_OK &&
                   context.emitted != 0U &&
                   m3_sparse_buffer_append(&header, "}") &&
                   m3_sparse_shard_name(id, shard_index, name) &&
                   m3_loader_test_path(path, component, name) &&
                   m3_sparse_write_file(path, &header, context.payload_bytes,
                                        logical_bytes, physical_bytes);

    m3_sparse_buffer_dispose(&header);
    return success;
}

static bool m3_sparse_create_index(const char *component,
                                   m3_component_id id, size_t shard_count)
{
    m3_sparse_buffer index = {NULL, 0U, 0U};
    m3_sparse_index_context context = {&index, id, shard_count, 0U, 0U};
    m3_music3_schema_summary summary;
    char path[M3_TEST_PATH_CAPACITY];
    const char *name = id == M3_COMPONENT_LANGUAGE_MODEL
                           ? "model.safetensors.index.json"
                           : "diffusion_pytorch_model.safetensors.index.json";
    m3_error error;
    bool success =
        m3_music3_schema_expected_summary(id, &summary, &error) ==
            M3_STATUS_OK &&
        m3_sparse_buffer_format(
            &index,
            "{\"metadata\":{\"total_size\":%" PRIu64
            "},\"weight_map\":{",
            summary.payload_bytes) &&
        m3_music3_schema_visit(id, m3_sparse_index_visit, &context, &error) ==
            M3_STATUS_OK &&
        m3_sparse_buffer_append(&index, "}}") &&
        m3_loader_test_path(path, component, name) &&
        m3_loader_test_write_file(path, (const uint8_t *)index.data,
                                  index.size);

    m3_sparse_buffer_dispose(&index);
    return success;
}

static bool m3_sparse_create_weight_component(
    const char *root, m3_component_id id, uint64_t *logical_bytes,
    uint64_t *physical_bytes)
{
    char component[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    size_t shard_count = id == M3_COMPONENT_LANGUAGE_MODEL
                             ? 4U
                             : (id == M3_COMPONENT_TRANSFORMER ? 2U : 1U);
    size_t shard_index;

    if (!m3_loader_test_path(component, root, m3_component_directory(id)) ||
        mkdir(component, 0700) != 0 ||
        !m3_loader_test_path(path, component, "config.json") ||
        !m3_test_music3_write_config(path, id)) {
        return false;
    }
    for (shard_index = 0U; shard_index < shard_count; ++shard_index) {
        if (!m3_sparse_create_shard(component, id, shard_index, shard_count,
                                    logical_bytes, physical_bytes)) {
            return false;
        }
    }
    return shard_count == 1U ||
           m3_sparse_create_index(component, id, shard_count);
}

static bool m3_sparse_create_resources(const char *root)
{
    char component[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];

    if (!m3_loader_test_path(component, root, "tokenizer") ||
        mkdir(component, 0700) != 0 ||
        !m3_loader_test_path(path, component, "tokenizer.json") ||
        !m3_loader_test_write_json(path, "{}") ||
        !m3_loader_test_path(path, component, "tokenizer_config.json") ||
        !m3_loader_test_write_json(path, "{}") ||
        !m3_loader_test_path(path, component, "chat_template.jinja") ||
        !m3_loader_test_write_file(path, (const uint8_t *)"{% generation %}",
                                   strlen("{% generation %}")) ||
        !m3_loader_test_path(component, root, "scheduler") ||
        mkdir(component, 0700) != 0 ||
        !m3_loader_test_path(path, component, "scheduler_config.json") ||
        !m3_loader_test_write_json(path, "{}")) {
        return false;
    }
    return true;
}

static bool m3_sparse_create_ignored_roots(const char *root)
{
    char path[M3_TEST_PATH_CAPACITY];

    return m3_loader_test_path(path, root, "qwen_7B") &&
           mkdir(path, 0700) == 0 &&
           m3_loader_test_path(path, root, "dav.pth") &&
           m3_loader_test_write_file(path, (const uint8_t *)"legacy", 6U);
}

bool m3_test_music3_create_sparse_layout(char root[256],
                                         uint64_t *logical_bytes,
                                         uint64_t *physical_bytes)
{
    char path[M3_TEST_PATH_CAPACITY];
    m3_component_id id;

    if (logical_bytes == NULL || physical_bytes == NULL) {
        return false;
    }
    *logical_bytes = 0U;
    *physical_bytes = 0U;
    if (!m3_loader_test_make_root(root) ||
        !m3_loader_test_path(path, root, "modular_model_index.json") ||
        !m3_loader_test_write_json(path, m3_loader_test_manifest)) {
        return false;
    }
    for (id = M3_COMPONENT_LANGUAGE_MODEL; id <= M3_COMPONENT_VOCODER;
         id = (m3_component_id)((int)id + 1)) {
        if (!m3_sparse_create_weight_component(
                root, id, logical_bytes, physical_bytes)) {
            return false;
        }
    }
    return m3_sparse_create_resources(root) &&
           m3_sparse_create_ignored_roots(root);
}
