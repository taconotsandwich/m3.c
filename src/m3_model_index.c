/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_model_index.h"

#include "m3_file.h"
#include "m3_json.h"

#include <stdlib.h>
#include <string.h>

void m3_model_index_init(m3_model_index *index)
{
    if (index == NULL) {
        return;
    }
    index->entries = NULL;
    index->entry_count = 0U;
    index->has_total_size = false;
    index->total_size = 0U;
}

void m3_model_index_dispose(m3_model_index *index)
{
    size_t entry_index;

    if (index == NULL) {
        return;
    }
    for (entry_index = 0U; entry_index < index->entry_count; ++entry_index) {
        free(index->entries[entry_index].tensor_name);
        free(index->entries[entry_index].shard_name);
    }
    free(index->entries);
    m3_model_index_init(index);
}

const char *m3_model_index_shard(const m3_model_index *index,
                                 const char *tensor_name)
{
    size_t entry_index;

    if (index == NULL || tensor_name == NULL) {
        return NULL;
    }
    for (entry_index = 0U; entry_index < index->entry_count; ++entry_index) {
        if (strcmp(index->entries[entry_index].tensor_name, tensor_name) == 0) {
            return index->entries[entry_index].shard_name;
        }
    }
    return NULL;
}

static bool m3_model_index_valid_shard_name(const char *name)
{
    static const char suffix[] = ".safetensors";
    size_t length;

    if (name == NULL || name[0] == '\0' || strchr(name, '/') != NULL ||
        strchr(name, '\\') != NULL || strcmp(name, ".") == 0 ||
        strcmp(name, "..") == 0) {
        return false;
    }
    length = strlen(name);
    return length > sizeof(suffix) - 1U &&
           strcmp(name + length - (sizeof(suffix) - 1U), suffix) == 0;
}

static m3_status m3_model_index_append(m3_model_index *index,
                                       char *tensor_name, char *shard_name,
                                       size_t *capacity, m3_error *error)
{
    m3_model_index_entry *new_entries;
    size_t new_capacity;

    if (tensor_name[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model index contains an empty tensor name");
    }
    if (m3_model_index_shard(index, tensor_name) != NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model index duplicates tensor '%s'", tensor_name);
    }
    if (!m3_model_index_valid_shard_name(shard_name)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model index has invalid shard '%s'", shard_name);
    }
    if (index->entry_count < *capacity) {
        index->entries[index->entry_count].tensor_name = tensor_name;
        index->entries[index->entry_count].shard_name = shard_name;
        index->entry_count += 1U;
        return M3_STATUS_OK;
    }
    new_capacity = *capacity == 0U ? 64U : *capacity * 2U;
    if (new_capacity < *capacity ||
        new_capacity > SIZE_MAX / sizeof(*new_entries)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "model index capacity overflows");
    }
    new_entries = realloc(index->entries,
                          new_capacity * sizeof(*new_entries));
    if (new_entries == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot grow model index");
    }
    index->entries = new_entries;
    *capacity = new_capacity;
    index->entries[index->entry_count].tensor_name = tensor_name;
    index->entries[index->entry_count].shard_name = shard_name;
    index->entry_count += 1U;
    return M3_STATUS_OK;
}

static m3_status m3_model_index_parse_metadata(m3_json_reader *reader,
                                               m3_model_index *index,
                                               m3_error *error)
{
    bool first = true;

    if (m3_json_expect(reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)'}')) {
        char *name = NULL;
        m3_status status;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_read_string(reader, &name, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (m3_json_expect(reader, (uint8_t)':', error) != M3_STATUS_OK) {
            free(name);
            return M3_STATUS_INVALID_FORMAT;
        }
        if (strcmp(name, "total_size") == 0 && !index->has_total_size) {
            status = m3_json_read_uint64(reader, &index->total_size, error);
            index->has_total_size = status == M3_STATUS_OK;
        } else if (strcmp(name, "total_size") == 0) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "model index duplicates total_size");
        } else {
            status = m3_json_skip_value(reader, error);
        }
        free(name);
        if (status != M3_STATUS_OK) {
            return status;
        }
        first = false;
    }
    return m3_json_expect(reader, (uint8_t)'}', error);
}

static m3_status m3_model_index_parse_weight_map(m3_json_reader *reader,
                                                 m3_model_index *index,
                                                 m3_error *error)
{
    size_t capacity = index->entry_count;
    bool first = true;

    if (m3_json_expect(reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)'}')) {
        char *tensor_name = NULL;
        char *shard_name = NULL;
        m3_status status;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_read_string(reader, &tensor_name, error);
        if (status == M3_STATUS_OK) {
            status = m3_json_expect(reader, (uint8_t)':', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_read_string(reader, &shard_name, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_model_index_append(index, tensor_name, shard_name,
                                           &capacity, error);
        }
        if (status != M3_STATUS_OK) {
            free(tensor_name);
            free(shard_name);
            return status;
        }
        first = false;
    }
    return m3_json_expect(reader, (uint8_t)'}', error);
}

static m3_status m3_model_index_parse(const uint8_t *data, size_t size,
                                      m3_model_index *index,
                                      m3_error *error)
{
    m3_json_reader reader;
    bool has_metadata = false;
    bool has_weight_map = false;
    bool first = true;
    m3_status status;

    m3_json_reader_init(&reader, data, size);
    if (m3_json_expect(&reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(&reader, (uint8_t)'}')) {
        char *name = NULL;

        if (!first &&
            m3_json_expect(&reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_read_string(&reader, &name, error);
        if (status == M3_STATUS_OK) {
            status = m3_json_expect(&reader, (uint8_t)':', error);
        }
        if (status == M3_STATUS_OK && strcmp(name, "metadata") == 0 &&
            !has_metadata) {
            status = m3_model_index_parse_metadata(&reader, index, error);
            has_metadata = status == M3_STATUS_OK;
        } else if (status == M3_STATUS_OK &&
                   strcmp(name, "weight_map") == 0 && !has_weight_map) {
            status = m3_model_index_parse_weight_map(&reader, index, error);
            has_weight_map = status == M3_STATUS_OK;
        } else if (status == M3_STATUS_OK) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "unknown or duplicate model index field '%s'",
                                  name);
        }
        free(name);
        if (status != M3_STATUS_OK) {
            return status;
        }
        first = false;
    }
    if (m3_json_expect(&reader, (uint8_t)'}', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    if (!has_weight_map || index->entry_count == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model index lacks a nonempty weight_map");
    }
    return m3_json_finish(&reader, error);
}

m3_status m3_model_index_read(const char *path, m3_model_index *index,
                              m3_error *error)
{
    uint8_t *data = NULL;
    size_t size = 0U;
    m3_model_index parsed;
    m3_status status;

    if (path == NULL || index == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "model index argument is null");
    }
    m3_model_index_init(&parsed);
    status = m3_file_read_bounded(path, M3_MODEL_JSON_MAX_BYTES, &data,
                                  &size, error);
    if (status == M3_STATUS_OK) {
        status = m3_model_index_parse(data, size, &parsed, error);
    }
    free(data);
    if (status != M3_STATUS_OK) {
        m3_model_index_dispose(&parsed);
        return status;
    }
    m3_model_index_dispose(index);
    *index = parsed;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
