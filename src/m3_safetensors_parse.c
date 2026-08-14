/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_safetensors_internal.h"

#include "m3_json.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} m3_name_set;

static void m3_name_set_dispose(m3_name_set *set)
{
    size_t index;

    for (index = 0U; index < set->count; ++index) {
        free(set->items[index]);
    }
    free(set->items);
    set->items = NULL;
    set->count = 0U;
    set->capacity = 0U;
}

static bool m3_name_set_contains(const m3_name_set *set, const char *name)
{
    size_t index;

    for (index = 0U; index < set->count; ++index) {
        if (strcmp(set->items[index], name) == 0) {
            return true;
        }
    }
    return false;
}

static m3_status m3_name_set_take(m3_name_set *set, char *name,
                                  m3_error *error)
{
    if (m3_name_set_contains(set, name)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "duplicate JSON name '%s'", name);
    }
    if (set->count == set->capacity) {
        size_t new_capacity = set->capacity == 0U ? 8U : set->capacity * 2U;
        char **new_items;

        if (new_capacity < set->capacity ||
            new_capacity > SIZE_MAX / sizeof(*new_items)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "JSON name set capacity overflows");
        }
        new_items = realloc(set->items,
                            new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot grow JSON name set");
        }
        set->items = new_items;
        set->capacity = new_capacity;
    }
    set->items[set->count++] = name;
    return M3_STATUS_OK;
}

static m3_status m3_safetensors_parse_dtype(const char *name,
                                            m3_dtype *dtype,
                                            m3_error *error)
{
    if (strcmp(name, "F32") == 0) {
        *dtype = M3_DTYPE_F32;
    } else if (strcmp(name, "F16") == 0) {
        *dtype = M3_DTYPE_F16;
    } else if (strcmp(name, "BF16") == 0) {
        *dtype = M3_DTYPE_BF16;
    } else {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "unsupported Safetensors dtype '%s'", name);
    }
    return M3_STATUS_OK;
}

static m3_status m3_safetensors_parse_shape(m3_json_reader *reader,
                                            uint64_t shape[], uint8_t *rank,
                                            m3_error *error)
{
    bool first = true;
    uint8_t count = 0U;

    if (m3_json_expect(reader, (uint8_t)'[', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)']')) {
        m3_status status;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        if (count >= M3_TENSOR_MAX_RANK) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "Safetensors rank exceeds %u",
                                (unsigned int)M3_TENSOR_MAX_RANK);
        }
        status = m3_json_read_uint64(reader, &shape[count], error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        count += 1U;
        first = false;
    }
    if (m3_json_expect(reader, (uint8_t)']', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    *rank = count;
    return M3_STATUS_OK;
}

static m3_status m3_safetensors_parse_offsets(m3_json_reader *reader,
                                              uint64_t offsets[2],
                                              m3_error *error)
{
    if (m3_json_expect(reader, (uint8_t)'[', error) != M3_STATUS_OK ||
        m3_json_read_uint64(reader, &offsets[0], error) != M3_STATUS_OK ||
        m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK ||
        m3_json_read_uint64(reader, &offsets[1], error) != M3_STATUS_OK ||
        m3_json_expect(reader, (uint8_t)']', error) != M3_STATUS_OK) {
        return error == NULL ? M3_STATUS_INVALID_FORMAT : error->status;
    }
    return M3_STATUS_OK;
}

static m3_status m3_safetensors_parse_metadata(m3_json_reader *reader,
                                               m3_error *error)
{
    m3_name_set names = {NULL, 0U, 0U};
    bool first = true;
    m3_status status = M3_STATUS_OK;

    if (m3_json_expect(reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)'}')) {
        char *name = NULL;
        char *value = NULL;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            status = M3_STATUS_INVALID_FORMAT;
            break;
        }
        status = m3_json_read_string(reader, &name, error);
        if (status != M3_STATUS_OK) {
            break;
        }
        status = m3_name_set_take(&names, name, error);
        if (status != M3_STATUS_OK) {
            free(name);
            break;
        }
        if (m3_json_expect(reader, (uint8_t)':', error) != M3_STATUS_OK) {
            status = M3_STATUS_INVALID_FORMAT;
            break;
        }
        status = m3_json_read_string(reader, &value, error);
        free(value);
        if (status != M3_STATUS_OK) {
            break;
        }
        first = false;
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)'}', error);
    }
    m3_name_set_dispose(&names);
    return status;
}

static m3_status m3_safetensors_parse_tensor(m3_json_reader *reader,
                                             char *name,
                                             uint64_t payload_size,
                                             m3_safetensors_tensor *tensor,
                                             m3_error *error)
{
    uint64_t shape[M3_TENSOR_MAX_RANK] = {0U};
    uint64_t offsets[2] = {0U, 0U};
    uint8_t rank = 0U;
    m3_dtype dtype = M3_DTYPE_F32;
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offsets = false;
    bool first = true;
    m3_status tensor_status;

    if (name[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Safetensors tensor name is empty");
    }
    if (m3_json_expect(reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)'}')) {
        char *field = NULL;
        m3_status status;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_read_string(reader, &field, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (m3_json_expect(reader, (uint8_t)':', error) != M3_STATUS_OK) {
            free(field);
            return M3_STATUS_INVALID_FORMAT;
        }
        if (strcmp(field, "dtype") == 0 && !has_dtype) {
            char *dtype_name = NULL;

            status = m3_json_read_string(reader, &dtype_name, error);
            if (status == M3_STATUS_OK) {
                status = m3_safetensors_parse_dtype(dtype_name, &dtype,
                                                    error);
            }
            free(dtype_name);
            has_dtype = status == M3_STATUS_OK;
        } else if (strcmp(field, "shape") == 0 && !has_shape) {
            status = m3_safetensors_parse_shape(reader, shape, &rank, error);
            has_shape = status == M3_STATUS_OK;
        } else if (strcmp(field, "data_offsets") == 0 && !has_offsets) {
            status = m3_safetensors_parse_offsets(reader, offsets, error);
            has_offsets = status == M3_STATUS_OK;
        } else {
            status = m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "unknown or duplicate tensor field '%s'", field);
        }
        free(field);
        if (status != M3_STATUS_OK) {
            return status;
        }
        first = false;
    }
    if (m3_json_expect(reader, (uint8_t)'}', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    if (!has_dtype || !has_shape || !has_offsets) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "tensor '%s' lacks required fields", name);
    }
    if (offsets[1] < offsets[0] || offsets[1] > payload_size) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "tensor '%s' data offsets exceed payload", name);
    }
    tensor_status = m3_tensor_metadata_init(&tensor->tensor, dtype, rank,
                                            shape, error);
    if (tensor_status != M3_STATUS_OK) {
        return tensor_status;
    }
    if (offsets[1] - offsets[0] != (uint64_t)tensor->tensor.byte_count) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "tensor '%s' byte count does not match shape",
                            name);
    }
    tensor->name = name;
    tensor->data_start = offsets[0];
    tensor->data_end = offsets[1];
    return M3_STATUS_OK;
}

static m3_status m3_safetensors_append_tensor(
    m3_safetensors_metadata *metadata, m3_safetensors_tensor *tensor,
    size_t *capacity, m3_error *error)
{
    if (metadata->tensor_count == *capacity) {
        size_t new_capacity = *capacity == 0U ? 64U : *capacity * 2U;
        m3_safetensors_tensor *new_tensors;

        if (new_capacity < *capacity ||
            new_capacity > SIZE_MAX / sizeof(*new_tensors)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "Safetensors inventory capacity overflows");
        }
        new_tensors = realloc(metadata->tensors,
                              new_capacity * sizeof(*new_tensors));
        if (new_tensors == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot grow Safetensors inventory");
        }
        metadata->tensors = new_tensors;
        *capacity = new_capacity;
    }
    if (tensor->tensor.byte_count > SIZE_MAX - metadata->tensor_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Safetensors byte total overflows");
    }
    metadata->tensors[metadata->tensor_count++] = *tensor;
    metadata->tensor_bytes += tensor->tensor.byte_count;
    return M3_STATUS_OK;
}

m3_status m3_safetensors_parse_header(
    const uint8_t *header, size_t header_size, uint64_t payload_size,
    m3_safetensors_metadata *metadata, m3_error *error)
{
    m3_json_reader reader;
    size_t capacity = 0U;
    bool first = true;
    m3_status status = M3_STATUS_OK;

    m3_json_reader_init(&reader, header, header_size);
    if (m3_json_expect(&reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(&reader, (uint8_t)'}')) {
        char *name = NULL;

        if (!first &&
            m3_json_expect(&reader, (uint8_t)',', error) != M3_STATUS_OK) {
            status = M3_STATUS_INVALID_FORMAT;
            break;
        }
        status = m3_json_read_string(&reader, &name, error);
        if (status != M3_STATUS_OK) {
            break;
        }
        if (m3_json_expect(&reader, (uint8_t)':', error) != M3_STATUS_OK) {
            free(name);
            status = M3_STATUS_INVALID_FORMAT;
            break;
        }
        if (strcmp(name, "__metadata__") == 0) {
            if (metadata->has_metadata) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "duplicate Safetensors metadata");
            } else {
                status = m3_safetensors_parse_metadata(&reader, error);
                metadata->has_metadata = status == M3_STATUS_OK;
            }
            free(name);
        } else {
            m3_safetensors_tensor tensor;
            size_t duplicate_index;

            (void)memset(&tensor, 0, sizeof(tensor));
            for (duplicate_index = 0U;
                 duplicate_index < metadata->tensor_count;
                 ++duplicate_index) {
                if (strcmp(metadata->tensors[duplicate_index].name, name) ==
                    0) {
                    status = m3_error_set(
                        error, M3_STATUS_INVALID_FORMAT,
                        "duplicate Safetensors tensor '%s'", name);
                    break;
                }
            }
            if (status != M3_STATUS_OK) {
                free(name);
                break;
            }
            status = m3_safetensors_parse_tensor(
                &reader, name, payload_size, &tensor, error);
            if (status == M3_STATUS_OK) {
                status = m3_safetensors_append_tensor(
                    metadata, &tensor, &capacity, error);
            }
            if (status == M3_STATUS_OK) {
                name = NULL;
            }
            free(name);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        first = false;
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(&reader, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_finish(&reader, error);
    }
    return status;
}

static int m3_safetensors_compare_offsets(const void *left,
                                          const void *right)
{
    const m3_safetensors_tensor *left_tensor = left;
    const m3_safetensors_tensor *right_tensor = right;

    if (left_tensor->data_start < right_tensor->data_start) {
        return -1;
    }
    if (left_tensor->data_start > right_tensor->data_start) {
        return 1;
    }
    if (left_tensor->data_end < right_tensor->data_end) {
        return -1;
    }
    return left_tensor->data_end > right_tensor->data_end ? 1 : 0;
}

m3_status m3_safetensors_validate_ranges(
    m3_safetensors_metadata *metadata, uint64_t payload_size,
    m3_error *error)
{
    uint64_t expected_start = 0U;
    size_t index;

    if (metadata->tensor_count > 1U) {
        qsort(metadata->tensors, metadata->tensor_count,
              sizeof(*metadata->tensors), m3_safetensors_compare_offsets);
    }
    for (index = 0U; index < metadata->tensor_count; ++index) {
        if (metadata->tensors[index].data_start != expected_start) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "Safetensors data contains a gap or overlap");
        }
        expected_start = metadata->tensors[index].data_end;
    }
    if (expected_start != payload_size) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Safetensors payload has unindexed bytes");
    }
    return M3_STATUS_OK;
}
