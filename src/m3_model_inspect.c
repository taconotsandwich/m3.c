/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_model.h"

#include "m3_file.h"
#include "m3_json.h"
#include "m3_manifest.h"
#include "m3_model_index.h"
#include "m3_music3_config.h"
#include "m3_music3_schema.h"
#include "m3_safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define M3_CONFIG_MAX_BYTES (4U * 1024U * 1024U)

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} m3_string_list;

static void m3_string_list_dispose(m3_string_list *list)
{
    size_t index;

    for (index = 0U; index < list->count; ++index) {
        free(list->items[index]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static bool m3_string_list_contains(const m3_string_list *list,
                                    const char *value)
{
    size_t index;

    for (index = 0U; index < list->count; ++index) {
        if (strcmp(list->items[index], value) == 0) {
            return true;
        }
    }
    return false;
}

static m3_status m3_string_list_add(m3_string_list *list, const char *value,
                                    bool reject_duplicate, m3_error *error)
{
    size_t length;
    char *copy;

    if (reject_duplicate && m3_string_list_contains(list, value)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "duplicate model resource '%s'", value);
    }
    if (list->count == list->capacity) {
        size_t new_capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
        char **new_items;

        if (new_capacity < list->capacity ||
            new_capacity > SIZE_MAX / sizeof(*new_items)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "model resource list overflows");
        }
        new_items = realloc(list->items,
                            new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot grow model resource list");
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }
    length = strlen(value);
    if (length == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "model resource name overflows");
    }
    copy = malloc(length + 1U);
    if (copy == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot copy model resource name");
    }
    (void)memcpy(copy, value, length + 1U);
    list->items[list->count++] = copy;
    return M3_STATUS_OK;
}

static char *m3_path_join(const char *directory, const char *name,
                          m3_error *error)
{
    size_t directory_length = strlen(directory);
    size_t name_length = strlen(name);
    bool needs_separator = directory_length != 0U &&
                           directory[directory_length - 1U] != '/';
    size_t separator_length = needs_separator ? 1U : 0U;
    size_t path_length;
    char *path;

    if (directory_length > SIZE_MAX - name_length - separator_length - 1U) {
        (void)m3_error_set(error, M3_STATUS_OVERFLOW, "model path overflows");
        return NULL;
    }
    path_length = directory_length + separator_length + name_length;
    path = malloc(path_length + 1U);
    if (path == NULL) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                           "cannot allocate model path");
        return NULL;
    }
    (void)memcpy(path, directory, directory_length);
    if (needs_separator) {
        path[directory_length] = '/';
    }
    (void)memcpy(path + directory_length + separator_length, name,
                 name_length + 1U);
    return path;
}

static bool m3_name_has_suffix(const char *name, const char *suffix)
{
    size_t name_length = strlen(name);
    size_t suffix_length = strlen(suffix);

    return name_length > suffix_length &&
           strcmp(name + name_length - suffix_length, suffix) == 0;
}

static m3_status m3_validate_json_resource(const char *path,
                                           size_t maximum_size,
                                           m3_error *error)
{
    uint8_t *data = NULL;
    size_t size = 0U;
    m3_json_reader reader;
    m3_status status = m3_file_read_bounded(path, maximum_size, &data, &size,
                                            error);

    if (status == M3_STATUS_OK) {
        m3_json_reader_init(&reader, data, size);
        if (!m3_json_next_is(&reader, (uint8_t)'{')) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "JSON resource '%s' is not an object", path);
        } else {
            status = m3_json_validate(data, size, error);
        }
    }
    free(data);
    return status;
}

static m3_status m3_validate_plain_resource(const char *path,
                                            m3_error *error)
{
    uint8_t *data = NULL;
    size_t size = 0U;
    m3_status status = m3_file_read_bounded(
        path, M3_MODEL_JSON_MAX_BYTES, &data, &size, error);

    free(data);
    return status;
}

static bool m3_weight_config_name(m3_component_id id, const char *name)
{
    return strcmp(name, "config.json") == 0 ||
           (id == M3_COMPONENT_LANGUAGE_MODEL &&
            strcmp(name, "generation_config.json") == 0);
}

static bool m3_tokenizer_json_name(const char *name)
{
    return strcmp(name, "tokenizer.json") == 0 ||
           strcmp(name, "tokenizer_config.json") == 0 ||
           strcmp(name, "special_tokens_map.json") == 0 ||
           strcmp(name, "added_tokens.json") == 0 ||
           strcmp(name, "vocab.json") == 0;
}

static int m3_compare_strings(const void *left, const void *right)
{
    const char *const *left_string = left;
    const char *const *right_string = right;

    return strcmp(*left_string, *right_string);
}

static m3_status m3_scan_weight_names(
    const char *component_path, m3_component_id id, m3_string_list *shards,
    char **index_name, m3_music3_component_config *config,
    m3_model_metadata *metadata, m3_error *error)
{
    DIR *directory = opendir(component_path);
    struct dirent *entry;
    bool has_config = false;
    m3_status status = M3_STATUS_OK;

    if (directory == NULL) {
        return m3_error_set(error, M3_STATUS_IO,
                            "cannot open component '%s': %s", component_path,
                            strerror(errno));
    }
    errno = 0;
    while ((entry = readdir(directory)) != NULL) {
        char *resource_path;

        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        resource_path = m3_path_join(component_path, entry->d_name, error);
        if (resource_path == NULL) {
            status = error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
            break;
        }
        if (m3_weight_config_name(id, entry->d_name)) {
            if (strcmp(entry->d_name, "config.json") == 0) {
                status = m3_music3_config_read_file(resource_path, id,
                                                    config, error);
                has_config = true;
            } else {
                status = m3_validate_json_resource(resource_path,
                                                   M3_CONFIG_MAX_BYTES,
                                                   error);
            }
        } else if (m3_name_has_suffix(entry->d_name,
                                      ".safetensors.index.json")) {
            if (*index_name != NULL) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "component '%s' has multiple indexes",
                                      component_path);
            } else {
                *index_name = resource_path;
                resource_path = NULL;
            }
        } else if (m3_name_has_suffix(entry->d_name, ".safetensors")) {
            status = m3_string_list_add(shards, entry->d_name, true, error);
        } else {
            status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                                  "unsupported resource '%s/%s'",
                                  component_path, entry->d_name);
        }
        if (status == M3_STATUS_OK) {
            status = m3_model_metadata_add_file(metadata, id, error);
        }
        free(resource_path);
        if (status != M3_STATUS_OK) {
            break;
        }
        errno = 0;
    }
    if (entry == NULL && errno != 0 && status == M3_STATUS_OK) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "cannot read component '%s': %s",
                              component_path, strerror(errno));
    }
    if (closedir(directory) != 0 && status == M3_STATUS_OK) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "cannot close component '%s': %s",
                              component_path, strerror(errno));
    }
    if (status == M3_STATUS_OK && (!has_config || shards->count == 0U)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "component '%s' lacks config or weights",
                              component_path);
    }
    if (status == M3_STATUS_OK) {
        size_t expected_shards =
            id == M3_COMPONENT_LANGUAGE_MODEL
                ? 4U
                : (id == M3_COMPONENT_TRANSFORMER ? 2U : 1U);
        bool expects_index = id == M3_COMPONENT_LANGUAGE_MODEL ||
                             id == M3_COMPONENT_TRANSFORMER;

        if (shards->count != expected_shards ||
            ((*index_name != NULL) != expects_index)) {
            status = m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "component '%s' does not use the official shard layout",
                component_path);
        }
    }
    return status;
}

static m3_status m3_inventory_take_shard(
    m3_safetensors_metadata *inventory, m3_safetensors_metadata *shard,
    size_t *capacity, m3_error *error)
{
    size_t required;

    if (shard->tensor_count > SIZE_MAX - inventory->tensor_count) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 inventory count overflows");
    }
    required = inventory->tensor_count + shard->tensor_count;
    if (required > *capacity) {
        size_t new_capacity = *capacity == 0U ? 64U : *capacity;
        m3_safetensors_tensor *new_tensors;

        while (new_capacity < required) {
            if (new_capacity > SIZE_MAX / 2U) {
                return m3_error_set(error, M3_STATUS_OVERFLOW,
                                    "Music3 inventory capacity overflows");
            }
            new_capacity *= 2U;
        }
        if (new_capacity > SIZE_MAX / sizeof(*new_tensors)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "Music3 inventory allocation overflows");
        }
        new_tensors = realloc(inventory->tensors,
                              new_capacity * sizeof(*new_tensors));
        if (new_tensors == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot grow Music3 tensor inventory");
        }
        inventory->tensors = new_tensors;
        *capacity = new_capacity;
    }
    if (shard->tensor_bytes > SIZE_MAX - inventory->tensor_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 inventory byte count overflows");
    }
    (void)memcpy(inventory->tensors + inventory->tensor_count,
                 shard->tensors,
                 shard->tensor_count * sizeof(*shard->tensors));
    inventory->tensor_count = required;
    inventory->tensor_bytes += shard->tensor_bytes;
    free(shard->tensors);
    shard->tensors = NULL;
    shard->tensor_count = 0U;
    shard->tensor_bytes = 0U;
    return M3_STATUS_OK;
}

static m3_status m3_validate_shard_tensors(
    const char *component_path, const char *shard_name,
    const m3_model_index *index, m3_safetensors_metadata *inventory,
    size_t *inventory_capacity, m3_error *error)
{
    char *path = m3_path_join(component_path, shard_name, error);
    m3_safetensors_metadata shard;
    m3_status status;
    size_t tensor_index;

    if (path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    m3_safetensors_metadata_init(&shard);
    status = m3_safetensors_inspect_file(path, &shard, error);
    free(path);
    if (status == M3_STATUS_OK && shard.tensor_count == 0U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "weight shard '%s' is empty", shard_name);
    }
    for (tensor_index = 0U;
         tensor_index < shard.tensor_count && status == M3_STATUS_OK;
         ++tensor_index) {
        const char *indexed_shard = index == NULL
                                        ? NULL
                                        : m3_model_index_shard(
                                              index,
                                              shard.tensors[tensor_index].name);

        if (index != NULL &&
            (indexed_shard == NULL || strcmp(indexed_shard, shard_name) != 0)) {
            status = m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "index does not map tensor '%s' to shard '%s'",
                shard.tensors[tensor_index].name, shard_name);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_inventory_take_shard(inventory, &shard,
                                         inventory_capacity, error);
    }
    m3_safetensors_metadata_dispose(&shard);
    return status;
}

static m3_status m3_inspect_weight_component(
    const char *root_path, m3_component_id id, m3_model_metadata *metadata,
    m3_music3_component_config *config, m3_error *error)
{
    char *component_path = m3_path_join(
        root_path, m3_component_directory(id), error);
    char *index_path = NULL;
    m3_string_list shards = {NULL, 0U, 0U};
    m3_model_index index;
    m3_model_index *active_index = NULL;
    m3_safetensors_metadata inventory;
    size_t inventory_capacity = 0U;
    m3_status status;
    size_t shard_index;
    size_t tensor_index;

    if (component_path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    m3_model_index_init(&index);
    m3_safetensors_metadata_init(&inventory);
    status = m3_scan_weight_names(component_path, id, &shards, &index_path,
                                  config, metadata, error);
    if (status == M3_STATUS_OK && index_path != NULL) {
        status = m3_model_index_read(index_path, &index, error);
        active_index = status == M3_STATUS_OK ? &index : NULL;
    }
    if (status == M3_STATUS_OK) {
        qsort(shards.items, shards.count, sizeof(*shards.items),
              m3_compare_strings);
    }
    for (shard_index = 0U;
         shard_index < shards.count && status == M3_STATUS_OK;
         ++shard_index) {
        status = m3_validate_shard_tensors(
            component_path, shards.items[shard_index], active_index,
            &inventory, &inventory_capacity, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_validate_inventory(
            id, inventory.tensors, inventory.tensor_count, error);
    }
    if (status == M3_STATUS_OK && active_index != NULL &&
        active_index->entry_count != inventory.tensor_count) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "model index tensor count does not match shards");
    }
    if (status == M3_STATUS_OK && active_index != NULL &&
        (!active_index->has_total_size ||
         active_index->total_size != (uint64_t)inventory.tensor_bytes)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "model index total_size is missing or incorrect");
    }
    for (tensor_index = 0U;
         tensor_index < inventory.tensor_count && status == M3_STATUS_OK;
         ++tensor_index) {
        status = m3_model_metadata_add_tensor(
            metadata, id, inventory.tensors[tensor_index].tensor.byte_count,
            error);
    }
    m3_model_index_dispose(&index);
    m3_safetensors_metadata_dispose(&inventory);
    m3_string_list_dispose(&shards);
    free(index_path);
    free(component_path);
    return status;
}

static m3_status m3_inspect_resource_component(
    const char *root_path, m3_component_id id, m3_model_metadata *metadata,
    m3_error *error)
{
    char *component_path = m3_path_join(
        root_path, m3_component_directory(id), error);
    DIR *directory;
    struct dirent *entry;
    bool has_primary = false;
    bool has_vocab = false;
    bool has_merges = false;
    m3_status status = M3_STATUS_OK;

    if (component_path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    directory = opendir(component_path);
    if (directory == NULL) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "cannot open component '%s': %s",
                              component_path, strerror(errno));
        free(component_path);
        return status;
    }
    errno = 0;
    while ((entry = readdir(directory)) != NULL) {
        char *resource_path;

        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        resource_path = m3_path_join(component_path, entry->d_name, error);
        if (resource_path == NULL) {
            status = error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
            break;
        }
        if (id == M3_COMPONENT_TOKENIZER &&
            m3_tokenizer_json_name(entry->d_name)) {
            status = m3_validate_json_resource(resource_path,
                                               M3_MODEL_JSON_MAX_BYTES,
                                               error);
            has_primary = has_primary ||
                          strcmp(entry->d_name, "tokenizer.json") == 0;
            has_vocab = has_vocab || strcmp(entry->d_name, "vocab.json") == 0;
        } else if (id == M3_COMPONENT_TOKENIZER &&
                   strcmp(entry->d_name, "merges.txt") == 0) {
            status = m3_validate_plain_resource(resource_path, error);
            has_merges = status == M3_STATUS_OK;
        } else if (id == M3_COMPONENT_TOKENIZER &&
                   strcmp(entry->d_name, "chat_template.jinja") == 0) {
            status = m3_validate_plain_resource(resource_path, error);
        } else if (id == M3_COMPONENT_SCHEDULER &&
                   strcmp(entry->d_name, "scheduler_config.json") == 0) {
            status = m3_validate_json_resource(resource_path,
                                               M3_CONFIG_MAX_BYTES, error);
            has_primary = status == M3_STATUS_OK;
        } else {
            status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                                  "unsupported resource '%s/%s'",
                                  component_path, entry->d_name);
        }
        if (status == M3_STATUS_OK) {
            status = m3_model_metadata_add_file(metadata, id, error);
        }
        free(resource_path);
        if (status != M3_STATUS_OK) {
            break;
        }
        errno = 0;
    }
    if (entry == NULL && errno != 0 && status == M3_STATUS_OK) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "cannot read component '%s': %s",
                              component_path, strerror(errno));
    }
    if (closedir(directory) != 0 && status == M3_STATUS_OK) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "cannot close component '%s': %s",
                              component_path, strerror(errno));
    }
    if (status == M3_STATUS_OK && id == M3_COMPONENT_TOKENIZER &&
        !has_primary && !(has_vocab && has_merges)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "tokenizer lacks tokenizer.json or vocab/merges");
    }
    if (status == M3_STATUS_OK && id == M3_COMPONENT_SCHEDULER &&
        !has_primary) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "scheduler lacks scheduler_config.json");
    }
    free(component_path);
    return status;
}

m3_status m3_model_inspect_directory(const char *path,
                                     m3_model_metadata *metadata,
                                     m3_error *error)
{
    char *model_index_path;
    struct stat root_stat;
    m3_model_metadata inspected;
    m3_music3_component_config configs[M3_COMPONENT_COUNT];
    m3_status status;
    size_t component_index;

    if (path == NULL || path[0] == '\0' || metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "model inspection argument is null");
    }
    if (stat(path, &root_stat) != 0 || !S_ISDIR(root_stat.st_mode)) {
        return m3_error_set(error, M3_STATUS_IO,
                            "model path is not a readable directory");
    }
    m3_model_metadata_init(&inspected);
    for (component_index = 0U; component_index < M3_COMPONENT_COUNT;
         ++component_index) {
        m3_music3_component_config_init(&configs[component_index]);
    }
    status = M3_STATUS_OK;
    model_index_path = m3_path_join(path, "modular_model_index.json", error);
    if (model_index_path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    if (status == M3_STATUS_OK) {
        status = m3_manifest_validate_file(model_index_path, error);
    }
    free(model_index_path);
    for (component_index = 0U;
         component_index < M3_COMPONENT_COUNT && status == M3_STATUS_OK;
         ++component_index) {
        m3_component_id id = (m3_component_id)component_index;

        if (m3_component_contains_weights(id)) {
            status = m3_inspect_weight_component(
                path, id, &inspected, &configs[component_index], error);
        } else {
            status = m3_inspect_resource_component(path, id, &inspected,
                                                   error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_config_validate_cross(configs, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (inspected.present_component_count != M3_COMPONENT_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model does not contain all official components");
    }
    *metadata = inspected;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
