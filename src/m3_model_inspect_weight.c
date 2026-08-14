/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_model_inspect_internal.h"

#include "m3_model_index.h"
#include "m3_music3_schema.h"
#include "m3_safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
                                    m3_error *error)
{
    size_t length;
    char *copy;

    if (m3_string_list_contains(list, value)) {
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

static bool m3_name_has_suffix(const char *name, const char *suffix)
{
    size_t name_length = strlen(name);
    size_t suffix_length = strlen(suffix);

    return name_length > suffix_length &&
           strcmp(name + name_length - suffix_length, suffix) == 0;
}

static int m3_compare_strings(const void *left, const void *right)
{
    const char *const *left_string = left;
    const char *const *right_string = right;

    return strcmp(*left_string, *right_string);
}

static size_t m3_weight_expected_shard_count(m3_component_id id)
{
    if (id == M3_COMPONENT_LANGUAGE_MODEL) {
        return 4U;
    }
    return id == M3_COMPONENT_TRANSFORMER ? 2U : 1U;
}

static bool m3_weight_expected_shard_name(m3_component_id id, size_t index,
                                          char name[80])
{
    int count;

    if (id == M3_COMPONENT_LANGUAGE_MODEL) {
        count = snprintf(name, 80U, "model-%05zu-of-00004.safetensors",
                         index + 1U);
    } else if (id == M3_COMPONENT_TRANSFORMER) {
        count = snprintf(
            name, 80U,
            "diffusion_pytorch_model-%05zu-of-00002.safetensors",
            index + 1U);
    } else {
        count = snprintf(name, 80U, "diffusion_pytorch_model.safetensors");
    }
    return count >= 0 && count < 80;
}

static const char *m3_weight_expected_index_name(m3_component_id id)
{
    if (id == M3_COMPONENT_LANGUAGE_MODEL) {
        return "model.safetensors.index.json";
    }
    if (id == M3_COMPONENT_TRANSFORMER) {
        return "diffusion_pytorch_model.safetensors.index.json";
    }
    return NULL;
}

static bool m3_weight_config_name(m3_component_id id, const char *name)
{
    return strcmp(name, "config.json") == 0 ||
           (id == M3_COMPONENT_LANGUAGE_MODEL &&
            strcmp(name, "generation_config.json") == 0);
}

static m3_status m3_scan_weight_names(
    const char *component_path, m3_component_id id, m3_string_list *shards,
    char **index_path, m3_music3_component_config *config,
    size_t *file_count, m3_error *error)
{
    DIR *directory = opendir(component_path);
    struct dirent *entry;
    const char *expected_index = m3_weight_expected_index_name(id);
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
        resource_path = m3_model_inspect_path_join(
            component_path, entry->d_name, error);
        if (resource_path == NULL) {
            status = error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
            break;
        }
        if (m3_weight_config_name(id, entry->d_name)) {
            if (strcmp(entry->d_name, "config.json") == 0) {
                status = m3_music3_config_read_file(resource_path, id, config,
                                                    error);
                has_config = status == M3_STATUS_OK;
            } else {
                status = m3_model_inspect_validate_json(
                    resource_path, M3_CONFIG_MAX_BYTES, error);
            }
        } else if (m3_name_has_suffix(entry->d_name,
                                      ".safetensors.index.json")) {
            if (expected_index == NULL ||
                strcmp(entry->d_name, expected_index) != 0 ||
                *index_path != NULL) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "component '%s' has a non-official model index",
                    component_path);
            } else {
                *index_path = resource_path;
                resource_path = NULL;
            }
        } else if (m3_name_has_suffix(entry->d_name, ".safetensors")) {
            status = m3_string_list_add(shards, entry->d_name, error);
        } else {
            status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                                  "unsupported resource '%s/%s'",
                                  component_path, entry->d_name);
        }
        if (status == M3_STATUS_OK) {
            if (*file_count == SIZE_MAX) {
                status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                      "component file count overflows");
            } else {
                *file_count += 1U;
            }
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
    if (status == M3_STATUS_OK &&
        (shards->count != m3_weight_expected_shard_count(id) ||
         ((*index_path != NULL) != (expected_index != NULL)))) {
        status = m3_error_set(
            error, M3_STATUS_INVALID_FORMAT,
            "component '%s' does not use the official shard layout",
            component_path);
    }
    return status;
}

static m3_status m3_validate_shard_names(m3_component_id id,
                                         m3_string_list *shards,
                                         m3_error *error)
{
    size_t index;

    qsort(shards->items, shards->count, sizeof(*shards->items),
          m3_compare_strings);
    for (index = 0U; index < shards->count; ++index) {
        char expected[80];

        if (!m3_weight_expected_shard_name(id, index, expected) ||
            strcmp(shards->items[index], expected) != 0) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "component %s has a non-official shard name",
                m3_component_directory(id));
        }
    }
    return M3_STATUS_OK;
}

static void m3_dispose_inspected_shards(m3_safetensors_metadata *metadata,
                                        char **paths, size_t shard_count)
{
    size_t index;

    for (index = 0U; index < shard_count; ++index) {
        m3_safetensors_metadata_dispose(&metadata[index]);
        free(paths[index]);
    }
    free(metadata);
    free(paths);
}

static m3_status m3_inspect_shards(
    const char *component_path, const m3_string_list *shards,
    m3_weight_table *table, m3_error *error)
{
    m3_safetensors_metadata *metadata;
    m3_weight_shard_source *sources;
    char **paths;
    size_t index;
    m3_status status = M3_STATUS_OK;

    metadata = calloc(shards->count, sizeof(*metadata));
    sources = calloc(shards->count, sizeof(*sources));
    paths = calloc(shards->count, sizeof(*paths));
    if (metadata == NULL || sources == NULL || paths == NULL) {
        free(metadata);
        free(sources);
        free(paths);
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate component shard inspection");
    }
    for (index = 0U; index < shards->count; ++index) {
        m3_safetensors_metadata_init(&metadata[index]);
        paths[index] = m3_model_inspect_path_join(
            component_path, shards->items[index], error);
        if (paths[index] == NULL) {
            status = error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
            break;
        }
        status = m3_safetensors_inspect_file(paths[index], &metadata[index],
                                             error);
        if (status != M3_STATUS_OK) {
            break;
        }
        sources[index].path = paths[index];
        sources[index].metadata = &metadata[index];
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_table_build(table, sources, shards->count, error);
    }
    m3_dispose_inspected_shards(metadata, paths, shards->count);
    free(sources);
    return status;
}

static const char *m3_path_basename(const char *path)
{
    const char *separator = strrchr(path, '/');

    return separator == NULL ? path : separator + 1;
}

static m3_status m3_validate_table_index(const m3_weight_table *table,
                                         const m3_model_index *index,
                                         m3_error *error)
{
    size_t binding_index;

    for (binding_index = 0U; binding_index < table->binding_count;
         ++binding_index) {
        const m3_weight_binding *binding = &table->bindings[binding_index];
        const char *indexed_shard =
            m3_model_index_shard(index, binding->name);
        const char *actual_shard =
            m3_path_basename(table->shards[binding->shard_index].path);

        if (indexed_shard == NULL ||
            strcmp(indexed_shard, actual_shard) != 0) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "index does not map tensor '%s' to shard '%s'",
                binding->name, actual_shard);
        }
    }
    if (index->entry_count != table->binding_count) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model index tensor count does not match shards");
    }
    if (!index->has_total_size ||
        index->total_size != table->aggregate_payload_bytes) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "model index total_size is missing or incorrect");
    }
    return M3_STATUS_OK;
}

static m3_status m3_validate_table_schema(m3_component_id id,
                                          const m3_weight_table *table,
                                          m3_error *error)
{
    m3_safetensors_tensor *inventory;
    size_t index;
    m3_status status;

    if (table->binding_count > SIZE_MAX / sizeof(*inventory)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 schema inventory allocation overflows");
    }
    inventory = calloc(table->binding_count, sizeof(*inventory));
    if (inventory == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Music3 schema inventory");
    }
    for (index = 0U; index < table->binding_count; ++index) {
        inventory[index].name = table->bindings[index].name;
        inventory[index].tensor = table->bindings[index].tensor;
        inventory[index].data_start = table->bindings[index].data_start;
        inventory[index].data_end = table->bindings[index].data_end;
    }
    status = m3_music3_schema_validate_inventory(
        id, inventory, table->binding_count, error);
    free(inventory);
    return status;
}

m3_status m3_music3_inspect_weight_component(
    const char *model_root, m3_component_id component, m3_weight_table *table,
    m3_music3_component_config *config, size_t *file_count,
    m3_error *error)
{
    char *component_path;
    char *index_path = NULL;
    m3_string_list shards = {NULL, 0U, 0U};
    m3_model_index index;
    m3_weight_table built;
    m3_music3_component_config inspected_config;
    size_t inspected_file_count = 0U;
    m3_status status;

    if (model_root == NULL || model_root[0] == '\0' ||
        !m3_component_contains_weights(component) || table == NULL ||
        config == NULL || file_count == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 weight inspection argument is invalid");
    }
    m3_model_index_init(&index);
    m3_weight_table_init(&built);
    m3_music3_component_config_init(&inspected_config);
    component_path = m3_model_inspect_path_join(
        model_root, m3_component_directory(component), error);
    if (component_path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    status = m3_scan_weight_names(
        component_path, component, &shards, &index_path, &inspected_config,
        &inspected_file_count, error);
    if (status == M3_STATUS_OK) {
        status = m3_validate_shard_names(component, &shards, error);
    }
    if (status == M3_STATUS_OK && index_path != NULL) {
        status = m3_model_index_read(index_path, &index, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_inspect_shards(component_path, &shards, &built, error);
    }
    if (status == M3_STATUS_OK && index_path != NULL) {
        status = m3_validate_table_index(&built, &index, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_validate_table_schema(component, &built, error);
    }
    m3_model_index_dispose(&index);
    m3_string_list_dispose(&shards);
    free(index_path);
    free(component_path);
    if (status != M3_STATUS_OK) {
        m3_weight_table_dispose(&built);
        return status;
    }
    m3_weight_table_dispose(table);
    *table = built;
    *config = inspected_config;
    *file_count = inspected_file_count;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_music3_inspect_weight_table(const char *model_root,
                                         m3_component_id component,
                                         m3_weight_table *table,
                                         m3_error *error)
{
    m3_music3_component_config config;
    size_t file_count = 0U;

    m3_music3_component_config_init(&config);
    return m3_music3_inspect_weight_component(
        model_root, component, table, &config, &file_count, error);
}
