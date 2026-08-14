/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_model_inspect_internal.h"

#include "m3_file.h"
#include "m3_json.h"
#include "m3_manifest.h"
#include "m3_model_index.h"

#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

char *m3_model_inspect_path_join(const char *directory, const char *name,
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

m3_status m3_model_inspect_validate_json(const char *path,
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

static bool m3_tokenizer_json_name(const char *name)
{
    return strcmp(name, "tokenizer.json") == 0 ||
           strcmp(name, "tokenizer_config.json") == 0 ||
           strcmp(name, "special_tokens_map.json") == 0 ||
           strcmp(name, "added_tokens.json") == 0 ||
           strcmp(name, "vocab.json") == 0;
}

static m3_status m3_inspect_resource_component(
    const char *root_path, m3_component_id id, m3_model_metadata *metadata,
    m3_error *error)
{
    char *component_path = m3_model_inspect_path_join(
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
        resource_path = m3_model_inspect_path_join(
            component_path, entry->d_name, error);
        if (resource_path == NULL) {
            status = error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
            break;
        }
        if (id == M3_COMPONENT_TOKENIZER &&
            m3_tokenizer_json_name(entry->d_name)) {
            status = m3_model_inspect_validate_json(
                resource_path, M3_MODEL_JSON_MAX_BYTES, error);
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
            status = m3_model_inspect_validate_json(
                resource_path, M3_CONFIG_MAX_BYTES, error);
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

static m3_status m3_add_weight_component_metadata(
    m3_model_metadata *metadata, m3_component_id id,
    const m3_weight_table *table, size_t file_count, m3_error *error)
{
    size_t index;
    m3_status status = M3_STATUS_OK;

    for (index = 0U; index < file_count && status == M3_STATUS_OK; ++index) {
        status = m3_model_metadata_add_file(metadata, id, error);
    }
    for (index = 0U;
         index < table->binding_count && status == M3_STATUS_OK; ++index) {
        status = m3_model_metadata_add_tensor(
            metadata, id, table->bindings[index].tensor.byte_count, error);
    }
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
    model_index_path = m3_model_inspect_path_join(
        path, "modular_model_index.json", error);
    if (model_index_path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    status = m3_manifest_validate_file(model_index_path, error);
    free(model_index_path);
    for (component_index = 0U;
         component_index < M3_COMPONENT_COUNT && status == M3_STATUS_OK;
         ++component_index) {
        m3_component_id id = (m3_component_id)component_index;

        if (m3_component_contains_weights(id)) {
            m3_weight_table table;
            size_t file_count = 0U;

            m3_weight_table_init(&table);
            status = m3_music3_inspect_weight_component(
                path, id, &table, &configs[component_index], &file_count,
                error);
            if (status == M3_STATUS_OK) {
                status = m3_add_weight_component_metadata(
                    &inspected, id, &table, file_count, error);
            }
            m3_weight_table_dispose(&table);
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
