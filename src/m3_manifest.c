/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_manifest.h"

#include "m3_file.h"
#include "m3_json.h"
#include "m3_model.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define M3_MANIFEST_MAX_BYTES (4U * 1024U * 1024U)

static bool m3_manifest_component_id(const char *name,
                                     m3_component_id *component_id)
{
    size_t component_index;

    for (component_index = 0U; component_index < M3_COMPONENT_COUNT;
         ++component_index) {
        m3_component_id candidate = (m3_component_id)component_index;

        if (strcmp(name, m3_component_directory(candidate)) == 0) {
            *component_id = candidate;
            return true;
        }
    }
    return false;
}

static m3_status m3_manifest_read_declaration(m3_json_reader *reader,
                                              m3_component_id component_id,
                                              m3_error *error)
{
    static const char *const libraries[M3_COMPONENT_COUNT] = {
        "transformers", "diffusers", "diffusers", "diffusers",
        "diffusers", "transformers", "diffusers"
    };
    static const char *const classes[M3_COMPONENT_COUNT] = {
        "Qwen3ForCausalLM",
        "MiniMaxMusic3RVQDepthDecoder",
        "MiniMaxMusic3ConditionEncoder",
        "MiniMaxMusic3Transformer1DModel",
        "MiniMaxMusic3Vocoder",
        "Qwen2Tokenizer",
        "FlowMatchEulerDiscreteScheduler"
    };
    const char *component_name = m3_component_directory(component_id);
    char *library_name = NULL;
    char *class_name = NULL;
    m3_status status;

    status = m3_json_expect(reader, (uint8_t)'[', error);
    if (status == M3_STATUS_OK) {
        status = m3_json_read_string(reader, &library_name, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)',', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_read_string(reader, &class_name, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)',', error);
    }
    if (status == M3_STATUS_OK && !m3_json_next_is(reader, (uint8_t)'{')) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "manifest component '%s' config is not an object",
                              component_name);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_skip_value(reader, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)']', error);
    }
    if (status == M3_STATUS_OK &&
        (strcmp(library_name, libraries[(size_t)component_id]) != 0 ||
         strcmp(class_name, classes[(size_t)component_id]) != 0)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "manifest component '%s' has the wrong identity",
                              component_name);
    }
    free(library_name);
    free(class_name);
    return status;
}

static m3_status m3_manifest_read_exact_string(m3_json_reader *reader,
                                               const char *field_name,
                                               const char *expected,
                                               m3_error *error)
{
    char *value = NULL;
    m3_status status = m3_json_read_string(reader, &value, error);

    if (status == M3_STATUS_OK && strcmp(value, expected) != 0) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "manifest field '%s' must be '%s'", field_name,
                              expected);
    }
    free(value);
    return status;
}

static m3_status m3_manifest_parse(const uint8_t *data, size_t size,
                                   m3_error *error)
{
    static const char pipeline_class[] = "MiniMaxMusic3ModularPipeline";
    static const char blocks_class[] = "MiniMaxMusic3Blocks";
    bool components[M3_COMPONENT_COUNT] = {false};
    bool has_pipeline_class = false;
    bool has_blocks_class = false;
    bool first = true;
    m3_json_reader reader;
    m3_status status;
    size_t component_index;

    m3_json_reader_init(&reader, data, size);
    status = m3_json_expect(&reader, (uint8_t)'{', error);
    while (status == M3_STATUS_OK &&
           !m3_json_next_is(&reader, (uint8_t)'}')) {
        char *field_name = NULL;
        m3_component_id component_id = M3_COMPONENT_COUNT;

        if (!first) {
            status = m3_json_expect(&reader, (uint8_t)',', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_read_string(&reader, &field_name, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_expect(&reader, (uint8_t)':', error);
        }
        if (status == M3_STATUS_OK && strcmp(field_name, "_class_name") == 0) {
            if (has_pipeline_class) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "manifest duplicates _class_name");
            } else {
                status = m3_manifest_read_exact_string(
                    &reader, "_class_name", pipeline_class, error);
                has_pipeline_class = status == M3_STATUS_OK;
            }
        } else if (status == M3_STATUS_OK &&
                   strcmp(field_name, "_blocks_class_name") == 0) {
            if (has_blocks_class) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "manifest duplicates _blocks_class_name");
            } else {
                status = m3_manifest_read_exact_string(
                    &reader, "_blocks_class_name", blocks_class, error);
                has_blocks_class = status == M3_STATUS_OK;
            }
        } else if (status == M3_STATUS_OK &&
                   m3_manifest_component_id(field_name, &component_id)) {
            if (components[(size_t)component_id]) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "manifest duplicates component '%s'",
                                      field_name);
            } else {
                status = m3_manifest_read_declaration(&reader, component_id,
                                                      error);
                components[(size_t)component_id] = status == M3_STATUS_OK;
            }
        } else if (status == M3_STATUS_OK) {
            status = m3_json_skip_value(&reader, error);
        }
        free(field_name);
        first = false;
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(&reader, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_finish(&reader, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!has_pipeline_class || !has_blocks_class) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "manifest lacks required pipeline classes");
    }
    for (component_index = 0U; component_index < M3_COMPONENT_COUNT;
         ++component_index) {
        if (!components[component_index]) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "manifest lacks component '%s'",
                                m3_component_directory(
                                    (m3_component_id)component_index));
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_manifest_validate_file(const char *path, m3_error *error)
{
    uint8_t *data = NULL;
    size_t size = 0U;
    m3_status status;

    if (path == NULL || path[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "manifest path is null");
    }
    status = m3_file_read_bounded(path, M3_MANIFEST_MAX_BYTES, &data, &size,
                                  error);
    if (status == M3_STATUS_OK) {
        status = m3_manifest_parse(data, size, error);
    }
    free(data);
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}
