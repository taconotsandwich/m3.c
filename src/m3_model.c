/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_model.h"

#include <stdint.h>

static const char *const m3_component_directories[M3_COMPONENT_COUNT] = {
    "language_model",
    "rvq_depth_decoder",
    "condition_encoder",
    "transformer",
    "vocoder",
    "tokenizer",
    "scheduler"
};

bool m3_component_id_is_valid(m3_component_id id)
{
    return id >= M3_COMPONENT_LANGUAGE_MODEL && id < M3_COMPONENT_COUNT;
}

const char *m3_component_directory(m3_component_id id)
{
    if (!m3_component_id_is_valid(id)) {
        return NULL;
    }
    return m3_component_directories[(size_t)id];
}

bool m3_component_contains_weights(m3_component_id id)
{
    return m3_component_id_is_valid(id) && id <= M3_COMPONENT_VOCODER;
}

void m3_model_metadata_init(m3_model_metadata *metadata)
{
    size_t index;

    if (metadata == NULL) {
        return;
    }

    metadata->present_component_count = 0U;
    metadata->file_count = 0U;
    metadata->tensor_count = 0U;
    metadata->tensor_bytes = 0U;
    for (index = 0U; index < M3_COMPONENT_COUNT; ++index) {
        m3_component_metadata *component = &metadata->components[index];

        component->id = (m3_component_id)index;
        component->present = false;
        component->file_count = 0U;
        component->tensor_count = 0U;
        component->tensor_bytes = 0U;
    }
}

const m3_component_metadata *m3_model_metadata_component(
    const m3_model_metadata *metadata, m3_component_id id)
{
    if (metadata == NULL || !m3_component_id_is_valid(id)) {
        return NULL;
    }
    return &metadata->components[(size_t)id];
}

static void m3_model_metadata_mark_present(
    m3_model_metadata *metadata, m3_component_metadata *component)
{
    if (!component->present) {
        component->present = true;
        metadata->present_component_count += 1U;
    }
}

m3_status m3_model_metadata_add_file(m3_model_metadata *metadata,
                                     m3_component_id id, m3_error *error)
{
    m3_component_metadata *component;

    if (metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "model metadata is null");
    }
    if (!m3_component_id_is_valid(id)) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "invalid model component value %d", (int)id);
    }
    component = &metadata->components[(size_t)id];
    if (component->file_count == SIZE_MAX || metadata->file_count == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "file count overflows for component %s",
                            m3_component_directory(id));
    }

    m3_model_metadata_mark_present(metadata, component);
    component->file_count += 1U;
    metadata->file_count += 1U;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_model_metadata_add_tensor(m3_model_metadata *metadata,
                                       m3_component_id id, size_t byte_count,
                                       m3_error *error)
{
    m3_component_metadata *component;

    if (metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "model metadata is null");
    }
    if (!m3_component_id_is_valid(id)) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "invalid model component value %d", (int)id);
    }
    if (!m3_component_contains_weights(id)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "component %s does not contain model tensors",
                            m3_component_directory(id));
    }

    component = &metadata->components[(size_t)id];
    if (component->tensor_count == SIZE_MAX ||
        metadata->tensor_count == SIZE_MAX ||
        byte_count > SIZE_MAX - component->tensor_bytes ||
        byte_count > SIZE_MAX - metadata->tensor_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "tensor totals overflow for component %s",
                            m3_component_directory(id));
    }

    m3_model_metadata_mark_present(metadata, component);
    component->tensor_count += 1U;
    component->tensor_bytes += byte_count;
    metadata->tensor_count += 1U;
    metadata->tensor_bytes += byte_count;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
