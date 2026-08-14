/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MODEL_H
#define M3_MODEL_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    M3_COMPONENT_LANGUAGE_MODEL = 0,
    M3_COMPONENT_RVQ_DEPTH_DECODER,
    M3_COMPONENT_CONDITION_ENCODER,
    M3_COMPONENT_TRANSFORMER,
    M3_COMPONENT_VOCODER,
    M3_COMPONENT_TOKENIZER,
    M3_COMPONENT_SCHEDULER,
    M3_COMPONENT_COUNT
} m3_component_id;

typedef struct {
    m3_component_id id;
    bool present;
    size_t file_count;
    size_t tensor_count;
    size_t tensor_bytes;
} m3_component_metadata;

typedef struct {
    m3_component_metadata components[M3_COMPONENT_COUNT];
    size_t present_component_count;
    size_t file_count;
    size_t tensor_count;
    size_t tensor_bytes;
} m3_model_metadata;

bool m3_component_id_is_valid(m3_component_id id);
const char *m3_component_directory(m3_component_id id);
bool m3_component_contains_weights(m3_component_id id);
void m3_model_metadata_init(m3_model_metadata *metadata);
const m3_component_metadata *m3_model_metadata_component(
    const m3_model_metadata *metadata, m3_component_id id);
m3_status m3_model_metadata_add_file(m3_model_metadata *metadata,
                                     m3_component_id id, m3_error *error);
m3_status m3_model_metadata_add_tensor(m3_model_metadata *metadata,
                                       m3_component_id id, size_t byte_count,
                                       m3_error *error);

#endif
