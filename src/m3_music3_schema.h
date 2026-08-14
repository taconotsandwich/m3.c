/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MUSIC3_SCHEMA_H
#define M3_MUSIC3_SCHEMA_H

#include "m3_model.h"
#include "m3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t tensor_count;
    uint64_t element_count;
    uint64_t payload_bytes;
    m3_dtype dtype;
} m3_music3_schema_summary;

typedef m3_status (*m3_music3_schema_visitor)(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error);

m3_status m3_music3_schema_visit(m3_component_id id,
                                 m3_music3_schema_visitor visitor,
                                 void *context, m3_error *error);
m3_status m3_music3_schema_expected_summary(
    m3_component_id id, m3_music3_schema_summary *summary, m3_error *error);
m3_status m3_music3_schema_validate_inventory(
    m3_component_id id, const m3_safetensors_tensor *tensors,
    size_t tensor_count, m3_error *error);

#endif
