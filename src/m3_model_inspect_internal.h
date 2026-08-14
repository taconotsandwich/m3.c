/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MODEL_INSPECT_INTERNAL_H
#define M3_MODEL_INSPECT_INTERNAL_H

#include "m3_model.h"
#include "m3_music3_config.h"

#define M3_CONFIG_MAX_BYTES (4U * 1024U * 1024U)

char *m3_model_inspect_path_join(const char *directory, const char *name,
                                 m3_error *error);
m3_status m3_model_inspect_validate_json(const char *path,
                                         size_t maximum_size,
                                         m3_error *error);
m3_status m3_music3_inspect_weight_component(
    const char *model_root, m3_component_id component, m3_weight_table *table,
    m3_music3_component_config *config, size_t *file_count,
    m3_error *error);

#endif
