/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MODEL_INDEX_H
#define M3_MODEL_INDEX_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_MODEL_JSON_MAX_BYTES (64U * 1024U * 1024U)

typedef struct {
    char *tensor_name;
    char *shard_name;
} m3_model_index_entry;

typedef struct {
    m3_model_index_entry *entries;
    size_t entry_count;
    bool has_total_size;
    uint64_t total_size;
} m3_model_index;

void m3_model_index_init(m3_model_index *index);
void m3_model_index_dispose(m3_model_index *index);
m3_status m3_model_index_read(const char *path, m3_model_index *index,
                              m3_error *error);
const char *m3_model_index_shard(const m3_model_index *index,
                                 const char *tensor_name);

#endif
