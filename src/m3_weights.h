/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WEIGHTS_H
#define M3_WEIGHTS_H

#include "m3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *path;
    const m3_safetensors_metadata *metadata;
} m3_weight_shard_source;

typedef struct {
    char *path;
    uint64_t data_section_offset;
    uint64_t payload_bytes;
    m3_file_snapshot snapshot;
} m3_weight_shard_record;

typedef struct {
    char *name;
    m3_tensor_metadata tensor;
    size_t shard_index;
    uint64_t data_start;
    uint64_t data_end;
} m3_weight_binding;

typedef struct {
    m3_weight_shard_record *shards;
    size_t shard_count;
    m3_weight_binding *bindings;
    size_t binding_count;
    uint64_t aggregate_payload_bytes;
} m3_weight_table;

typedef struct {
    const char *name;
    m3_tensor_metadata tensor;
} m3_weight_requirement;

void m3_weight_table_init(m3_weight_table *table);
void m3_weight_table_dispose(m3_weight_table *table);
m3_status m3_weight_table_build(m3_weight_table *table,
                                const m3_weight_shard_source *shards,
                                size_t shard_count, m3_error *error);
const m3_weight_binding *m3_weight_table_find(const m3_weight_table *table,
                                              const char *name);
m3_status m3_weight_table_validate_required(
    const m3_weight_table *table, const m3_weight_requirement *requirements,
    size_t requirement_count, m3_error *error);
m3_status m3_weight_table_validate_no_extra(
    const m3_weight_table *table, const m3_weight_requirement *requirements,
    size_t requirement_count, m3_error *error);

#endif
