/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_UNICODE_INTERNAL_H
#define M3_UNICODE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t first;
    uint32_t last;
} m3_unicode_range;

typedef struct {
    uint32_t code_point;
    uint8_t canonical_combining_class;
} m3_unicode_ccc;

typedef struct {
    uint32_t code_point;
    uint32_t first;
    uint32_t second;
} m3_unicode_decomposition;

typedef struct {
    uint32_t first;
    uint32_t second;
    uint32_t composed;
} m3_unicode_composition;

typedef struct {
    uint32_t code_point;
    uint32_t folded;
} m3_unicode_fold;

#define M3_UNICODE_SHARD(type) \
    typedef struct { \
        const type *items; \
        size_t count; \
    } type##_shard

M3_UNICODE_SHARD(m3_unicode_range);
M3_UNICODE_SHARD(m3_unicode_ccc);
M3_UNICODE_SHARD(m3_unicode_decomposition);
M3_UNICODE_SHARD(m3_unicode_composition);
M3_UNICODE_SHARD(m3_unicode_fold);

#undef M3_UNICODE_SHARD

extern const m3_unicode_range_shard m3_unicode_letter_shards[];
extern const size_t m3_unicode_letter_shard_count;
extern const m3_unicode_range_shard m3_unicode_number_shards[];
extern const size_t m3_unicode_number_shard_count;
extern const m3_unicode_range_shard m3_unicode_whitespace_shards[];
extern const size_t m3_unicode_whitespace_shard_count;
extern const m3_unicode_ccc_shard m3_unicode_ccc_shards[];
extern const size_t m3_unicode_ccc_shard_count;
extern const m3_unicode_decomposition_shard m3_unicode_decomposition_shards[];
extern const size_t m3_unicode_decomposition_shard_count;
extern const m3_unicode_composition_shard m3_unicode_composition_shards[];
extern const size_t m3_unicode_composition_shard_count;
extern const m3_unicode_fold_shard m3_unicode_fold_shards[];
extern const size_t m3_unicode_fold_shard_count;

#endif
