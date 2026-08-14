/* SPDX-License-Identifier: Unicode-3.0 */
/*
 * Unicode Character Database 16.0.0, mechanically derived for m3.c.
 * Copyright 1991-2024 Unicode, Inc. All rights reserved.
 * Distributed under the Unicode Terms of Use:
 * https://www.unicode.org/copyright.html
 * Complete notice: LICENSES/Unicode-3.0.txt
 *
 * UnicodeData.txt SHA-256:
 * ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f
 * CompositionExclusions.txt SHA-256:
 * 89e83cf9cc8bef6c1f8bf77e42cf6f0341dfa42e66261f4dbe9b492e7a23c8ee
 * CaseFolding.txt SHA-256:
 * 6f1f9c588eb4a5c718d9e8f93b782685e5c7fec872cf05e8e6878053599e09bb
 * PropList.txt SHA-256:
 * 53d614508e2a0b2305a8aa21cd60d993de9326cdf65993660dfcce4503548583
 */
#include "m3_unicode_internal.h"

extern const m3_unicode_range m3_unicode_letter_0[];
extern const m3_unicode_range m3_unicode_letter_1[];
extern const m3_unicode_range m3_unicode_number_0[];
extern const m3_unicode_range m3_unicode_whitespace_0[];
extern const m3_unicode_ccc m3_unicode_ccc_0[];
extern const m3_unicode_ccc m3_unicode_ccc_1[];
extern const m3_unicode_ccc m3_unicode_ccc_2[];
extern const m3_unicode_decomposition m3_unicode_decomposition_0[];
extern const m3_unicode_decomposition m3_unicode_decomposition_1[];
extern const m3_unicode_decomposition m3_unicode_decomposition_2[];
extern const m3_unicode_decomposition m3_unicode_decomposition_3[];
extern const m3_unicode_decomposition m3_unicode_decomposition_4[];
extern const m3_unicode_decomposition m3_unicode_decomposition_5[];
extern const m3_unicode_composition m3_unicode_composition_0[];
extern const m3_unicode_composition m3_unicode_composition_1[];
extern const m3_unicode_composition m3_unicode_composition_2[];
extern const m3_unicode_fold m3_unicode_fold_0[];

const m3_unicode_range_shard m3_unicode_letter_shards[] = {
    {m3_unicode_letter_0, 400U},
    {m3_unicode_letter_1, 277U},
};
const size_t m3_unicode_letter_shard_count =
    sizeof(m3_unicode_letter_shards) / sizeof(m3_unicode_letter_shards[0]);

const m3_unicode_range_shard m3_unicode_number_shards[] = {
    {m3_unicode_number_0, 144U},
};
const size_t m3_unicode_number_shard_count =
    sizeof(m3_unicode_number_shards) / sizeof(m3_unicode_number_shards[0]);

const m3_unicode_range_shard m3_unicode_whitespace_shards[] = {
    {m3_unicode_whitespace_0, 10U},
};
const size_t m3_unicode_whitespace_shard_count =
    sizeof(m3_unicode_whitespace_shards) / sizeof(m3_unicode_whitespace_shards[0]);

const m3_unicode_ccc_shard m3_unicode_ccc_shards[] = {
    {m3_unicode_ccc_0, 400U},
    {m3_unicode_ccc_1, 400U},
    {m3_unicode_ccc_2, 134U},
};
const size_t m3_unicode_ccc_shard_count =
    sizeof(m3_unicode_ccc_shards) / sizeof(m3_unicode_ccc_shards[0]);

const m3_unicode_decomposition_shard m3_unicode_decomposition_shards[] = {
    {m3_unicode_decomposition_0, 400U},
    {m3_unicode_decomposition_1, 400U},
    {m3_unicode_decomposition_2, 400U},
    {m3_unicode_decomposition_3, 400U},
    {m3_unicode_decomposition_4, 400U},
    {m3_unicode_decomposition_5, 81U},
};
const size_t m3_unicode_decomposition_shard_count =
    sizeof(m3_unicode_decomposition_shards) / sizeof(m3_unicode_decomposition_shards[0]);

const m3_unicode_composition_shard m3_unicode_composition_shards[] = {
    {m3_unicode_composition_0, 400U},
    {m3_unicode_composition_1, 400U},
    {m3_unicode_composition_2, 161U},
};
const size_t m3_unicode_composition_shard_count =
    sizeof(m3_unicode_composition_shards) / sizeof(m3_unicode_composition_shards[0]);

const m3_unicode_fold_shard m3_unicode_fold_shards[] = {
    {m3_unicode_fold_0, 9U},
};
const size_t m3_unicode_fold_shard_count =
    sizeof(m3_unicode_fold_shards) / sizeof(m3_unicode_fold_shards[0]);
