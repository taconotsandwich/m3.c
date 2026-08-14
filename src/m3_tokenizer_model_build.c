/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_model_internal.h"

#include <stdlib.h>
#include <string.h>

size_t m3_tokenizer_hash_capacity(size_t count)
{
    size_t required;
    size_t capacity = 8U;

    if (count > SIZE_MAX / 2U) {
        return 0U;
    }
    required = count * 2U;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2U) {
            return 0U;
        }
        capacity *= 2U;
    }
    return capacity;
}

uint64_t m3_tokenizer_hash_bytes(const uint8_t *data, size_t size)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;

    for (index = 0U; index < size; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

m3_status m3_tokenizer_build_init(m3_tokenizer_build *build,
                                   const m3_tokenizer_contract *contract,
                                   m3_error *error)
{
    size_t name_slots;
    size_t merge_slots;
    m3_tokenizer_state *state;

    if (build == NULL || contract == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer build argument is null");
    }
    (void)memset(build, 0, sizeof(*build));
    name_slots = m3_tokenizer_hash_capacity(contract->model_vocab_count);
    merge_slots = m3_tokenizer_hash_capacity(contract->merge_count);
    if (name_slots == 0U || merge_slots == 0U ||
        name_slots > SIZE_MAX / sizeof(*build->name_slots) ||
        merge_slots > SIZE_MAX / sizeof(*state->merge_slots)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "tokenizer table size overflows");
    }
    state = calloc(1U, sizeof(*state));
    build->names = calloc(contract->model_vocab_count, sizeof(*build->names));
    build->name_slots = malloc(name_slots * sizeof(*build->name_slots));
    if (state != NULL) {
        state->model_tokens = calloc(contract->model_vocab_count,
                                     sizeof(*state->model_tokens));
        state->merge_slots = malloc(merge_slots * sizeof(*state->merge_slots));
    }
    if (state == NULL || build->names == NULL || build->name_slots == NULL ||
        state->model_tokens == NULL || state->merge_slots == NULL) {
        m3_tokenizer_state_dispose(state);
        free(build->names);
        free(build->name_slots);
        (void)memset(build, 0, sizeof(*build));
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate tokenizer tables");
    }
    (void)memset(build->name_slots, 0xFF,
                 name_slots * sizeof(*build->name_slots));
    (void)memset(state->merge_slots, 0xFF,
                 merge_slots * sizeof(*state->merge_slots));
    state->model_vocab_count = contract->model_vocab_count;
    state->merge_count = contract->merge_count;
    state->merge_slot_count = merge_slots;
    build->state = state;
    build->contract = contract;
    build->name_slot_count = name_slots;
    return M3_STATUS_OK;
}

void m3_tokenizer_build_dispose(m3_tokenizer_build *build,
                                bool keep_state)
{
    if (build != NULL) {
        if (!keep_state) {
            m3_tokenizer_state_dispose(build->state);
        }
        free(build->names);
        free(build->name_slots);
        m3_byte_buffer_dispose(&build->name_bytes);
        (void)memset(build, 0, sizeof(*build));
    }
}

bool m3_tokenizer_name_lookup(const m3_tokenizer_build *build,
                              const uint8_t *name, size_t length,
                              uint32_t *id)
{
    size_t mask;
    size_t slot;

    if (build == NULL || name == NULL || build->name_slot_count == 0U) {
        return false;
    }
    mask = build->name_slot_count - 1U;
    slot = (size_t)m3_tokenizer_hash_bytes(name, length) & mask;
    for (;;) {
        uint32_t entry_index = build->name_slots[slot];

        if (entry_index == M3_TOKENIZER_EMPTY_SLOT) {
            return false;
        }
        if (entry_index < build->name_count) {
            const m3_vocab_name *entry = &build->names[entry_index];

            if ((size_t)entry->name.length == length &&
                memcmp(build->name_bytes.data + entry->name.offset,
                       name, length) == 0) {
                if (id != NULL) {
                    *id = entry->id;
                }
                return true;
            }
        }
        slot = (slot + 1U) & mask;
    }
}

m3_status m3_tokenizer_name_insert(m3_tokenizer_build *build,
                                    m3_token_span name, uint32_t id,
                                    m3_error *error)
{
    const uint8_t *bytes = build->name_bytes.data + name.offset;
    size_t mask = build->name_slot_count - 1U;
    size_t slot = (size_t)m3_tokenizer_hash_bytes(bytes, name.length) & mask;

    for (;;) {
        uint32_t entry_index = build->name_slots[slot];

        if (entry_index == M3_TOKENIZER_EMPTY_SLOT) {
            if (build->name_count >= build->contract->model_vocab_count) {
                return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                    "BPE vocabulary has too many entries");
            }
            build->names[build->name_count].name = name;
            build->names[build->name_count].id = id;
            build->name_slots[slot] = build->name_count;
            build->name_count += 1U;
            return M3_STATUS_OK;
        }
        {
            const m3_vocab_name *entry = &build->names[entry_index];

            if (entry->name.length == name.length &&
                memcmp(build->name_bytes.data + entry->name.offset,
                       bytes, name.length) == 0) {
                return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                    "duplicate BPE vocabulary token");
            }
        }
        slot = (slot + 1U) & mask;
    }
}

static uint64_t m3_pair_hash(uint32_t left, uint32_t right)
{
    uint64_t value = ((uint64_t)left << 32U) | right;

    value ^= value >> 30U;
    value *= UINT64_C(0xBF58476D1CE4E5B9);
    value ^= value >> 27U;
    value *= UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31U);
}

m3_status m3_tokenizer_merge_insert(m3_tokenizer_state *state,
                                     uint32_t left, uint32_t right,
                                     uint32_t rank, uint32_t output,
                                     m3_error *error)
{
    size_t mask = state->merge_slot_count - 1U;
    size_t slot = (size_t)m3_pair_hash(left, right) & mask;

    for (;;) {
        m3_merge_slot *entry = &state->merge_slots[slot];

        if (entry->left == M3_TOKENIZER_EMPTY_SLOT) {
            entry->left = left;
            entry->right = right;
            entry->rank = rank;
            entry->output = output;
            return M3_STATUS_OK;
        }
        if (entry->left == left && entry->right == right) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "duplicate BPE merge pair");
        }
        slot = (slot + 1U) & mask;
    }
}

const m3_merge_slot *m3_tokenizer_find_merge(
    const m3_tokenizer_state *state, uint32_t left, uint32_t right)
{
    size_t mask;
    size_t slot;

    if (state == NULL || state->merge_slot_count == 0U) {
        return NULL;
    }
    mask = state->merge_slot_count - 1U;
    slot = (size_t)m3_pair_hash(left, right) & mask;
    for (;;) {
        const m3_merge_slot *entry = &state->merge_slots[slot];

        if (entry->left == M3_TOKENIZER_EMPTY_SLOT) {
            return NULL;
        }
        if (entry->left == left && entry->right == right) {
            return entry;
        }
        slot = (slot + 1U) & mask;
    }
}

m3_status m3_tokenizer_state_build(const uint8_t *data, size_t size,
                                    const m3_tokenizer_contract *contract,
                                    m3_tokenizer_state **state,
                                    m3_error *error)
{
    m3_tokenizer_sections tokenizer_sections;
    m3_model_sections model_sections;
    m3_tokenizer_build build = {0};
    m3_status status;

    if (state == NULL || contract == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer state build argument is null");
    }
    *state = NULL;
    status = m3_tokenizer_parse_sections(data, size, &tokenizer_sections,
                                         error);
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_parse_model_sections(
            data, &tokenizer_sections.model, contract, &model_sections,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_build_init(&build, contract, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_parse_vocab(&build, data,
                                          &model_sections.vocab, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_parse_added(&build, data,
                                          &tokenizer_sections.added_tokens,
                                          error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_parse_merges(&build, data,
                                           &model_sections.merges, error);
    }
    if (status == M3_STATUS_OK) {
        *state = build.state;
        m3_tokenizer_build_dispose(&build, true);
        m3_error_reset(error);
    } else {
        m3_tokenizer_build_dispose(&build, false);
    }
    return status;
}
