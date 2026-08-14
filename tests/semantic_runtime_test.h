/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_SEMANTIC_RUNTIME_TEST_H
#define M3_SEMANTIC_RUNTIME_TEST_H

#include "m3_semantic_internal.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_SEMANTIC_TEST_STATE_CAPACITY 8U
#define M3_SEMANTIC_TEST_UNIFORM_CAPACITY 64U
#define M3_SEMANTIC_TEST_PROGRESS_CAPACITY 512U

typedef enum {
    M3_SEMANTIC_TEST_FAIL_NONE = 0,
    M3_SEMANTIC_TEST_FAIL_START,
    M3_SEMANTIC_TEST_FAIL_PREFILL,
    M3_SEMANTIC_TEST_FAIL_EMBEDDING,
    M3_SEMANTIC_TEST_FAIL_DECODE,
    M3_SEMANTIC_TEST_FAIL_FEEDBACK,
    M3_SEMANTIC_TEST_FAIL_ADVANCE
} m3_semantic_test_failure;

typedef struct {
    uint64_t values[M3_SEMANTIC_TEST_PROGRESS_CAPACITY];
    size_t count;
    uint64_t cancel_at;
    uint64_t total;
    bool total_stable;
} m3_semantic_test_progress;

typedef struct {
    m3_backend *backend;
    m3_storage *prompt_storage;
    m3_storage *hidden_storage;
    m3_storage *eos_storage;
    m3_storage *semantic_storage;
    m3_storage *embedding_storage;
    m3_tensor_view prompt;
    m3_tensor_view hidden;
    m3_tensor_view eos;
    m3_tensor_view semantic;
    m3_tensor_view embedding;
    m3_semantic_operations operations;
    bool eos_states[M3_SEMANTIC_TEST_STATE_CAPACITY];
    uint32_t semantic_codes[M3_SEMANTIC_TEST_STATE_CAPACITY];
    uint32_t guidance_code_a;
    uint32_t guidance_code_b;
    size_t state_count;
    size_t state_index;
    size_t start_calls;
    size_t finish_calls;
    size_t prefill_calls;
    size_t embedding_calls;
    size_t decode_calls;
    size_t feedback_calls;
    size_t advance_calls;
    uint64_t cache_capacity;
    uint32_t embedded_codes[M3_SEMANTIC_TEST_STATE_CAPACITY];
    float residual_uniforms[M3_SEMANTIC_TEST_UNIFORM_CAPACITY];
    size_t residual_uniform_count;
    bool adversarial_guidance;
    bool swap_guidance_rows;
    bool contract_valid;
    m3_semantic_test_failure failure;
} m3_semantic_test_fixture;

bool m3_semantic_test_fixture_init(m3_semantic_test_fixture *fixture,
                                   m3_backend *backend);
void m3_semantic_test_fixture_dispose(
    m3_semantic_test_fixture *fixture);
void m3_semantic_test_outcomes(m3_semantic_test_fixture *fixture,
                               size_t state_count, size_t eos_index);
void m3_semantic_test_guidance_rows(
    m3_semantic_test_fixture *fixture, uint32_t code_a,
    uint32_t code_b, bool swapped);
bool m3_semantic_test_progress_callback(void *context,
                                        uint64_t completed,
                                        uint64_t total);
bool m3_semantic_test_expected_rng(const m3_rng *initial,
                                   size_t uniform_count,
                                   m3_rng *expected);
bool m3_semantic_test_output_bits(const m3_semantic_output *output,
                                  uint16_t *bits, size_t count);
bool m3_semantic_test_seed_output(m3_backend *backend,
                                  uint16_t value,
                                  m3_semantic_output *output);

#endif
