/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MUSIC3_ENGINE_TEST_H
#define M3_MUSIC3_ENGINE_TEST_H

#include "flow_runtime_test.h"
#include "m3_music3_internal.h"
#include "semantic_runtime_test.h"
#include "tokenizer_fixture.h"
#include "vocoder_runtime_test.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_MUSIC3_TEST_PROGRESS_CAPACITY 32768U

typedef enum {
    M3_MUSIC3_TEST_CORRUPT_NONE = 0,
    M3_MUSIC3_TEST_CORRUPT_STAGE_EMPTY,
    M3_MUSIC3_TEST_CORRUPT_STAGE_VIEW,
    M3_MUSIC3_TEST_CORRUPT_STAGE_DUPLICATE,
    M3_MUSIC3_TEST_CORRUPT_STAGE_PROGRESS,
    M3_MUSIC3_TEST_CORRUPT_SEMANTIC_LIMIT,
    M3_MUSIC3_TEST_CORRUPT_SEMANTIC_OVERSIZE,
    M3_MUSIC3_TEST_CORRUPT_FLOW_LENGTH,
    M3_MUSIC3_TEST_CORRUPT_FLOW_DUPLICATE,
    M3_MUSIC3_TEST_CORRUPT_RUNTIME_VIEW,
    M3_MUSIC3_TEST_CORRUPT_RUNTIME_DUPLICATE,
    M3_MUSIC3_TEST_CORRUPT_DECODE_LENGTH,
    M3_MUSIC3_TEST_CORRUPT_DECODE_DUPLICATE,
    M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_SHORT,
    M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_LONG,
    M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_OVERSIZE,
    M3_MUSIC3_TEST_ALIAS_OLD_STAGE,
    M3_MUSIC3_TEST_ALIAS_OLD_SEMANTIC,
    M3_MUSIC3_TEST_ALIAS_OLD_FLOW,
    M3_MUSIC3_TEST_ALIAS_OLD_RUNTIME,
    M3_MUSIC3_TEST_ALIAS_OLD_DECODE,
    M3_MUSIC3_TEST_ALIAS_OLD_ASSEMBLE,
    M3_MUSIC3_TEST_ALIAS_FIRST_STAGE
} m3_music3_test_corruption;

typedef struct {
    m3_music3_phase phase[M3_MUSIC3_TEST_PROGRESS_CAPACITY];
    uint64_t completed[M3_MUSIC3_TEST_PROGRESS_CAPACITY];
    uint64_t total[M3_MUSIC3_TEST_PROGRESS_CAPACITY];
    size_t count;
    m3_music3_phase cancel_phase;
    uint64_t cancel_completed;
    bool cancel_enabled;
    bool contract_valid;
} m3_music3_test_progress;

typedef struct {
    m3_music3_engine *engine;
    m3_tokenizer_fixture tokenizer_data;
    m3_semantic_test_fixture semantic;
    m3_flow_test_fixture flow;
    m3_vocoder_test_fixture vocoder_source;
    bool semantic_ready;
    bool flow_ready;
    bool real_pipeline;
    bool prompt_rows_valid;
    uint64_t scripted_frames;
    m3_music3_test_corruption corruption;
    m3_storage *alias_storage;
    m3_storage *prior_stage_storage;
    m3_storage *last_decoded_storage;
    size_t stage_calls[M3_MUSIC3_WEIGHT_COMPONENT_COUNT];
    size_t semantic_calls;
    size_t flow_calls;
    size_t materialize_calls;
    size_t runtime_validate_calls;
    size_t decode_calls;
    size_t assemble_calls;
    m3_rng semantic_rng_entry;
    m3_rng semantic_rng_exit;
    m3_rng flow_rng_entry;
    m3_rng flow_rng_exit;
    bool semantic_rng_observed;
    bool flow_rng_observed;
} m3_music3_test_fixture;

m3_status m3_music3_test_fixture_create(
    m3_music3_test_fixture *fixture, bool real_pipeline,
    m3_error *error);
void m3_music3_test_fixture_dispose(m3_music3_test_fixture *fixture);
void m3_music3_test_request(m3_music3_request *request,
                            uint64_t maximum_frames);
void m3_music3_test_progress_init(m3_music3_test_progress *progress);
bool m3_music3_test_progress_call(
    void *context, m3_music3_phase phase, uint64_t completed,
    uint64_t total);
bool m3_music3_test_phase_reached(
    const m3_music3_test_progress *progress, m3_music3_phase phase,
    uint64_t completed);

void m3_music3_test_operations(
    m3_music3_test_fixture *fixture, m3_music3_operations *operations);
void m3_music3_test_vocoder_operations(
    m3_music3_operations *operations);

#endif
