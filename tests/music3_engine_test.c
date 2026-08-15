/* SPDX-License-Identifier: GPL-2.0-only */

#include "music3_engine_test.h"

#include "m3_music3_schema.h"
#include "tokenizer_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *m3_music3_test_copy(const char *text)
{
    size_t length = strlen(text);
    char *copy = malloc(length + 1U);

    if (copy != NULL) {
        (void)memcpy(copy, text, length + 1U);
    }
    return copy;
}

static bool m3_music3_test_table(
    m3_weight_table *table, size_t component, m3_error *error)
{
    const uint64_t shape[] = {1U};
    char path[32];
    char name[32];
    int path_length;
    int name_length;
    size_t index;

    m3_weight_table_init(table);
    table->shards = calloc(2U, sizeof(*table->shards));
    table->bindings = calloc(2U, sizeof(*table->bindings));
    if (table->shards == NULL || table->bindings == NULL) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                           "cannot allocate Music3 test table");
        return false;
    }
    table->shard_count = 2U;
    table->binding_count = 2U;
    for (index = 0U; index < 2U; ++index) {
        path_length = snprintf(
            path, sizeof(path), "component-%zu-%zu", component, index);
        name_length = snprintf(
            name, sizeof(name), "weight-%zu-%zu", component, index);
        if (path_length <= 0 || (size_t)path_length >= sizeof(path) ||
            name_length <= 0 || (size_t)name_length >= sizeof(name)) {
            (void)m3_error_set(error, M3_STATUS_INTERNAL,
                               "cannot name Music3 test table");
            return false;
        }
        table->shards[index].path = m3_music3_test_copy(path);
        table->bindings[index].name = m3_music3_test_copy(name);
        if (table->shards[index].path == NULL ||
            table->bindings[index].name == NULL ||
            m3_tensor_metadata_init(
                &table->bindings[index].tensor, M3_DTYPE_F32, 1U, shape,
                error) != M3_STATUS_OK) {
            return false;
        }
        table->shards[index].payload_bytes = sizeof(float);
        table->bindings[index].shard_index = index;
        table->bindings[index].data_start = 0U;
        table->bindings[index].data_end = sizeof(float);
    }
    table->aggregate_payload_bytes = 2U * sizeof(float);
    return true;
}

static bool m3_music3_test_tables(
    m3_music3_test_fixture *fixture, m3_error *error)
{
    size_t index;

    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT; ++index) {
        m3_music3_schema_summary expected;

        if (!m3_music3_test_table(
                &fixture->engine->tables[index], index, error) ||
            m3_music3_schema_expected_summary(
                (m3_component_id)index, &expected, error) !=
                M3_STATUS_OK) {
            return false;
        }
        fixture->engine->component_payload_bytes[index] =
            expected.payload_bytes;
        fixture->engine->component_largest_shard_bytes[index] =
            expected.payload_bytes;
    }
    return true;
}

m3_status m3_music3_test_fixture_create(
    m3_music3_test_fixture *fixture, bool real_pipeline,
    m3_error *error)
{
    m3_status status;

    if (fixture == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 test fixture is null");
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    fixture->real_pipeline = real_pipeline;
    fixture->scripted_frames = 1U;
    fixture->engine = calloc(1U, sizeof(*fixture->engine));
    if (fixture->engine == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Music3 test engine");
    }
    status = real_pipeline
                 ? m3_backend_create_metal(&fixture->engine->backend, error)
                 : m3_backend_create_host(&fixture->engine->backend, error);
    if (status != M3_STATUS_OK) {
        m3_music3_test_fixture_dispose(fixture);
        return status;
    }
    m3_tokenizer_init(&fixture->engine->tokenizer);
    if (!m3_test_tokenizer_fixture_load(
            &fixture->engine->tokenizer, &fixture->tokenizer_data,
            M3_TOKENIZER_FIXTURE_VALID, error) ||
        !m3_music3_test_tables(fixture, error)) {
        status = error == NULL ? M3_STATUS_INTERNAL : error->status;
        m3_music3_test_fixture_dispose(fixture);
        return status;
    }
    if (real_pipeline) {
        fixture->semantic_ready = m3_semantic_test_fixture_init(
            &fixture->semantic, fixture->engine->backend);
        fixture->flow_ready = fixture->semantic_ready &&
            m3_flow_test_fixture_init_music3(
                &fixture->flow, 1U, fixture->engine->backend);
        if (!fixture->flow_ready) {
            m3_music3_test_fixture_dispose(fixture);
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "cannot build real Music3 test cores");
        }
    }
    m3_music3_test_operations(
        fixture, &fixture->engine->operations);
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_music3_test_fixture_dispose(m3_music3_test_fixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    m3_vocoder_test_fixture_dispose(&fixture->vocoder_source);
    if (fixture->flow_ready) {
        m3_flow_test_fixture_dispose(&fixture->flow);
    }
    if (fixture->semantic_ready) {
        m3_semantic_test_fixture_dispose(&fixture->semantic);
    }
    m3_music3_engine_free(fixture->engine);
    m3_tokenizer_fixture_dispose(&fixture->tokenizer_data);
    (void)memset(fixture, 0, sizeof(*fixture));
}

void m3_music3_test_request(m3_music3_request *request,
                            uint64_t maximum_frames)
{
    static const char caption[] = "quiet instrumental";
    static const char lyrics[] = "one clear lyric line";

    *request = (m3_music3_request){
        {caption, sizeof(caption) - 1U},
        {lyrics, sizeof(lyrics) - 1U}, maximum_frames,
        UINT64_C(0x123456789abcdef0), UINT64_C(0x1020304050607080)};
}

void m3_music3_test_progress_init(m3_music3_test_progress *progress)
{
    (void)memset(progress, 0, sizeof(*progress));
    progress->contract_valid = true;
}

bool m3_music3_test_progress_call(
    void *context, m3_music3_phase phase, uint64_t completed,
    uint64_t total)
{
    m3_music3_test_progress *progress = context;

    if (progress->count >= M3_MUSIC3_TEST_PROGRESS_CAPACITY ||
        phase >= M3_MUSIC3_PHASE_COUNT || completed > total) {
        progress->contract_valid = false;
        return false;
    }
    if (progress->count != 0U) {
        size_t prior = progress->count - 1U;

        if (phase < progress->phase[prior] ||
            (phase == progress->phase[prior] &&
             (total != progress->total[prior] ||
              completed <= progress->completed[prior])) ||
            (phase > progress->phase[prior] &&
             (phase != progress->phase[prior] + 1 || completed != 0U))) {
            progress->contract_valid = false;
        }
    } else if (phase != M3_MUSIC3_PHASE_PREPARE || completed != 0U) {
        progress->contract_valid = false;
    }
    progress->phase[progress->count] = phase;
    progress->completed[progress->count] = completed;
    progress->total[progress->count] = total;
    ++progress->count;
    return !(progress->cancel_enabled && phase == progress->cancel_phase &&
             completed == progress->cancel_completed);
}

bool m3_music3_test_phase_reached(
    const m3_music3_test_progress *progress, m3_music3_phase phase,
    uint64_t completed)
{
    size_t index;

    for (index = 0U; index < progress->count; ++index) {
        if (progress->phase[index] == phase &&
            progress->completed[index] == completed) {
            return true;
        }
    }
    return false;
}
