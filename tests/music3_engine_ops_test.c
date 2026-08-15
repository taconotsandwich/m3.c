/* SPDX-License-Identifier: GPL-2.0-only */

#include "music3_engine_test.h"

#include "m3_flow_internal.h"
#include "m3_waveform_internal.h"
#include "m3_weight_stage_internal.h"

#include <stdlib.h>
#include <string.h>

static m3_status m3_music3_test_child_progress(
    m3_progress_callback progress, void *context, uint64_t total,
    m3_error *error)
{
    uint64_t completed;

    for (completed = 0U; completed <= total; ++completed) {
        if (progress != NULL && !progress(context, completed, total)) {
            return m3_error_set(error, M3_STATUS_CANCELLED,
                                "Music3 test operation was cancelled");
        }
    }
    return M3_STATUS_OK;
}

static bool m3_music3_test_metadata_equal(
    const m3_tensor_metadata *left, const m3_tensor_metadata *right)
{
    uint8_t axis;

    if (left->dtype != right->dtype || left->rank != right->rank ||
        left->element_count != right->element_count ||
        left->byte_count != right->byte_count) {
        return false;
    }
    for (axis = 0U; axis < left->rank; ++axis) {
        if (left->shape[axis] != right->shape[axis]) {
            return false;
        }
    }
    return true;
}

static m3_status m3_music3_test_small_stage(
    m3_music3_test_fixture *fixture, m3_component_id component,
    const m3_weight_table *table, m3_backend *backend,
    m3_weight_stage *stage, m3_error *error)
{
    m3_weight_stage built;
    m3_storage *borrowed = NULL;
    float value = 0.0F;
    size_t index;
    m3_status status = M3_STATUS_OK;

    m3_weight_stage_init(&built);
    built.table = table;
    built.backend = backend;
    built.storage_count = table->shard_count;
    built.view_count = table->binding_count;
    built.loaded_bytes = table->aggregate_payload_bytes;
    built.storages = calloc(
        built.storage_count, sizeof(*built.storages));
    built.views = calloc(built.view_count, sizeof(*built.views));
    if (built.storages == NULL || built.views == NULL) {
        status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                              "cannot allocate Music3 test stage");
        m3_weight_stage_dispose(&built);
        return status;
    }
    for (index = 0U; status == M3_STATUS_OK &&
                    index < built.storage_count; ++index) {
        if (index == 0U &&
            (fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_STAGE ||
             (fixture->corruption ==
                  M3_MUSIC3_TEST_ALIAS_FIRST_STAGE &&
              component == M3_COMPONENT_RVQ_DEPTH_DECODER))) {
            borrowed = fixture->corruption ==
                               M3_MUSIC3_TEST_ALIAS_OLD_STAGE
                           ? fixture->alias_storage
                           : fixture->prior_stage_storage;
            built.storages[index] = borrowed;
        } else {
            status = m3_storage_allocate(
                backend, sizeof(value), 64U, &built.storages[index],
                error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_view_contiguous(
                &built.views[index], built.storages[index], M3_DTYPE_F32,
                1U, table->bindings[index].tensor.shape, 0U, error);
        }
        if (status == M3_STATUS_OK && built.storages[index] != borrowed) {
            status = m3_storage_write(
                built.storages[index], 0U, &value, sizeof(value), error);
        }
    }
    if (status == M3_STATUS_OK &&
        fixture->corruption == M3_MUSIC3_TEST_CORRUPT_STAGE_DUPLICATE) {
        m3_storage_free(built.storages[1]);
        built.storages[1] = built.storages[0];
        m3_tensor_view_init(&built.views[1]);
        status = m3_tensor_view_contiguous(
            &built.views[1], built.storages[1], M3_DTYPE_F32, 1U,
            table->bindings[1].tensor.shape, 0U, error);
    }
    if (status == M3_STATUS_OK && fixture->corruption ==
                                      M3_MUSIC3_TEST_CORRUPT_STAGE_VIEW) {
        built.views[0].byte_offset = 1U;
    }
    if (status != M3_STATUS_OK) {
        for (index = 0U; index < built.storage_count; ++index) {
            if (built.storages[index] == borrowed) {
                built.storages[index] = NULL;
            }
        }
        m3_weight_stage_dispose(&built);
        return status;
    }
    *stage = built;
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_full_vocoder_stage(
    m3_music3_test_fixture *fixture, m3_backend *backend,
    m3_weight_stage *stage, m3_error *error)
{
    m3_vocoder_plan_config config;

    m3_vocoder_test_fixture_dispose(&fixture->vocoder_source);
    m3_vocoder_plan_official_config(&config);
    if (!m3_vocoder_test_fixture_create_config(
            &fixture->vocoder_source, backend, false, &config, error)) {
        return error == NULL ? M3_STATUS_INTERNAL : error->status;
    }
    fixture->vocoder_source.stage.loaded_bytes =
        M3_MUSIC3_STAGE_VOCODER_BYTES;
    *stage = fixture->vocoder_source.stage;
    m3_weight_stage_init(&fixture->vocoder_source.stage);
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_stage(
    void *context, m3_component_id component,
    const m3_weight_table *table, m3_backend *backend,
    m3_progress_callback progress, void *progress_context,
    m3_weight_stage *stage, m3_error *error)
{
    m3_music3_test_fixture *fixture = context;
    uint64_t total = fixture->engine->component_payload_bytes[component];
    m3_status status;

    ++fixture->stage_calls[component];
    if (fixture->corruption == M3_MUSIC3_TEST_CORRUPT_STAGE_PROGRESS) {
        if (progress == NULL ||
            progress(progress_context, 0U, total + 1U)) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Music3 malformed progress was accepted");
        }
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "Music3 malformed progress was rejected");
    }
    if (progress != NULL && !progress(progress_context, 0U, total)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "Music3 test stage was cancelled");
    }
    if (fixture->corruption == M3_MUSIC3_TEST_CORRUPT_STAGE_EMPTY) {
        stage->table = table;
        stage->backend = backend;
        stage->loaded_bytes = table->aggregate_payload_bytes;
        status = M3_STATUS_OK;
    } else {
        status = fixture->real_pipeline && component == M3_COMPONENT_VOCODER
                 ? m3_music3_test_full_vocoder_stage(
                       fixture, backend, stage, error)
                 : m3_music3_test_small_stage(
                       fixture, component, table, backend, stage, error);
    }
    if (status == M3_STATUS_OK &&
        component == M3_COMPONENT_LANGUAGE_MODEL &&
        stage->storage_count != 0U && stage->storages != NULL) {
        fixture->prior_stage_storage = stage->storages[0];
    }
    if (status == M3_STATUS_OK && progress != NULL && total > 1U &&
        !progress(progress_context, 1U, total)) {
        status = m3_error_set(error, M3_STATUS_CANCELLED,
                              "Music3 test stage was cancelled");
    }
    if (status == M3_STATUS_OK && progress != NULL &&
        !progress(progress_context, total, total)) {
        status = m3_error_set(error, M3_STATUS_CANCELLED,
                              "Music3 test stage was cancelled");
    }
    if (status != M3_STATUS_OK) {
        if ((fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_STAGE ||
             fixture->corruption == M3_MUSIC3_TEST_ALIAS_FIRST_STAGE) &&
            stage->storages != NULL &&
            (stage->storages[0] == fixture->alias_storage ||
             stage->storages[0] == fixture->prior_stage_storage)) {
            stage->storages[0] = NULL;
            if (stage->views != NULL) {
                m3_tensor_view_init(&stage->views[0]);
            }
        }
        m3_weight_stage_dispose(stage);
    }
    return status;
}

static m3_status m3_music3_test_vocoder_stage_validate(
    const m3_music3_test_fixture *fixture, const m3_weight_stage *stage,
    m3_backend *backend, m3_error *error)
{
    size_t index;

    if (stage->table != &fixture->vocoder_source.table ||
        stage->backend != backend ||
        stage->storage_count != fixture->vocoder_source.plan.source_count ||
        stage->view_count != fixture->vocoder_source.plan.source_count ||
        stage->loaded_bytes != M3_MUSIC3_STAGE_VOCODER_BYTES ||
        stage->storages == NULL || stage->views == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Music3 test vocoder stage is invalid");
    }
    for (index = 0U; index < stage->storage_count; ++index) {
        const m3_tensor_metadata *expected =
            &fixture->vocoder_source.table.bindings[index].tensor;
        m3_tensor_view checked;
        size_t earlier;

        if (stage->storages[index] == NULL ||
            stage->views[index].storage != stage->storages[index] ||
            m3_storage_backend(stage->storages[index]) != backend ||
            m3_storage_size(stage->storages[index]) !=
                expected->byte_count ||
            !m3_music3_test_metadata_equal(
                &stage->views[index].metadata, expected) ||
            !m3_tensor_is_contiguous(&stage->views[index])) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "Music3 test vocoder source is invalid");
        }
        for (earlier = 0U; earlier < index; ++earlier) {
            if (stage->storages[index] == stage->storages[earlier]) {
                return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                    "Music3 test vocoder owners repeat");
            }
        }
        m3_tensor_view_init(&checked);
        if (m3_tensor_reshape(
                &stage->views[index], expected->rank, expected->shape,
                &checked, error) != M3_STATUS_OK) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "Music3 test vocoder source is bounded");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_stage_validate(
    void *context, m3_component_id component,
    const m3_weight_stage *stage, const m3_weight_table *table,
    m3_backend *backend, m3_error *error)
{
    const m3_music3_test_fixture *fixture = context;

    return fixture->real_pipeline && component == M3_COMPONENT_VOCODER
               ? m3_music3_test_vocoder_stage_validate(
                     fixture, stage, backend, error)
               : m3_weight_stage_validate(stage, table, backend, error);
}

static bool m3_music3_test_prompt_rows(
    m3_music3_test_fixture *fixture, const m3_tensor_view *prompt)
{
    int32_t *values;
    size_t count;
    size_t row;
    size_t caption_end = SIZE_MAX;
    size_t lyrics_start = SIZE_MAX;
    bool valid;
    m3_error error;

    if (prompt == NULL || prompt->metadata.dtype != M3_DTYPE_I32 ||
        prompt->metadata.rank != 2U || prompt->metadata.shape[0] != 2U ||
        !m3_tensor_is_contiguous(prompt)) {
        return false;
    }
    count = prompt->metadata.element_count;
    values = malloc(count * sizeof(*values));
    if (values == NULL || m3_storage_read(
                              prompt->storage, prompt->byte_offset, values,
                              count * sizeof(*values), &error) !=
                              M3_STATUS_OK) {
        free(values);
        return false;
    }
    row = (size_t)prompt->metadata.shape[1];
    valid = row >= 7U && values[0] == M3_TOKEN_IM_START &&
            values[1] == M3_TOKEN_CAPTION_START &&
            values[row - 3U] == M3_TOKEN_LYRICS_END &&
            values[row - 2U] == M3_TOKEN_IM_END &&
            values[row - 1U] == M3_TOKEN_AUDIO_START &&
            values[row] == M3_TOKEN_IM_START &&
            values[2U * row - 2U] == M3_TOKEN_IM_END &&
            values[2U * row - 1U] == M3_TOKEN_AUDIO_START;
    for (count = 0U; count < row; ++count) {
        valid = valid && values[count] >= 0 &&
                (uint32_t)values[count] < M3_TOKENIZER_ID_COUNT;
        if (values[count] == M3_TOKEN_CAPTION_END) {
            caption_end = count;
        }
        if (values[count] == M3_TOKEN_LYRICS_START) {
            lyrics_start = count;
        }
        if (count >= 1U && count + 2U < row) {
            valid = valid && values[row + count] == M3_TOKEN_AUDIO_CFG;
        }
    }
    free(values);
    fixture->prompt_rows_valid =
        valid && caption_end > 1U && lyrics_start == caption_end + 1U &&
        lyrics_start < row - 3U;
    return fixture->prompt_rows_valid;
}

static m3_status m3_music3_test_tensor(
    m3_backend *backend, m3_dtype dtype, uint8_t rank,
    const uint64_t *shape, m3_storage **storage, m3_tensor_view *view,
    m3_error *error)
{
    m3_tensor_metadata metadata;
    m3_status status;

    if (backend == NULL || storage == NULL || view == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 test tensor outputs are invalid");
    }
    status = m3_tensor_metadata_init(
        &metadata, dtype, rank, shape, error);

    if (status == M3_STATUS_OK) {
        status = m3_storage_allocate(
            backend, metadata.byte_count, 64U, storage, error);
    }
    if (status == M3_STATUS_OK && *storage == NULL) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 test storage was not published");
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            view, *storage, dtype, rank, shape, 0U, error);
    }
    return status;
}

static m3_status m3_music3_test_semantic_script(
    m3_music3_test_fixture *fixture, m3_backend *backend,
    uint64_t frame_limit, m3_progress_callback progress,
    void *progress_context, m3_semantic_output *output,
    m3_error *error)
{
    m3_semantic_output built;
    m3_storage *borrowed = NULL;
    uint64_t frames = fixture->scripted_frames;
    uint64_t shape[4];
    uint64_t total;
    m3_status status;

    if (fixture->corruption == M3_MUSIC3_TEST_CORRUPT_SEMANTIC_LIMIT) {
        frames = frame_limit + 1U;
    }
    shape[0] = 1U;
    shape[1] = frames;
    shape[2] = M3_RVQ_CODEBOOK_COUNT;
    shape[3] = M3_QWEN_HIDDEN_SIZE;
    m3_semantic_output_init(&built);
    if (fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_SEMANTIC) {
        borrowed = fixture->alias_storage;
        built.storage = borrowed;
        status = M3_STATUS_OK;
    } else if (fixture->corruption ==
        M3_MUSIC3_TEST_CORRUPT_SEMANTIC_OVERSIZE) {
        m3_tensor_metadata metadata;

        status = m3_tensor_metadata_init(
            &metadata, M3_DTYPE_BF16, 4U, shape, error);
        if (status == M3_STATUS_OK) {
            status = m3_storage_allocate(
                backend, metadata.byte_count + 64U, 64U,
                &built.storage, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_view_contiguous(
                &built.frame_hiddens, built.storage, M3_DTYPE_BF16, 4U,
                shape, 0U, error);
        }
    } else {
        status = m3_music3_test_tensor(
            backend, M3_DTYPE_BF16, 4U, shape, &built.storage,
            &built.frame_hiddens, error);
    }
    total = 49U * frame_limit + 48U;
    if (status == M3_STATUS_OK) {
        status = m3_music3_test_child_progress(
            progress, progress_context, total, error);
    }
    if (status != M3_STATUS_OK) {
        if (built.storage == borrowed) {
            built.storage = NULL;
            m3_tensor_view_init(&built.frame_hiddens);
        }
        m3_semantic_output_dispose(&built);
        return status;
    }
    *output = built;
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_semantic(
    void *context, const m3_weight_stage *language_model,
    const m3_weight_stage *rvq, const m3_tensor_view *prompt_ids,
    uint64_t frame_limit, m3_rng *rng, m3_progress_callback progress,
    void *progress_context, m3_semantic_output *output,
    m3_error *error)
{
    m3_music3_test_fixture *fixture = context;

    (void)language_model;
    (void)rvq;
    ++fixture->semantic_calls;
    fixture->last_decoded_storage = NULL;
    if (!m3_music3_test_prompt_rows(fixture, prompt_ids)) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 prompt row order is invalid");
    }
    if (!fixture->real_pipeline) {
        m3_rng local = *rng;
        uint32_t consumed;
        m3_status status;

        fixture->semantic_rng_entry = local;
        status = m3_rng_u32(&local, &consumed, error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_test_semantic_script(
            fixture, fixture->engine->backend, frame_limit, progress,
            progress_context, output, error);
        }
        fixture->semantic_rng_exit = local;
        fixture->semantic_rng_observed = true;
        if (status == M3_STATUS_OK) {
            *rng = local;
        }
        (void)consumed;
        return status;
    }
    m3_semantic_test_outcomes(&fixture->semantic, 2U, SIZE_MAX);
    return m3_semantic_generate_core(
        &fixture->semantic.operations, prompt_ids, frame_limit, rng,
        progress, progress_context, output, error);
}

static m3_status m3_music3_test_flow_script(
    m3_music3_test_fixture *fixture, m3_backend *backend,
    const m3_tensor_view *frames, m3_progress_callback progress,
    void *progress_context, m3_flow_output *output, m3_error *error)
{
    m3_flow_config config;
    m3_flow_output built;
    m3_storage *borrowed = NULL;
    size_t count = 0U;
    size_t index;
    uint64_t total;
    m3_status status;

    m3_flow_config_init(&config);
    m3_flow_output_init(&built);
    status = m3_flow_chunk_count(
        &config, frames->metadata.shape[1], &count, error);
    if (status == M3_STATUS_OK) {
        built.storages = calloc(count, sizeof(*built.storages));
        built.chunks = calloc(count, sizeof(*built.chunks));
        built.chunk_count = count;
        if (built.storages == NULL || built.chunks == NULL) {
            (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                               "cannot allocate test flow output");
            m3_flow_output_dispose(&built);
            return M3_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status != M3_STATUS_OK) {
        m3_flow_output_dispose(&built);
        return status;
    }
    for (index = 0U; status == M3_STATUS_OK && index < count; ++index) {
        uint64_t start = 0U;
        uint64_t window = 0U;
        uint64_t length = 0U;
        uint64_t shape[3];

        status = m3_flow_chunk_window(
            &config, frames->metadata.shape[1], index, &start, &window,
            error);
        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                window, config.condition.resize_numerator,
                config.condition.resize_denominator, &length, error);
        }
        if (status == M3_STATUS_OK && index == 0U &&
            fixture->corruption == M3_MUSIC3_TEST_CORRUPT_FLOW_LENGTH) {
            ++length;
        }
        shape[0] = 1U;
        shape[1] = M3_FLOW_LATENT_CHANNELS;
        shape[2] = length;
        if (status == M3_STATUS_OK && index == 0U &&
            fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_FLOW) {
            borrowed = fixture->alias_storage;
            built.storages[index] = borrowed;
        } else if (status == M3_STATUS_OK) {
            status = m3_music3_test_tensor(
                backend, M3_DTYPE_F32, 3U, shape,
                &built.storages[index], &built.chunks[index], error);
        }
        (void)start;
    }
    if (status == M3_STATUS_OK && count > 1U &&
        fixture->corruption == M3_MUSIC3_TEST_CORRUPT_FLOW_DUPLICATE) {
        m3_storage_free(built.storages[1]);
        built.storages[1] = built.storages[0];
        built.chunks[1] = built.chunks[0];
    }
    total = (uint64_t)count * M3_FLOW_INFERENCE_STEPS;
    if (status == M3_STATUS_OK) {
        status = m3_music3_test_child_progress(
            progress, progress_context, total, error);
    }
    if (status != M3_STATUS_OK) {
        for (index = 0U; index < count; ++index) {
            if (built.storages[index] == borrowed) {
                built.storages[index] = NULL;
                m3_tensor_view_init(&built.chunks[index]);
            }
        }
        m3_flow_output_dispose(&built);
        return status;
    }
    *output = built;
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_flow(
    void *context, m3_backend *backend,
    const m3_weight_stage *condition, const m3_weight_stage *flow,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *output, m3_error *error)
{
    m3_music3_test_fixture *fixture = context;
    uint64_t shape[4];
    size_t strides[4];
    m3_tensor_view reduced;
    m3_status status;

    (void)condition;
    (void)flow;
    ++fixture->flow_calls;
    if (!fixture->real_pipeline) {
        m3_rng local = *rng;
        float consumed;

        fixture->flow_rng_entry = local;
        status = m3_rng_uniform_f32(&local, &consumed, error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_test_flow_script(
            fixture, backend, frame_hiddens, progress, progress_context,
            output, error);
        }
        fixture->flow_rng_exit = local;
        fixture->flow_rng_observed = true;
        if (status == M3_STATUS_OK) {
            *rng = local;
        }
        (void)consumed;
        return status;
    }
    shape[0] = 1U;
    shape[1] = frame_hiddens->metadata.shape[1];
    shape[2] = fixture->flow.config.condition.layer_count;
    shape[3] = fixture->flow.config.condition.hidden_size;
    (void)memcpy(strides, frame_hiddens->byte_strides, sizeof(strides));
    m3_tensor_view_init(&reduced);
    status = m3_tensor_view_strided(
        &reduced, frame_hiddens->storage, M3_DTYPE_BF16, 4U, shape,
        strides, frame_hiddens->byte_offset, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    return m3_flow_synthesize_core(
        backend, &fixture->flow.config, &fixture->flow.weights,
        &fixture->flow.condition_weights, &reduced, rng, progress,
        progress_context, output, error);
}

void m3_music3_test_operations(
    m3_music3_test_fixture *fixture, m3_music3_operations *operations)
{
    *operations = *m3_music3_production_operations();
    operations->context = fixture;
    operations->stage_weights = m3_music3_test_stage;
    operations->stage_validate = m3_music3_test_stage_validate;
    operations->semantic_generate = m3_music3_test_semantic;
    operations->flow_synthesize = m3_music3_test_flow;
    m3_music3_test_vocoder_operations(operations);
}
