/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include <stdlib.h>

typedef struct {
    m3_music3_prepared_prompt prompt;
    m3_weight_stage language_model;
    m3_weight_stage rvq;
    m3_weight_stage condition;
    m3_weight_stage flow_stage;
    m3_weight_stage vocoder_stage;
    m3_semantic_output semantic;
    m3_flow_output latents;
    m3_vocoder_runtime *vocoder;
    m3_vocoder_output *decoded;
    m3_tensor_view *decoded_views;
    size_t decoded_count;
    m3_music3_output *built_output;
    m3_storage *prior_output_storage;
} m3_music3_run;

static void m3_music3_run_init(m3_music3_run *run,
                               m3_storage *prior_output_storage)
{
    m3_music3_prepared_prompt_init(&run->prompt);
    m3_weight_stage_init(&run->language_model);
    m3_weight_stage_init(&run->rvq);
    m3_weight_stage_init(&run->condition);
    m3_weight_stage_init(&run->flow_stage);
    m3_weight_stage_init(&run->vocoder_stage);
    m3_semantic_output_init(&run->semantic);
    m3_flow_output_init(&run->latents);
    run->vocoder = NULL;
    run->decoded = NULL;
    run->decoded_views = NULL;
    run->decoded_count = 0U;
    run->built_output = NULL;
    run->prior_output_storage = prior_output_storage;
}

static void m3_music3_run_dispose(m3_music3_engine *engine,
                                  m3_music3_run *run)
{
    size_t index;

    m3_music3_output_free(run->built_output);
    for (index = 0U; index < run->decoded_count; ++index) {
        size_t later;

        for (later = index + 1U; later < run->decoded_count; ++later) {
            if (run->decoded[later].storage ==
                run->decoded[index].storage) {
                run->decoded[later].storage = NULL;
            }
        }
        m3_vocoder_output_dispose(&run->decoded[index]);
    }
    free(run->decoded_views);
    free(run->decoded);
    if (run->vocoder != NULL) {
        engine->operations.vocoder_free(
            engine->operations.context, run->vocoder);
    }
    m3_flow_output_dispose(&run->latents);
    m3_semantic_output_dispose(&run->semantic);
    m3_weight_stage_dispose(&run->vocoder_stage);
    m3_weight_stage_dispose(&run->flow_stage);
    m3_weight_stage_dispose(&run->condition);
    m3_weight_stage_dispose(&run->rvq);
    m3_weight_stage_dispose(&run->language_model);
    m3_music3_prepared_prompt_dispose(&run->prompt);
}

static m3_status m3_music3_result_validation(
    m3_status status, const char *result, m3_error *error)
{
    if (status == M3_STATUS_OK || status == M3_STATUS_OUT_OF_MEMORY) {
        return status;
    }
    return m3_error_set(error, M3_STATUS_INTERNAL,
                        "Music3 %s result is invalid", result);
}

static m3_status m3_music3_phase_total(
    const m3_music3_engine *engine, m3_component_id first,
    m3_component_id last, uint64_t expected, uint64_t *total,
    m3_error *error)
{
    m3_component_id component;
    uint64_t built = 0U;
    m3_status status = M3_STATUS_OK;

    for (component = first; component <= last && status == M3_STATUS_OK;
         component = (m3_component_id)((int)component + 1)) {
        status = m3_music3_checked_add(
            built, engine->component_payload_bytes[(size_t)component],
            &built, error);
    }
    if (status == M3_STATUS_OK && built != expected) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Music3 stage byte total is not official");
    }
    if (status == M3_STATUS_OK) {
        *total = built;
    }
    return status;
}

static m3_status m3_music3_stage_one(
    m3_music3_engine *engine, m3_component_id component,
    uint64_t offset, m3_music3_progress_state *progress,
    m3_weight_stage *stage, const m3_storage *first,
    const m3_storage *second, const m3_weight_stage *prior_stage,
    const m3_flow_output *prior_flow,
    m3_error *error)
{
    const m3_weight_table *table = &engine->tables[(size_t)component];
    m3_music3_progress_bridge bridge;
    m3_status status;

    m3_music3_progress_bridge_init(
        &bridge, progress, offset,
        engine->component_payload_bytes[(size_t)component]);
    status = engine->operations.stage_weights(
        engine->operations.context, component, table, engine->backend,
        m3_music3_progress_bridge_report, &bridge, stage, error);
    if (status == M3_STATUS_OK) {
        status = m3_music3_stage_ownership_validate(
            stage, first, second, prior_stage, prior_flow, error);
    }
    if (status == M3_STATUS_OK) {
        status = engine->operations.stage_validate(
            engine->operations.context, component, stage, table,
            engine->backend, error);
        status = m3_music3_result_validation(status, "weight stage",
                                             error);
    }
    return m3_music3_progress_bridge_complete(&bridge, status, error);
}

static m3_status m3_music3_prepare_phase(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_run *run, m3_rng *rng,
    m3_music3_progress_state *progress, m3_error *error)
{
    m3_status status = m3_music3_progress_begin(
        progress, M3_MUSIC3_PHASE_PREPARE, 1U, error);

    if (status == M3_STATUS_OK) {
        status = m3_rng_seed(rng, request->seed, request->sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_prepare_prompt(engine, request, &run->prompt,
                                          error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    return status;
}

static m3_status m3_music3_semantic_phases(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_run *run, m3_rng *rng,
    m3_music3_progress_state *progress, m3_error *error)
{
    m3_music3_progress_bridge bridge;
    uint64_t stage_total = 0U;
    m3_status status = m3_music3_phase_total(
        engine, M3_COMPONENT_LANGUAGE_MODEL,
        M3_COMPONENT_RVQ_DEPTH_DECODER,
        M3_MUSIC3_STAGE_SEMANTIC_BYTES, &stage_total, error);

    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_begin(
            progress, M3_MUSIC3_PHASE_STAGE_SEMANTIC, stage_total, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_stage_one(
            engine, M3_COMPONENT_LANGUAGE_MODEL, 0U, progress,
            &run->language_model, run->prior_output_storage,
            run->prompt.storage, NULL, NULL, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_stage_one(
            engine, M3_COMPONENT_RVQ_DEPTH_DECODER,
            engine->component_payload_bytes[
                M3_COMPONENT_LANGUAGE_MODEL],
            progress, &run->rvq, run->prior_output_storage,
            run->prompt.storage, &run->language_model, NULL, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_begin(
            progress, M3_MUSIC3_PHASE_SEMANTIC,
            run->prompt.semantic_plan.progress_total, error);
    }
    if (status == M3_STATUS_OK) {
        m3_music3_progress_bridge_init(
            &bridge, progress, 0U,
            run->prompt.semantic_plan.progress_total);
        status = engine->operations.semantic_generate(
            engine->operations.context, &run->language_model, &run->rvq,
            &run->prompt.ids, request->maximum_frames, rng,
            m3_music3_progress_bridge_report, &bridge, &run->semantic,
            error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_semantic_ownership_validate(
                &run->semantic, &run->prompt, &run->language_model,
                &run->rvq, run->prior_output_storage, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_semantic_output_validate(
                engine, &run->semantic, request->maximum_frames, error);
            status = m3_music3_result_validation(
                status, "semantic", error);
        }
        status = m3_music3_progress_bridge_complete(
            &bridge, status, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        m3_music3_prepared_prompt_dispose(&run->prompt);
        m3_weight_stage_dispose(&run->rvq);
        m3_weight_stage_dispose(&run->language_model);
    }
    return status;
}

static m3_status m3_music3_flow_phases(
    m3_music3_engine *engine, m3_music3_run *run, m3_rng *rng,
    m3_music3_progress_state *progress, m3_error *error)
{
    m3_music3_progress_bridge bridge;
    uint64_t stage_total = 0U;
    uint64_t flow_total = 0U;
    size_t chunk_count = 0U;
    uint64_t frames = run->semantic.frame_hiddens.metadata.shape[1];
    m3_status status = m3_music3_flow_progress_plan(
        frames, &chunk_count, &flow_total, error);

    if (status == M3_STATUS_OK) {
        status = m3_music3_phase_total(
            engine, M3_COMPONENT_CONDITION_ENCODER,
            M3_COMPONENT_TRANSFORMER, M3_MUSIC3_STAGE_FLOW_BYTES,
            &stage_total, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_begin(
            progress, M3_MUSIC3_PHASE_STAGE_FLOW, stage_total, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_preflight_flow(
            engine, &run->semantic.frame_hiddens, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_stage_one(
            engine, M3_COMPONENT_CONDITION_ENCODER, 0U, progress,
            &run->condition, run->prior_output_storage,
            run->semantic.storage, NULL, NULL, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_stage_one(
            engine, M3_COMPONENT_TRANSFORMER,
            engine->component_payload_bytes[
                M3_COMPONENT_CONDITION_ENCODER],
            progress, &run->flow_stage, run->prior_output_storage,
            run->semantic.storage, &run->condition, NULL, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_begin(
            progress, M3_MUSIC3_PHASE_FLOW, flow_total, error);
    }
    if (status == M3_STATUS_OK) {
        m3_music3_progress_bridge_init(
            &bridge, progress, 0U, flow_total);
        status = engine->operations.flow_synthesize(
            engine->operations.context, engine->backend, &run->condition,
            &run->flow_stage, &run->semantic.frame_hiddens, rng,
            m3_music3_progress_bridge_report, &bridge, &run->latents,
            error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_flow_ownership_validate(
                &run->latents, &run->semantic, &run->condition,
                &run->flow_stage, run->prior_output_storage, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_flow_output_validate(
                engine, &run->latents, frames, chunk_count, error);
            status = m3_music3_result_validation(status, "flow", error);
        }
        status = m3_music3_progress_bridge_complete(
            &bridge, status, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        m3_weight_stage_dispose(&run->flow_stage);
        m3_weight_stage_dispose(&run->condition);
        m3_semantic_output_dispose(&run->semantic);
    }
    return status;
}

static m3_status m3_music3_vocoder_stage_phases(
    m3_music3_engine *engine, m3_music3_run *run, uint64_t frame_count,
    m3_music3_progress_state *progress, m3_error *error)
{
    m3_music3_progress_bridge bridge;
    uint64_t stage_total = 0U;
    m3_status status = m3_music3_phase_total(
        engine, M3_COMPONENT_VOCODER, M3_COMPONENT_VOCODER,
        M3_MUSIC3_STAGE_VOCODER_BYTES, &stage_total, error);

    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_begin(
            progress, M3_MUSIC3_PHASE_STAGE_VOCODER, stage_total, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_preflight_vocoder(
            engine, &run->latents, frame_count, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_stage_one(
            engine, M3_COMPONENT_VOCODER, 0U, progress,
            &run->vocoder_stage, run->prior_output_storage, NULL, NULL,
            &run->latents, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_begin(
            progress, M3_MUSIC3_PHASE_MATERIALIZE_VOCODER,
            M3_VOCODER_SOURCE_WEIGHT_COUNT, error);
    }
    if (status == M3_STATUS_OK) {
        m3_music3_progress_bridge_init(
            &bridge, progress, 0U, M3_VOCODER_SOURCE_WEIGHT_COUNT);
        status = engine->operations.vocoder_create(
            engine->operations.context, &run->vocoder_stage,
            m3_music3_progress_bridge_report, &bridge, &run->vocoder,
            error);
        if (status == M3_STATUS_OK && run->vocoder == NULL) {
            status = m3_error_set(error, M3_STATUS_INTERNAL,
                                  "Music3 vocoder was not published");
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_runtime_ownership_validate(
                run->vocoder, &run->vocoder_stage, &run->latents,
                run->prior_output_storage, error);
        }
        if (status == M3_STATUS_OK) {
            status = engine->operations.vocoder_validate(
                engine->operations.context, run->vocoder,
                engine->backend, error);
            status = m3_music3_result_validation(
                status, "vocoder runtime", error);
        }
        status = m3_music3_progress_bridge_complete(
            &bridge, status, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        m3_weight_stage_dispose(&run->vocoder_stage);
    }
    return status;
}

static m3_status m3_music3_decode_phase(
    m3_music3_engine *engine, m3_music3_run *run,
    m3_music3_progress_state *progress, m3_error *error)
{
    m3_vocoder_output *decoded = NULL;
    m3_tensor_view *decoded_views = NULL;
    uint64_t total;
    size_t index;
    m3_status status;

    if (run->latents.chunk_count > SIZE_MAX / sizeof(*run->decoded) ||
        run->latents.chunk_count > SIZE_MAX / sizeof(*run->decoded_views) ||
        (uint64_t)run->latents.chunk_count >
            UINT64_MAX / M3_VOCODER_DECODE_OPERATION_COUNT) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 decode plan overflows");
    }
    total = (uint64_t)run->latents.chunk_count *
            M3_VOCODER_DECODE_OPERATION_COUNT;
    status = m3_music3_progress_begin(
        progress, M3_MUSIC3_PHASE_DECODE, total, error);
    if (status == M3_STATUS_OK) {
        run->decoded = calloc(run->latents.chunk_count,
                              sizeof(*run->decoded));
        run->decoded_views = calloc(run->latents.chunk_count,
                                    sizeof(*run->decoded_views));
        if (run->decoded == NULL || run->decoded_views == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate Music3 decoded chunks");
        } else {
            run->decoded_count = run->latents.chunk_count;
            decoded = run->decoded;
            decoded_views = run->decoded_views;
        }
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (decoded == NULL || decoded_views == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 decoded arrays were not published");
    }
    for (index = 0U; status == M3_STATUS_OK &&
                    index < run->decoded_count; ++index) {
        m3_music3_progress_bridge bridge;

        m3_vocoder_output_init(&decoded[index]);
        m3_music3_progress_bridge_init(
            &bridge, progress,
            (uint64_t)index * M3_VOCODER_DECODE_OPERATION_COUNT,
            M3_VOCODER_DECODE_OPERATION_COUNT);
        status = engine->operations.vocoder_decode(
            engine->operations.context, run->vocoder,
            &run->latents.chunks[index],
            m3_music3_progress_bridge_report, &bridge,
            &decoded[index], error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_decode_ownership_validate(
                &decoded[index], run->vocoder, &run->latents,
                decoded, index, run->prior_output_storage, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_decode_output_validate(
                engine, &run->latents.chunks[index],
                &decoded[index], error);
            status = m3_music3_result_validation(
                status, "decoded chunk", error);
        }
        if (status == M3_STATUS_OK) {
            decoded_views[index] = decoded[index].waveform;
        }
        status = m3_music3_progress_bridge_complete(
            &bridge, status, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    if (status == M3_STATUS_OK) {
        m3_flow_output_dispose(&run->latents);
        engine->operations.vocoder_free(
            engine->operations.context, run->vocoder);
        run->vocoder = NULL;
    }
    return status;
}

static m3_status m3_music3_assemble_phase(
    m3_music3_engine *engine, m3_music3_run *run, uint64_t frame_count,
    m3_music3_progress_state *progress, m3_error *error)
{
    m3_music3_progress_bridge bridge;
    m3_vocoder_output assembled;
    uint64_t total;
    m3_status status;

    m3_vocoder_output_init(&assembled);
    if ((uint64_t)run->decoded_count > UINT64_MAX / 2U) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 assembly progress overflows");
    }
    total = (uint64_t)run->decoded_count * 2U;
    status = m3_music3_progress_begin(
        progress, M3_MUSIC3_PHASE_ASSEMBLE, total, error);
    if (status == M3_STATUS_OK) {
        m3_music3_progress_bridge_init(&bridge, progress, 0U, total);
        status = engine->operations.waveform_assemble(
            engine->operations.context, run->decoded_views,
            run->decoded_count, frame_count,
            m3_music3_progress_bridge_report, &bridge, &assembled, error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_assemble_ownership_validate(
                &assembled, run->decoded, run->decoded_count,
                run->prior_output_storage, error);
        }
        if (status == M3_STATUS_OK) {
            run->built_output = calloc(1U, sizeof(*run->built_output));
            if (run->built_output == NULL) {
                status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                      "cannot allocate Music3 output");
            } else {
                run->built_output->engine = engine;
                run->built_output->waveform = assembled;
                m3_vocoder_output_init(&assembled);
                status = m3_music3_assembled_output_validate(
                    run->built_output, frame_count, run->decoded_count,
                    error);
                status = m3_music3_result_validation(
                    status, "assembled waveform", error);
            }
        }
        status = m3_music3_progress_bridge_complete(
            &bridge, status, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_progress_finish(progress, error);
    }
    m3_vocoder_output_dispose(&assembled);
    return status;
}

static m3_status m3_music3_generate_run(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_progress_callback callback, void *context,
    m3_music3_output **output, m3_error *error)
{
    m3_music3_progress_state progress;
    m3_music3_run run;
    m3_rng rng;
    uint64_t frame_count = 0U;
    m3_status status;

    m3_music3_progress_init(&progress, callback, context);
    m3_music3_run_init(
        &run, *output == NULL ? NULL : (*output)->waveform.storage);
    status = m3_music3_prepare_phase(
        engine, request, &run, &rng, &progress, error);
    if (status == M3_STATUS_OK) {
        status = m3_music3_semantic_phases(
            engine, request, &run, &rng, &progress, error);
    }
    if (status == M3_STATUS_OK) {
        frame_count = run.semantic.frame_hiddens.metadata.shape[1];
        status = m3_music3_flow_phases(
            engine, &run, &rng, &progress, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_vocoder_stage_phases(
            engine, &run, frame_count, &progress, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_decode_phase(
            engine, &run, &progress, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_assemble_phase(
            engine, &run, frame_count, &progress, error);
    }
    if (status == M3_STATUS_OK) {
        m3_music3_output *old = *output;

        *output = run.built_output;
        run.built_output = NULL;
        m3_music3_output_free(old);
        m3_error_reset(error);
    }
    m3_music3_run_dispose(engine, &run);
    return status;
}

m3_status m3_music3_generate(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_progress_callback progress, void *progress_context,
    m3_music3_output **output, m3_error *error)
{
    m3_status status;

    if (engine == NULL || request == NULL || output == NULL ||
        engine->backend == NULL ||
        !m3_music3_operations_valid(&engine->operations)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 generation arguments are invalid");
    }
    if (engine->generating) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 engine is already generating");
    }
    if (*output != NULL) {
        if ((*output)->engine != engine) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Music3 output belongs to another engine");
        }
        status = m3_music3_output_validate(*output, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    engine->generating = true;
    status = m3_music3_generate_run(
        engine, request, progress, progress_context, output, error);
    engine->generating = false;
    return status;
}
