/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include "m3_provider.h"
#include "m3_weight_stage_internal.h"

static m3_status m3_music3_create_backend(void *context,
                                           m3_backend **backend,
                                           m3_error *error)
{
    (void)context;
    return m3_backend_create_preferred(true, true, backend, error);
}

static m3_status m3_music3_inspect_model(void *context, const char *root,
                                         m3_model_metadata *metadata,
                                         m3_error *error)
{
    (void)context;
    return m3_model_inspect_directory(root, metadata, error);
}

static m3_status m3_music3_inspect_weights(
    void *context, const char *root, m3_component_id component,
    m3_weight_table *table, m3_error *error)
{
    (void)context;
    return m3_music3_inspect_weight_table(root, component, table, error);
}

static m3_status m3_music3_load_tokenizer(void *context,
                                           m3_tokenizer *tokenizer,
                                           const char *path,
                                           m3_error *error)
{
    (void)context;
    return m3_tokenizer_load(tokenizer, path, error);
}

static m3_status m3_music3_stage_weights(
    void *context, m3_component_id component,
    const m3_weight_table *table, m3_backend *backend,
    m3_progress_callback progress, void *progress_context,
    m3_weight_stage *stage, m3_error *error)
{
    (void)context;
    (void)component;
    return m3_weight_stage_load(stage, table, backend, progress,
                                progress_context, error);
}

static m3_status m3_music3_stage_validate(
    void *context, m3_component_id component,
    const m3_weight_stage *stage, const m3_weight_table *table,
    m3_backend *backend, m3_error *error)
{
    (void)context;
    (void)component;
    return m3_weight_stage_validate(stage, table, backend, error);
}

static m3_status m3_music3_semantic(
    void *context, const m3_weight_stage *language_model,
    const m3_weight_stage *rvq, const m3_tensor_view *prompt_ids,
    uint64_t frame_limit, m3_rng *rng, m3_progress_callback progress,
    void *progress_context, m3_semantic_output *output, m3_error *error)
{
    (void)context;
    return m3_semantic_generate(
        language_model, rvq, prompt_ids, frame_limit, rng, progress,
        progress_context, output, error);
}

static m3_status m3_music3_flow(
    void *context, m3_backend *backend,
    const m3_weight_stage *condition, const m3_weight_stage *flow,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *output, m3_error *error)
{
    m3_condition_weights condition_weights;
    m3_flow_weights flow_weights;
    m3_status status;

    (void)context;
    status = m3_condition_weights_bind(
        condition, &condition_weights, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_weights_bind(flow, &flow_weights, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_synthesize(
            backend, &flow_weights, &condition_weights, frame_hiddens,
            rng, progress, progress_context, output, error);
    }
    return status;
}

static m3_status m3_music3_vocoder_create(
    void *context, const m3_weight_stage *vocoder,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_runtime **runtime, m3_error *error)
{
    (void)context;
    return m3_vocoder_runtime_create(
        runtime, vocoder, progress, progress_context, error);
}

static void m3_music3_vocoder_free(void *context,
                                    m3_vocoder_runtime *runtime)
{
    (void)context;
    m3_vocoder_runtime_free(runtime);
}

static m3_status m3_music3_vocoder_validate(
    void *context, const m3_vocoder_runtime *runtime,
    m3_backend *backend, m3_error *error)
{
    (void)context;
    return m3_vocoder_runtime_validate(runtime, backend, error);
}

static m3_status m3_music3_vocoder_decode(
    void *context, m3_vocoder_runtime *runtime,
    const m3_tensor_view *latents, m3_progress_callback progress,
    void *progress_context, m3_vocoder_output *output, m3_error *error)
{
    (void)context;
    return m3_vocoder_decode_chunk(
        runtime, latents, progress, progress_context, output, error);
}

static m3_status m3_music3_assemble(
    void *context, const m3_tensor_view *chunks, size_t chunk_count,
    uint64_t frame_count, m3_progress_callback progress,
    void *progress_context, m3_vocoder_output *output, m3_error *error)
{
    (void)context;
    return m3_waveform_assemble(
        chunks, chunk_count, frame_count, progress, progress_context,
        output, error);
}

const m3_music3_operations *m3_music3_production_operations(void)
{
    static const m3_music3_operations operations = {
        NULL,
        m3_music3_create_backend,
        m3_music3_inspect_model,
        m3_music3_inspect_weights,
        m3_music3_load_tokenizer,
        m3_music3_stage_weights,
        m3_music3_stage_validate,
        m3_music3_semantic,
        m3_music3_flow,
        m3_music3_vocoder_create,
        m3_music3_vocoder_free,
        m3_music3_vocoder_validate,
        m3_music3_vocoder_decode,
        m3_music3_assemble
    };

    return &operations;
}

bool m3_music3_operations_valid(const m3_music3_operations *operations)
{
    return operations != NULL && operations->create_backend != NULL &&
           operations->inspect_model != NULL &&
           operations->inspect_weights != NULL &&
           operations->load_tokenizer != NULL &&
           operations->stage_weights != NULL &&
           operations->stage_validate != NULL &&
           operations->semantic_generate != NULL &&
           operations->flow_synthesize != NULL &&
           operations->vocoder_create != NULL &&
           operations->vocoder_free != NULL &&
           operations->vocoder_validate != NULL &&
           operations->vocoder_decode != NULL &&
           operations->waveform_assemble != NULL;
}
