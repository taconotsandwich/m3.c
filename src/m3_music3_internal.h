/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MUSIC3_INTERNAL_H
#define M3_MUSIC3_INTERNAL_H

#include "m3_flow_runtime.h"
#include "m3_model.h"
#include "m3_prompt.h"
#include "m3_semantic_internal.h"
#include "m3_tokenizer.h"
#include "m3_vocoder_internal.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_MUSIC3_WEIGHT_COMPONENT_COUNT 5U
#define M3_MUSIC3_SAMPLE_RATE 44100U
#define M3_MUSIC3_CHANNEL_COUNT 2U
#define M3_MUSIC3_STAGE_SEMANTIC_BYTES UINT64_C(18461001728)
#define M3_MUSIC3_STAGE_FLOW_BYTES UINT64_C(9828295204)
#define M3_MUSIC3_STAGE_VOCODER_BYTES UINT64_C(216682888)

_Static_assert(M3_MUSIC3_WEIGHT_COMPONENT_COUNT == 5U,
               "Music3 retains exactly five weighted components");
_Static_assert(M3_COMPONENT_LANGUAGE_MODEL == 0 &&
                   M3_COMPONENT_RVQ_DEPTH_DECODER == 1 &&
                   M3_COMPONENT_CONDITION_ENCODER == 2 &&
                   M3_COMPONENT_TRANSFORMER == 3 &&
                   M3_COMPONENT_VOCODER == 4,
               "Music3 weighted component indexes must remain contiguous");

typedef struct {
    void *context;
    /* Result counts and arrays remain accurate, traversable, and safe for
       the normal disposer on every return. On success the scheduler may
       reject and detach repeated or forbidden borrowed owners. On failure
       the operation has already detached every borrowed/pre-existing owner
       before scheduler cleanup. */
    m3_status (*create_backend)(void *context, m3_backend **backend,
                                m3_error *error);
    m3_status (*inspect_model)(void *context, const char *root,
                               m3_model_metadata *metadata,
                               m3_error *error);
    m3_status (*inspect_weights)(void *context, const char *root,
                                 m3_component_id component,
                                 m3_weight_table *table,
                                 m3_error *error);
    m3_status (*load_tokenizer)(void *context, m3_tokenizer *tokenizer,
                                const char *path, m3_error *error);
    m3_status (*stage_weights)(
        void *context, m3_component_id component,
        const m3_weight_table *table, m3_backend *backend,
        m3_progress_callback progress, void *progress_context,
        m3_weight_stage *stage, m3_error *error);
    m3_status (*stage_validate)(
        void *context, m3_component_id component,
        const m3_weight_stage *stage, const m3_weight_table *table,
        m3_backend *backend, m3_error *error);
    m3_status (*semantic_generate)(
        void *context, const m3_weight_stage *language_model,
        const m3_weight_stage *rvq, const m3_tensor_view *prompt_ids,
        uint64_t frame_limit, m3_rng *rng, m3_progress_callback progress,
        void *progress_context, m3_semantic_output *output,
        m3_error *error);
    m3_status (*flow_synthesize)(
        void *context, m3_backend *backend,
        const m3_weight_stage *condition,
        const m3_weight_stage *flow,
        const m3_tensor_view *frame_hiddens, m3_rng *rng,
        m3_progress_callback progress, void *progress_context,
        m3_flow_output *output, m3_error *error);
    m3_status (*vocoder_create)(
        void *context, const m3_weight_stage *vocoder,
        m3_progress_callback progress, void *progress_context,
        m3_vocoder_runtime **runtime, m3_error *error);
    void (*vocoder_free)(void *context, m3_vocoder_runtime *runtime);
    m3_status (*vocoder_validate)(
        void *context, const m3_vocoder_runtime *runtime,
        m3_backend *backend, m3_error *error);
    m3_status (*vocoder_decode)(
        void *context, m3_vocoder_runtime *runtime,
        const m3_tensor_view *latents, m3_progress_callback progress,
        void *progress_context, m3_vocoder_output *output,
        m3_error *error);
    m3_status (*waveform_assemble)(
        void *context, const m3_tensor_view *chunks, size_t chunk_count,
        uint64_t frame_count, m3_progress_callback progress,
        void *progress_context, m3_vocoder_output *output,
        m3_error *error);
} m3_music3_operations;

struct m3_music3_engine {
    m3_backend *backend;
    m3_tokenizer tokenizer;
    m3_weight_table tables[M3_MUSIC3_WEIGHT_COMPONENT_COUNT];
    uint64_t component_payload_bytes[M3_MUSIC3_WEIGHT_COMPONENT_COUNT];
    uint64_t component_largest_shard_bytes[
        M3_MUSIC3_WEIGHT_COMPONENT_COUNT];
    m3_music3_operations operations;
    bool generating;
};

struct m3_music3_output {
    m3_music3_engine *engine;
    m3_vocoder_output waveform;
};

typedef struct {
    m3_music3_progress_callback callback;
    void *context;
    m3_music3_phase phase;
    uint64_t total;
    uint64_t last_completed;
    bool active;
} m3_music3_progress_state;

typedef struct {
    m3_music3_progress_state *state;
    uint64_t offset;
    uint64_t child_total;
    uint64_t last_child_completed;
    bool saw_total;
    bool cancelled;
    bool invalid;
} m3_music3_progress_bridge;

typedef struct {
    m3_storage *storage;
    m3_tensor_view ids;
    m3_semantic_plan semantic_plan;
} m3_music3_prepared_prompt;

typedef struct {
    uint64_t added_bytes;
    uint64_t largest_storage_bytes;
} m3_music3_allocation_plan;

const m3_music3_operations *m3_music3_production_operations(void);
bool m3_music3_operations_valid(const m3_music3_operations *operations);

m3_status m3_music3_engine_open_core(
    m3_music3_engine **engine, const char *trusted_model_root,
    const m3_music3_operations *operations, m3_error *error);

void m3_music3_progress_init(
    m3_music3_progress_state *state,
    m3_music3_progress_callback callback, void *context);
m3_status m3_music3_progress_begin(
    m3_music3_progress_state *state, m3_music3_phase phase,
    uint64_t total, m3_error *error);
m3_status m3_music3_progress_finish(
    m3_music3_progress_state *state, m3_error *error);
void m3_music3_progress_bridge_init(
    m3_music3_progress_bridge *bridge,
    m3_music3_progress_state *state, uint64_t offset,
    uint64_t child_total);
bool m3_music3_progress_bridge_report(
    void *context, uint64_t completed, uint64_t total);
m3_status m3_music3_progress_bridge_complete(
    m3_music3_progress_bridge *bridge, m3_status child_status,
    m3_error *error);

void m3_music3_prepared_prompt_init(m3_music3_prepared_prompt *prompt);
void m3_music3_prepared_prompt_dispose(m3_music3_prepared_prompt *prompt);
m3_status m3_music3_prepare_prompt(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_prepared_prompt *prompt, m3_error *error);

m3_status m3_music3_checked_add(uint64_t left, uint64_t right,
                                uint64_t *sum, m3_error *error);
m3_status m3_music3_table_plan(
    const m3_weight_table *table, m3_music3_allocation_plan *plan,
    m3_error *error);
m3_status m3_music3_component_plan(
    const m3_music3_engine *engine, m3_component_id component,
    m3_music3_allocation_plan *plan, m3_error *error);
m3_status m3_music3_preflight_added(
    m3_backend *backend, const m3_music3_allocation_plan *plan,
    m3_error *error);
m3_status m3_music3_preflight_semantic(
    const m3_music3_engine *engine, const m3_semantic_plan *semantic,
    m3_error *error);
m3_status m3_music3_preflight_flow(
    const m3_music3_engine *engine,
    const m3_tensor_view *frame_hiddens, m3_error *error);
m3_status m3_music3_preflight_vocoder(
    const m3_music3_engine *engine, const m3_flow_output *latents,
    uint64_t frame_count, m3_error *error);

m3_status m3_music3_semantic_output_validate(
    const m3_music3_engine *engine, const m3_semantic_output *output,
    uint64_t frame_limit, m3_error *error);
m3_status m3_music3_flow_progress_plan(
    uint64_t frames, size_t *chunk_count, uint64_t *progress_total,
    m3_error *error);
m3_status m3_music3_flow_output_validate(
    const m3_music3_engine *engine, const m3_flow_output *output,
    uint64_t frame_count, size_t expected, m3_error *error);
m3_status m3_music3_decode_output_validate(
    const m3_music3_engine *engine, const m3_tensor_view *latent,
    const m3_vocoder_output *output, m3_error *error);
m3_status m3_music3_assembled_output_validate(
    const m3_music3_output *output, uint64_t frame_count,
    size_t chunk_count, m3_error *error);

m3_status m3_music3_semantic_ownership_validate(
    m3_semantic_output *output,
    const m3_music3_prepared_prompt *prompt,
    const m3_weight_stage *language_model,
    const m3_weight_stage *rvq, const m3_storage *forbidden,
    m3_error *error);
m3_status m3_music3_stage_ownership_validate(
    m3_weight_stage *stage, const m3_storage *first,
    const m3_storage *second, const m3_weight_stage *prior_stage,
    const m3_flow_output *prior_flow,
    m3_error *error);
m3_status m3_music3_flow_ownership_validate(
    m3_flow_output *output, const m3_semantic_output *semantic,
    const m3_weight_stage *condition, const m3_weight_stage *flow,
    const m3_storage *forbidden, m3_error *error);
m3_status m3_music3_runtime_ownership_validate(
    m3_vocoder_runtime *runtime, const m3_weight_stage *vocoder,
    const m3_flow_output *latents, const m3_storage *forbidden,
    m3_error *error);
m3_status m3_music3_decode_ownership_validate(
    m3_vocoder_output *output, const m3_vocoder_runtime *runtime,
    const m3_flow_output *latents, const m3_vocoder_output *decoded,
    size_t decoded_count, const m3_storage *forbidden,
    m3_error *error);
m3_status m3_music3_assemble_ownership_validate(
    m3_vocoder_output *output, const m3_vocoder_output *decoded,
    size_t decoded_count, const m3_storage *forbidden,
    m3_error *error);

m3_status m3_music3_output_validate(
    const m3_music3_output *output, m3_error *error);

#endif
