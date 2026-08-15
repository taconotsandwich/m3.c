/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

static bool m3_music3_stage_has(
    const m3_weight_stage *stage, const m3_storage *storage)
{
    size_t index;

    if (stage == NULL || stage->storages == NULL || storage == NULL) {
        return false;
    }
    for (index = 0U; index < stage->storage_count; ++index) {
        if (stage->storages[index] == storage) {
            return true;
        }
    }
    return false;
}

static bool m3_music3_flow_has(
    const m3_flow_output *flow, const m3_storage *storage)
{
    size_t index;

    if (flow == NULL || flow->storages == NULL || storage == NULL) {
        return false;
    }
    for (index = 0U; index < flow->chunk_count; ++index) {
        if (flow->storages[index] == storage) {
            return true;
        }
    }
    return false;
}

static bool m3_music3_runtime_has(
    const m3_vocoder_runtime *runtime, const m3_storage *storage)
{
    size_t index;

    if (runtime == NULL || runtime->weights.storages == NULL ||
        storage == NULL) {
        return false;
    }
    for (index = 0U; index < runtime->weights.count; ++index) {
        if (runtime->weights.storages[index] == storage) {
            return true;
        }
    }
    return false;
}

static bool m3_music3_decoded_has(
    const m3_vocoder_output *decoded, size_t count,
    const m3_storage *storage)
{
    size_t index;

    if (decoded == NULL || storage == NULL) {
        return false;
    }
    for (index = 0U; index < count; ++index) {
        if (decoded[index].storage == storage) {
            return true;
        }
    }
    return false;
}

static void m3_music3_output_disown(m3_vocoder_output *output)
{
    output->storage = NULL;
    m3_tensor_view_init(&output->waveform);
}

m3_status m3_music3_stage_ownership_validate(
    m3_weight_stage *stage, const m3_storage *first,
    const m3_storage *second, const m3_weight_stage *prior_stage,
    const m3_flow_output *prior_flow,
    m3_error *error)
{
    size_t index;
    bool aliased = false;

    if (stage == NULL) {
        return M3_STATUS_OK;
    }
    if (stage->storages != NULL) {
        for (index = 0U; index < stage->storage_count; ++index) {
            m3_storage *storage = stage->storages[index];

            if ((storage != NULL &&
                 (storage == first || storage == second)) ||
                m3_music3_stage_has(prior_stage, storage) ||
                m3_music3_flow_has(prior_flow, storage)) {
                stage->storages[index] = NULL;
                aliased = true;
            }
        }
    }
    if (stage->views != NULL) {
        for (index = 0U; index < stage->view_count; ++index) {
            m3_storage *storage = stage->views[index].storage;

            if ((storage != NULL &&
                 (storage == first || storage == second)) ||
                m3_music3_stage_has(prior_stage, storage) ||
                m3_music3_flow_has(prior_flow, storage)) {
                m3_tensor_view_init(&stage->views[index]);
            }
        }
    }
    return aliased
               ? m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 stage aliases the prior output")
               : M3_STATUS_OK;
}

m3_status m3_music3_semantic_ownership_validate(
    m3_semantic_output *output,
    const m3_music3_prepared_prompt *prompt,
    const m3_weight_stage *language_model,
    const m3_weight_stage *rvq, const m3_storage *forbidden,
    m3_error *error)
{
    if (output != NULL && output->storage != NULL &&
        (output->storage == forbidden ||
         output->storage == prompt->storage ||
         m3_music3_stage_has(language_model, output->storage) ||
         m3_music3_stage_has(rvq, output->storage))) {
        output->storage = NULL;
        m3_tensor_view_init(&output->frame_hiddens);
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 semantic output aliases an input");
    }
    return M3_STATUS_OK;
}

m3_status m3_music3_flow_ownership_validate(
    m3_flow_output *output, const m3_semantic_output *semantic,
    const m3_weight_stage *condition, const m3_weight_stage *flow,
    const m3_storage *forbidden, m3_error *error)
{
    size_t index;
    bool aliased = false;

    if (output == NULL || output->storages == NULL) {
        return M3_STATUS_OK;
    }
    for (index = 0U; index < output->chunk_count; ++index) {
        m3_storage *storage = output->storages[index];

        if (storage != NULL &&
            (storage == forbidden || storage == semantic->storage ||
             m3_music3_stage_has(condition, storage) ||
             m3_music3_stage_has(flow, storage))) {
            output->storages[index] = NULL;
            if (output->chunks != NULL) {
                m3_tensor_view_init(&output->chunks[index]);
            }
            aliased = true;
        }
    }
    return aliased
               ? m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 flow output aliases an input")
               : M3_STATUS_OK;
}

m3_status m3_music3_runtime_ownership_validate(
    m3_vocoder_runtime *runtime, const m3_weight_stage *vocoder,
    const m3_flow_output *latents, const m3_storage *forbidden,
    m3_error *error)
{
    size_t index;
    bool aliased = false;

    if (runtime == NULL || runtime->weights.storages == NULL) {
        return M3_STATUS_OK;
    }
    for (index = 0U; index < runtime->weights.count; ++index) {
        m3_storage *storage = runtime->weights.storages[index];

        if (storage == forbidden ||
            m3_music3_stage_has(vocoder, storage) ||
            m3_music3_flow_has(latents, storage)) {
            runtime->weights.storages[index] = NULL;
            if (runtime->weights.views != NULL) {
                m3_tensor_view_init(&runtime->weights.views[index]);
            }
            aliased = true;
        }
    }
    return aliased
               ? m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 vocoder runtime aliases an input")
               : M3_STATUS_OK;
}

m3_status m3_music3_decode_ownership_validate(
    m3_vocoder_output *output, const m3_vocoder_runtime *runtime,
    const m3_flow_output *latents, const m3_vocoder_output *decoded,
    size_t decoded_count, const m3_storage *forbidden,
    m3_error *error)
{
    if (output != NULL && output->storage != NULL &&
        (output->storage == forbidden ||
         m3_music3_runtime_has(runtime, output->storage) ||
         m3_music3_flow_has(latents, output->storage) ||
         m3_music3_decoded_has(
             decoded, decoded_count, output->storage))) {
        m3_music3_output_disown(output);
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 decoded output aliases prior storage");
    }
    return M3_STATUS_OK;
}

m3_status m3_music3_assemble_ownership_validate(
    m3_vocoder_output *output, const m3_vocoder_output *decoded,
    size_t decoded_count, const m3_storage *forbidden,
    m3_error *error)
{
    if (output != NULL && output->storage != NULL &&
        (output->storage == forbidden ||
         m3_music3_decoded_has(
             decoded, decoded_count, output->storage))) {
        m3_music3_output_disown(output);
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 assembled output aliases a chunk");
    }
    return M3_STATUS_OK;
}
