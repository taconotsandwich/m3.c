/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include "m3_flow_internal.h"
#include "m3_waveform_internal.h"

#include <stdlib.h>

static void m3_music3_maximum(uint64_t value, uint64_t *maximum)
{
    if (value > *maximum) {
        *maximum = value;
    }
}

static m3_status m3_music3_add_peak(uint64_t base, uint64_t added,
                                    uint64_t *peak, m3_error *error)
{
    uint64_t total;
    m3_status status = m3_music3_checked_add(
        base, added, &total, error);

    if (status == M3_STATUS_OK) {
        m3_music3_maximum(total, peak);
    }
    return status;
}

static m3_status m3_music3_vocoder_latents(
    const m3_music3_engine *engine, const m3_flow_output *latents,
    uint64_t frame_count, uint64_t **lengths, uint64_t *latent_bytes,
    m3_error *error)
{
    m3_flow_config config;
    size_t expected = 0U;
    uint64_t *built;
    size_t index;
    m3_status status;

    *lengths = NULL;
    *latent_bytes = 0U;
    if (engine == NULL || latents == NULL || latents->chunk_count == 0U ||
        latents->storages == NULL || latents->chunks == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 latent chunks are invalid");
    }
    m3_flow_config_init(&config);
    status = m3_flow_chunk_count(
        &config, frame_count, &expected, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (expected != latents->chunk_count ||
        expected > SIZE_MAX / sizeof(*built)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 latent chunk count is invalid");
    }
    built = calloc(expected, sizeof(*built));
    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Music3 latent plan");
    }
    for (index = 0U; index < expected; ++index) {
        const m3_tensor_view *view = &latents->chunks[index];
        m3_tensor_view checked;
        uint64_t start;
        uint64_t frames;
        uint64_t length;
        size_t storage_bytes;

        status = m3_flow_chunk_window(
            &config, frame_count, index, &start, &frames, error);
        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                frames, config.condition.resize_numerator,
                config.condition.resize_denominator, &length, error);
        }
        storage_bytes = m3_storage_size(latents->storages[index]);
        if (status == M3_STATUS_OK &&
            (latents->storages[index] == NULL ||
             view->storage != latents->storages[index] ||
             view->metadata.dtype != M3_DTYPE_F32 ||
             view->metadata.rank != 3U ||
             view->metadata.shape[0] != 1U ||
             view->metadata.shape[1] != M3_FLOW_LATENT_CHANNELS ||
             view->metadata.shape[2] != length ||
             !m3_tensor_is_contiguous(view) ||
             m3_storage_backend(view->storage) != engine->backend ||
             storage_bytes != view->metadata.byte_count)) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "Music3 latent chunk shape is invalid");
        }
        if (status == M3_STATUS_OK) {
            m3_tensor_view_init(&checked);
            status = m3_tensor_reshape(
                view, view->metadata.rank, view->metadata.shape, &checked,
                error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_checked_add(
                *latent_bytes, (uint64_t)storage_bytes, latent_bytes,
                error);
        }
        if (status != M3_STATUS_OK) {
            free(built);
            return status;
        }
        built[index] = length;
        (void)start;
    }
    *lengths = built;
    return M3_STATUS_OK;
}

static m3_status m3_music3_vocoder_runtime_plan(
    m3_vocoder_plan *vocoder, uint64_t *bytes, uint64_t *largest,
    m3_error *error)
{
    m3_vocoder_plan_config config;
    size_t index;
    m3_status status;

    *bytes = 0U;
    *largest = 0U;
    m3_vocoder_plan_official_config(&config);
    m3_vocoder_plan_init(vocoder);
    status = m3_vocoder_plan_build(&config, vocoder, error);
    if (status == M3_STATUS_OK &&
        (vocoder->entry_count != M3_VOCODER_RUNTIME_WEIGHT_COUNT ||
         vocoder->source_count != M3_VOCODER_SOURCE_WEIGHT_COUNT)) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 vocoder plan count is invalid");
    }
    for (index = 0U; index < vocoder->entry_count &&
                    status == M3_STATUS_OK; ++index) {
        uint64_t entry =
            (uint64_t)vocoder->entries[index].output_metadata.byte_count;

        status = m3_music3_checked_add(*bytes, entry, bytes, error);
        if (status == M3_STATUS_OK) {
            m3_music3_maximum(entry, largest);
        }
    }
    return status;
}

static m3_status m3_music3_vocoder_decode_peaks(
    const m3_vocoder_plan_config *config, const uint64_t *lengths,
    size_t chunk_count, uint64_t current, uint64_t runtime_bytes,
    uint64_t *peak, uint64_t *largest, uint64_t *decoded_bytes,
    m3_error *error)
{
    size_t index;
    m3_status status = M3_STATUS_OK;

    if (config == NULL || lengths == NULL || chunk_count == 0U ||
        peak == NULL || largest == NULL || decoded_bytes == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 vocoder decode plan is invalid");
    }
    *decoded_bytes = 0U;
    for (index = 0U; index < chunk_count && status == M3_STATUS_OK;
         ++index) {
        m3_vocoder_decode_measurement measurement = {{0U, 0U, 0U},
                                                      0U, 0U, 0U};
        uint64_t active;
        size_t slot;

        status = m3_vocoder_decode_measure(
            config, lengths[index], &measurement, error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_checked_add(
                current, runtime_bytes, &active, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_checked_add(
                active, *decoded_bytes, &active, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_checked_add(
                active, (uint64_t)measurement.workspace_bytes, &active,
                error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_add_peak(
                active, (uint64_t)measurement.output_bytes, peak, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_checked_add(
                *decoded_bytes, (uint64_t)measurement.output_bytes,
                decoded_bytes, error);
        }
        if (status == M3_STATUS_OK) {
            for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT;
                 ++slot) {
                m3_music3_maximum(
                    (uint64_t)measurement.buffer_bytes[slot], largest);
            }
            m3_music3_maximum(
                (uint64_t)measurement.output_bytes, largest);
        }
    }
    return status;
}

m3_status m3_music3_preflight_vocoder(
    const m3_music3_engine *engine, const m3_flow_output *latents,
    uint64_t frame_count, m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_music3_allocation_plan stage = {0U, 0U};
    m3_music3_allocation_plan future = {0U, 0U};
    m3_vocoder_plan vocoder;
    m3_waveform_measurement waveform;
    uint64_t *lengths = NULL;
    uint64_t latent_bytes = 0U;
    uint64_t runtime_bytes = 0U;
    uint64_t runtime_largest = 0U;
    uint64_t current;
    uint64_t baseline;
    uint64_t peak;
    uint64_t decoded_bytes = 0U;
    uint64_t active;
    m3_status status;

    status = m3_music3_vocoder_latents(
        engine, latents, frame_count, &lengths, &latent_bytes, error);
    if (status == M3_STATUS_OK) {
        status = m3_music3_component_plan(
            engine, M3_COMPONENT_VOCODER, &stage, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_vocoder_runtime_plan(
            &vocoder, &runtime_bytes, &runtime_largest, error);
    } else {
        m3_vocoder_plan_init(&vocoder);
    }
    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(
            engine->backend, &stats, error);
    }
    current = status == M3_STATUS_OK
                  ? (uint64_t)stats.live_allocated_bytes
                  : 0U;
    if (status == M3_STATUS_OK &&
        ((size_t)current != stats.live_allocated_bytes ||
         latent_bytes > current)) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 latent ownership is inconsistent");
    }
    baseline = status == M3_STATUS_OK ? current - latent_bytes : 0U;
    peak = current;
    if (status == M3_STATUS_OK) {
        future.largest_storage_bytes = stage.largest_storage_bytes;
        m3_music3_maximum(runtime_largest,
                          &future.largest_storage_bytes);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_checked_add(
            stage.added_bytes, runtime_bytes, &active, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_add_peak(current, active, &peak, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_vocoder_decode_peaks(
            &vocoder.config, lengths, latents->chunk_count, current,
            runtime_bytes, &peak, &future.largest_storage_bytes,
            &decoded_bytes, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_waveform_measure(
            frame_count, latents->chunk_count, &waveform, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_checked_add(
            baseline, decoded_bytes, &active, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_add_peak(
            active, (uint64_t)waveform.output_bytes, &peak, error);
        m3_music3_maximum(
            (uint64_t)waveform.output_bytes,
            &future.largest_storage_bytes);
    }
    if (status == M3_STATUS_OK && peak <= current) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 vocoder plan has no future storage");
    }
    if (status == M3_STATUS_OK) {
        future.added_bytes = peak - current;
        status = m3_music3_preflight_added(
            engine->backend, &future, error);
    }
    m3_vocoder_plan_dispose(&vocoder);
    free(lengths);
    return status;
}
