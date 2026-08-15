/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include "m3_flow_internal.h"
#include "m3_waveform_internal.h"

m3_status m3_music3_semantic_output_validate(
    const m3_music3_engine *engine, const m3_semantic_output *output,
    uint64_t frame_limit, m3_error *error)
{
    m3_tensor_view checked;
    m3_status status;

    if (output->storage == NULL ||
        output->frame_hiddens.storage != output->storage ||
        output->frame_hiddens.metadata.dtype != M3_DTYPE_BF16 ||
        output->frame_hiddens.metadata.rank != 4U ||
        output->frame_hiddens.metadata.shape[0] != 1U ||
        output->frame_hiddens.metadata.shape[1] == 0U ||
        output->frame_hiddens.metadata.shape[1] > frame_limit ||
        output->frame_hiddens.metadata.shape[1] > M3_FLOW_MAX_FRAMES ||
        output->frame_hiddens.metadata.shape[2] != M3_RVQ_CODEBOOK_COUNT ||
        output->frame_hiddens.metadata.shape[3] != M3_QWEN_HIDDEN_SIZE ||
        output->frame_hiddens.byte_offset != 0U ||
        !m3_tensor_is_contiguous(&output->frame_hiddens) ||
        m3_storage_backend(output->storage) != engine->backend ||
        m3_storage_size(output->storage) !=
            output->frame_hiddens.metadata.byte_count) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 semantic output is invalid");
    }
    m3_tensor_view_init(&checked);
    status = m3_tensor_reshape(
        &output->frame_hiddens, output->frame_hiddens.metadata.rank,
        output->frame_hiddens.metadata.shape, &checked, error);
    return status == M3_STATUS_OK
               ? M3_STATUS_OK
               : m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 semantic output is out of bounds");
}

m3_status m3_music3_flow_progress_plan(
    uint64_t frames, size_t *chunk_count, uint64_t *progress_total,
    m3_error *error)
{
    m3_flow_config config;
    uint64_t count;
    m3_status status;

    m3_flow_config_init(&config);
    status = m3_flow_chunk_count(&config, frames, chunk_count, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    count = (uint64_t)*chunk_count;
    if ((size_t)count != *chunk_count ||
        count > UINT64_MAX / M3_FLOW_INFERENCE_STEPS) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 flow progress total overflows");
    }
    *progress_total = count * M3_FLOW_INFERENCE_STEPS;
    return M3_STATUS_OK;
}

m3_status m3_music3_flow_output_validate(
    const m3_music3_engine *engine, const m3_flow_output *output,
    uint64_t frame_count, size_t expected, m3_error *error)
{
    m3_flow_config config;
    size_t index;

    if (output->chunk_count != expected || output->storages == NULL ||
        output->chunks == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 flow output count is invalid");
    }
    m3_flow_config_init(&config);
    for (index = 0U; index < expected; ++index) {
        const m3_tensor_view *chunk = &output->chunks[index];
        m3_tensor_view checked;
        size_t earlier;
        uint64_t start;
        uint64_t frames;
        uint64_t length;
        m3_status status = m3_flow_chunk_window(
            &config, frame_count, index, &start, &frames, error);

        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                frames, config.condition.resize_numerator,
                config.condition.resize_denominator, &length, error);
        }
        if (status != M3_STATUS_OK || output->storages[index] == NULL ||
            chunk->storage != output->storages[index] ||
            chunk->metadata.dtype != M3_DTYPE_F32 ||
            chunk->metadata.rank != 3U ||
            chunk->metadata.shape[0] != 1U ||
            chunk->metadata.shape[1] != M3_FLOW_LATENT_CHANNELS ||
            chunk->metadata.shape[2] != length ||
            !m3_tensor_is_contiguous(chunk) ||
            m3_storage_backend(chunk->storage) != engine->backend ||
            m3_storage_size(output->storages[index]) !=
                chunk->metadata.byte_count) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Music3 flow output shape is invalid");
        }
        for (earlier = 0U; earlier < index; ++earlier) {
            if (output->storages[index] == output->storages[earlier]) {
                return m3_error_set(
                    error, M3_STATUS_INTERNAL,
                    "Music3 flow chunks share owned storage");
            }
        }
        m3_tensor_view_init(&checked);
        status = m3_tensor_reshape(
            chunk, chunk->metadata.rank, chunk->metadata.shape, &checked,
            error);
        if (status != M3_STATUS_OK) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Music3 flow output is out of bounds");
        }
        (void)start;
    }
    return M3_STATUS_OK;
}

m3_status m3_music3_decode_output_validate(
    const m3_music3_engine *engine, const m3_tensor_view *latent,
    const m3_vocoder_output *output, m3_error *error)
{
    uint64_t length = latent->metadata.shape[2];
    m3_tensor_view checked;
    m3_status status;

    if (length > UINT64_MAX / 512U || output->storage == NULL ||
        output->waveform.storage != output->storage ||
        output->waveform.metadata.dtype != M3_DTYPE_F32 ||
        output->waveform.metadata.rank != 3U ||
        output->waveform.metadata.shape[0] != 1U ||
        output->waveform.metadata.shape[1] != M3_MUSIC3_CHANNEL_COUNT ||
        output->waveform.metadata.shape[2] != length * 512U ||
        !m3_tensor_is_contiguous(&output->waveform) ||
        m3_storage_backend(output->storage) != engine->backend ||
        m3_storage_size(output->storage) !=
            output->waveform.metadata.byte_count) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 decoded chunk is invalid");
    }
    m3_tensor_view_init(&checked);
    status = m3_tensor_reshape(
        &output->waveform, output->waveform.metadata.rank,
        output->waveform.metadata.shape, &checked, error);
    return status == M3_STATUS_OK
               ? M3_STATUS_OK
               : m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 decoded chunk is out of bounds");
}

m3_status m3_music3_assembled_output_validate(
    const m3_music3_output *output, uint64_t frame_count,
    size_t chunk_count, m3_error *error)
{
    m3_waveform_measurement measurement;
    m3_status status = m3_music3_output_validate(output, error);

    if (status == M3_STATUS_OK) {
        status = m3_waveform_measure(
            frame_count, chunk_count, &measurement, error);
    }
    if (status != M3_STATUS_OK) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 assembly measurement is invalid");
    }
    if (
        output->waveform.waveform.metadata.shape[2] !=
            (uint64_t)measurement.output_samples) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Music3 assembled length is invalid");
    }
    return status;
}
