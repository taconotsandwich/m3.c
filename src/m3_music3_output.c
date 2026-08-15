/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include "m3_wav.h"

#include <stdint.h>
#include <stdlib.h>

m3_status m3_music3_output_validate(
    const m3_music3_output *output, m3_error *error)
{
    m3_tensor_view checked;
    m3_status status;

    if (output == NULL || output->engine == NULL ||
        output->engine->backend == NULL || output->waveform.storage == NULL ||
        output->waveform.waveform.storage != output->waveform.storage ||
        output->waveform.waveform.metadata.dtype != M3_DTYPE_F32 ||
        output->waveform.waveform.metadata.rank != 3U ||
        output->waveform.waveform.metadata.shape[0] != 1U ||
        output->waveform.waveform.metadata.shape[1] !=
            M3_MUSIC3_CHANNEL_COUNT ||
        output->waveform.waveform.metadata.shape[2] == 0U ||
        output->waveform.waveform.byte_offset != 0U ||
        !m3_tensor_is_contiguous(&output->waveform.waveform) ||
        m3_storage_backend(output->waveform.storage) !=
            output->engine->backend ||
        m3_storage_size(output->waveform.storage) !=
            output->waveform.waveform.metadata.byte_count) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 output is invalid");
    }
    m3_tensor_view_init(&checked);
    status = m3_tensor_reshape(
        &output->waveform.waveform,
        output->waveform.waveform.metadata.rank,
        output->waveform.waveform.metadata.shape, &checked, error);
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

void m3_music3_output_free(m3_music3_output *output)
{
    if (output != NULL) {
        m3_vocoder_output_dispose(&output->waveform);
        free(output);
    }
}

m3_status m3_music3_output_get_info(
    const m3_music3_output *output, m3_music3_output_info *info,
    m3_error *error)
{
    m3_music3_output_info built;
    m3_status status;

    if (info == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 output info is null");
    }
    status = m3_music3_output_validate(output, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    built.sample_rate = M3_MUSIC3_SAMPLE_RATE;
    built.channel_count = M3_MUSIC3_CHANNEL_COUNT;
    built.samples_per_channel =
        output->waveform.waveform.metadata.shape[2];
    *info = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_music3_output_read_channel_f32(
    const m3_music3_output *output, uint32_t channel,
    uint64_t sample_offset, float *destination, size_t sample_count,
    m3_error *error)
{
    uint64_t samples;
    uint64_t plane_bytes;
    uint64_t relative;
    uint64_t absolute;
    size_t byte_offset;
    size_t byte_count;
    m3_status status = m3_music3_output_validate(output, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    samples = output->waveform.waveform.metadata.shape[2];
    if (channel >= M3_MUSIC3_CHANNEL_COUNT || sample_offset > samples ||
        (uint64_t)sample_count > samples - sample_offset ||
        (sample_count != 0U && destination == NULL)) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Music3 channel read range is invalid");
    }
    if (sample_count == 0U) {
        m3_error_reset(error);
        return M3_STATUS_OK;
    }
    if (samples > UINT64_MAX / sizeof(float) ||
        sample_offset > UINT64_MAX / sizeof(float) ||
        sample_count > SIZE_MAX / sizeof(float)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 channel read size overflows");
    }
    plane_bytes = samples * sizeof(float);
    relative = (uint64_t)channel * plane_bytes +
               sample_offset * sizeof(float);
    if ((uint64_t)output->waveform.waveform.byte_offset >
        UINT64_MAX - relative) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 channel read offset overflows");
    }
    absolute = (uint64_t)output->waveform.waveform.byte_offset + relative;
    byte_offset = (size_t)absolute;
    byte_count = sample_count * sizeof(float);
    if ((uint64_t)byte_offset != absolute) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 channel read does not fit size_t");
    }
    status = m3_storage_read(output->waveform.storage, byte_offset,
                             destination, byte_count, error);
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_music3_output_write_wav(const m3_music3_output *output,
                                     const char *path,
                                     m3_error *error)
{
    m3_status status = m3_music3_output_validate(output, error);

    if (status == M3_STATUS_OK) {
        status = m3_wav_write_f32(path, &output->waveform.waveform, error);
    }
    return status;
}
