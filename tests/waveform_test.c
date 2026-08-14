/* SPDX-License-Identifier: GPL-2.0-only */

#include "waveform_test.h"

#include <stdlib.h>
#include <string.h>

static size_t m3_waveform_test_count(uint64_t frames)
{
    return frames <= 200U ? 1U : (size_t)((frames - 1U) / 100U);
}

static uint64_t m3_waveform_test_frames(
    uint64_t frames, size_t index, size_t count)
{
    return index + 1U < count ? 200U : frames - 100U * (uint64_t)index;
}

static size_t m3_waveform_test_samples(
    uint64_t frames, size_t index, size_t count)
{
    uint64_t window = m3_waveform_test_frames(frames, index, count);

    return (size_t)(window * 441U / 128U) * 512U;
}

void m3_waveform_test_fixture_init(m3_waveform_test_fixture *fixture)
{
    if (fixture != NULL) {
        (void)memset(fixture, 0, sizeof(*fixture));
    }
}

uint32_t m3_waveform_test_pattern(
    size_t chunk, size_t channel, size_t sample)
{
    uint32_t mixed =
        (uint32_t)(chunk * 1000003U + channel * 2000003U + sample);

    return UINT32_C(0x3f000000) | (mixed & UINT32_C(0x007fffff));
}

void m3_waveform_test_fixture_dispose(m3_waveform_test_fixture *fixture)
{
    size_t index;

    if (fixture == NULL) {
        return;
    }
    for (index = 0U; index < fixture->chunk_count; ++index) {
        m3_storage_free(fixture->storages[index]);
    }
    m3_waveform_test_fixture_init(fixture);
}

bool m3_waveform_test_fixture_create(
    m3_waveform_test_fixture *fixture, m3_backend *backend,
    uint64_t frame_count, m3_error *error)
{
    size_t count;
    size_t index;

    if (fixture == NULL || backend == NULL || frame_count == 0U ||
        frame_count > 9000U) {
        return false;
    }
    m3_waveform_test_fixture_init(fixture);
    fixture->backend = backend;
    fixture->frame_count = frame_count;
    count = m3_waveform_test_count(frame_count);
    fixture->chunk_count = count;
    for (index = 0U; index < count; ++index) {
        size_t samples = m3_waveform_test_samples(
            frame_count, index, count);
        size_t value_count = samples * 2U;
        size_t byte_count = value_count * sizeof(uint32_t);
        uint64_t shape[] = {1U, 2U, (uint64_t)samples};
        uint32_t *values = malloc(byte_count);
        size_t channel;
        m3_status status;

        if (values == NULL) {
            m3_waveform_test_fixture_dispose(fixture);
            return false;
        }
        for (channel = 0U; channel < 2U; ++channel) {
            size_t sample;

            for (sample = 0U; sample < samples; ++sample) {
                values[channel * samples + sample] =
                    m3_waveform_test_pattern(index, channel, sample);
            }
        }
        status = m3_storage_allocate(
            backend, byte_count, 64U, &fixture->storages[index], error);
        if (status == M3_STATUS_OK) {
            status = m3_tensor_view_contiguous(
                &fixture->chunks[index], fixture->storages[index],
                M3_DTYPE_F32, 3U, shape, 0U, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_storage_write(
                fixture->storages[index], 0U, values, byte_count, error);
        }
        free(values);
        if (status != M3_STATUS_OK) {
            m3_waveform_test_fixture_dispose(fixture);
            return false;
        }
    }
    return true;
}

size_t m3_waveform_test_output_samples(uint64_t frame_count)
{
    size_t count = m3_waveform_test_count(frame_count);
    size_t latents = 0U;
    size_t index;

    for (index = 0U; index < count; ++index) {
        uint64_t frames = m3_waveform_test_frames(
            frame_count, index, count);

        latents += (size_t)(frames * 441U / 128U);
    }
    latents -= 344U * (count - 1U);
    return latents * 512U;
}

bool m3_waveform_test_output_matches(
    const m3_waveform_test_fixture *fixture,
    const m3_vocoder_output *output, m3_error *error)
{
    size_t output_samples = m3_waveform_test_output_samples(
        fixture->frame_count);
    size_t value_count = output_samples * 2U;
    uint32_t *actual = malloc(value_count * sizeof(*actual));
    size_t channel;
    bool matches = actual != NULL && output != NULL &&
                   output->storage != NULL &&
                   output->waveform.storage == output->storage &&
                   output->waveform.metadata.dtype == M3_DTYPE_F32 &&
                   output->waveform.metadata.rank == 3U &&
                   output->waveform.metadata.shape[0] == 1U &&
                   output->waveform.metadata.shape[1] == 2U &&
                   output->waveform.metadata.shape[2] == output_samples &&
                   m3_tensor_is_contiguous(&output->waveform);

    if (matches) {
        matches = m3_storage_read(
                      output->storage, output->waveform.byte_offset,
                      actual, value_count * sizeof(*actual), error) ==
                  M3_STATUS_OK;
    }
    for (channel = 0U; channel < 2U && matches; ++channel) {
        size_t destination = 0U;
        size_t chunk;

        for (chunk = 0U; chunk < fixture->chunk_count && matches; ++chunk) {
            size_t source_samples = (size_t)
                fixture->chunks[chunk].metadata.shape[2];
            size_t begin = chunk == 0U ? 0U : 86U * 512U;
            size_t end = chunk + 1U == fixture->chunk_count
                             ? source_samples
                             : source_samples - 258U * 512U;
            size_t sample;

            for (sample = begin; sample < end; ++sample) {
                uint32_t expected = m3_waveform_test_pattern(
                    chunk, channel, sample);

                if (actual[channel * output_samples + destination] !=
                    expected) {
                    matches = false;
                    break;
                }
                ++destination;
            }
        }
        matches = matches && destination == output_samples;
    }
    free(actual);
    return matches;
}

bool m3_waveform_test_output_create(
    m3_backend *backend, const uint32_t bits[2U],
    m3_vocoder_output *output, m3_error *error)
{
    const uint64_t shape[] = {1U, 2U, 1U};
    m3_status status;

    m3_vocoder_output_init(output);
    status = m3_storage_allocate(
        backend, 2U * sizeof(uint32_t), 64U, &output->storage, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &output->waveform, output->storage, M3_DTYPE_F32, 3U, shape,
            0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_write(
            output->storage, 0U, bits, 2U * sizeof(uint32_t), error);
    }
    if (status != M3_STATUS_OK) {
        m3_vocoder_output_dispose(output);
    }
    return status == M3_STATUS_OK;
}
