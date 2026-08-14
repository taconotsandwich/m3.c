/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_wav.h"

#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define M3_WAV_HEADER_SIZE 44U
#define M3_WAV_RIFF_BASE_SIZE 36U
#define M3_WAV_FLOAT_SIZE 4U
#define M3_WAV_BUFFER_SAMPLE_CAPACITY 1024U

_Static_assert(sizeof(float) == M3_WAV_FLOAT_SIZE,
               "WAVE output requires 32-bit float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "WAVE output requires IEEE binary32 float");

static void m3_wav_store_u16(uint8_t *output, uint16_t value)
{
    output[0] = (uint8_t)(value & UINT16_C(0xff));
    output[1] = (uint8_t)(value >> 8U);
}

static void m3_wav_store_u32(uint8_t *output, uint32_t value)
{
    output[0] = (uint8_t)(value & UINT32_C(0xff));
    output[1] = (uint8_t)((value >> 8U) & UINT32_C(0xff));
    output[2] = (uint8_t)((value >> 16U) & UINT32_C(0xff));
    output[3] = (uint8_t)(value >> 24U);
}

static bool m3_wav_write_all(FILE *stream, const uint8_t *data, size_t size)
{
    size_t offset = 0U;

    while (offset < size) {
        size_t written = fwrite(data + offset, 1U, size - offset, stream);

        if (written == 0U) {
            return false;
        }
        offset += written;
    }
    return true;
}

static m3_status m3_wav_validate(const char *path, const float *samples,
                                 uint32_t sample_rate,
                                 uint16_t channel_count,
                                 uint64_t frame_count,
                                 uint32_t *data_size,
                                 uint16_t *block_align,
                                 uint32_t *byte_rate, m3_error *error)
{
    uint64_t sample_count;
    uint64_t data_size_u64;
    uint64_t block_align_u64;
    uint64_t byte_rate_u64;

    if (path == NULL || path[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "WAVE output path is empty");
    }
    if (sample_rate == 0U) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "WAVE sample rate must be non-zero");
    }
    if (channel_count == 0U) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "WAVE channel count must be non-zero");
    }
    if (frame_count != 0U && samples == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "WAVE samples are null for non-empty audio");
    }
    if (frame_count > UINT64_MAX / (uint64_t)channel_count) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE sample count overflows uint64_t");
    }
    sample_count = frame_count * (uint64_t)channel_count;
    if (sample_count > UINT64_MAX / M3_WAV_FLOAT_SIZE) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE data size overflows uint64_t");
    }
    data_size_u64 = sample_count * M3_WAV_FLOAT_SIZE;
    if (data_size_u64 > UINT32_MAX - M3_WAV_RIFF_BASE_SIZE) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE data exceeds RIFF size limit");
    }

    block_align_u64 = (uint64_t)channel_count * M3_WAV_FLOAT_SIZE;
    if (block_align_u64 > UINT16_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE block alignment exceeds uint16_t");
    }
    byte_rate_u64 = (uint64_t)sample_rate * block_align_u64;
    if (byte_rate_u64 > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE byte rate exceeds uint32_t");
    }

    *data_size = (uint32_t)data_size_u64;
    *block_align = (uint16_t)block_align_u64;
    *byte_rate = (uint32_t)byte_rate_u64;
    return M3_STATUS_OK;
}

m3_status m3_wav_write_f32(const char *path, const float *samples,
                           uint32_t sample_rate, uint16_t channel_count,
                           uint64_t frame_count, m3_error *error)
{
    uint8_t header[M3_WAV_HEADER_SIZE] = {0U};
    uint8_t sample_buffer[M3_WAV_BUFFER_SAMPLE_CAPACITY * M3_WAV_FLOAT_SIZE];
    uint32_t data_size;
    uint32_t byte_rate;
    uint16_t block_align;
    uint64_t sample_count;
    uint64_t sample_offset = 0U;
    m3_status status;
    FILE *stream;

    status = m3_wav_validate(path, samples, sample_rate, channel_count,
                             frame_count, &data_size, &block_align,
                             &byte_rate, error);
    if (status != M3_STATUS_OK) {
        return status;
    }

    (void)memcpy(header, "RIFF", 4U);
    m3_wav_store_u32(header + 4U, M3_WAV_RIFF_BASE_SIZE + data_size);
    (void)memcpy(header + 8U, "WAVEfmt ", 8U);
    m3_wav_store_u32(header + 16U, 16U);
    m3_wav_store_u16(header + 20U, 3U);
    m3_wav_store_u16(header + 22U, channel_count);
    m3_wav_store_u32(header + 24U, sample_rate);
    m3_wav_store_u32(header + 28U, byte_rate);
    m3_wav_store_u16(header + 32U, block_align);
    m3_wav_store_u16(header + 34U, 32U);
    (void)memcpy(header + 36U, "data", 4U);
    m3_wav_store_u32(header + 40U, data_size);

    stream = fopen(path, "wb");
    if (stream == NULL) {
        return m3_error_set(error, M3_STATUS_IO,
                            "failed to open WAVE output");
    }
    if (!m3_wav_write_all(stream, header, sizeof(header))) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "failed to write WAVE header");
        (void)fclose(stream);
        return status;
    }

    sample_count = frame_count * (uint64_t)channel_count;
    while (sample_offset < sample_count) {
        uint64_t remaining = sample_count - sample_offset;
        size_t chunk_count = remaining > M3_WAV_BUFFER_SAMPLE_CAPACITY
                                 ? M3_WAV_BUFFER_SAMPLE_CAPACITY
                                 : (size_t)remaining;
        size_t index;

        for (index = 0U; index < chunk_count; ++index) {
            uint32_t bits;

            (void)memcpy(&bits, samples + (size_t)sample_offset + index,
                         sizeof(bits));
            m3_wav_store_u32(sample_buffer + index * M3_WAV_FLOAT_SIZE, bits);
        }
        if (!m3_wav_write_all(stream, sample_buffer,
                              chunk_count * M3_WAV_FLOAT_SIZE)) {
            status = m3_error_set(error, M3_STATUS_IO,
                                  "failed to write WAVE samples");
            (void)fclose(stream);
            return status;
        }
        sample_offset += (uint64_t)chunk_count;
    }

    if (fclose(stream) != 0) {
        return m3_error_set(error, M3_STATUS_IO,
                            "failed to close WAVE output");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}
