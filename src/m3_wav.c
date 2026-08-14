/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_wav_internal.h"

#include "m3_backend.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define M3_WAV_HEADER_SIZE 44U
#define M3_WAV_RIFF_BASE_SIZE 36U
#define M3_WAV_FLOAT_SIZE 4U
#define M3_WAV_CHANNEL_COUNT 2U
#define M3_WAV_SAMPLE_RATE 44100U
#define M3_WAV_BLOCK_ALIGN 8U
#define M3_WAV_BYTE_RATE 352800U
#define M3_WAV_FRAME_CAPACITY 512U

_Static_assert(sizeof(float) == M3_WAV_FLOAT_SIZE,
               "WAVE output requires 32-bit float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "WAVE output requires IEEE binary32 float");

typedef struct {
    uint32_t data_size;
    uint64_t frame_count;
    size_t left_offset;
    size_t right_offset;
} m3_wav_layout;

static int m3_wav_posix_create(void *context, char *path_template)
{
    (void)context;
    if (path_template == NULL) {
        errno = EINVAL;
        return -1;
    }
    return mkstemp(path_template);
}

static ssize_t m3_wav_posix_write(void *context, int descriptor,
                                  const void *data, size_t size)
{
    (void)context;
    return write(descriptor, data, size);
}

static int m3_wav_posix_sync(void *context, int descriptor)
{
    (void)context;
    return fsync(descriptor);
}

static int m3_wav_posix_close(void *context, int descriptor)
{
    (void)context;
    return close(descriptor);
}

static int m3_wav_posix_replace(void *context, const char *source,
                                const char *destination)
{
    (void)context;
    return rename(source, destination);
}

static int m3_wav_posix_remove(void *context, const char *path)
{
    (void)context;
    return unlink(path);
}

const m3_wav_io *m3_wav_default_io(void)
{
    static const m3_wav_io io = {
        NULL,
        m3_wav_posix_create,
        m3_wav_posix_write,
        m3_wav_posix_sync,
        m3_wav_posix_close,
        m3_wav_posix_replace,
        m3_wav_posix_remove
    };

    return &io;
}

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

static bool m3_wav_io_valid(const m3_wav_io *io)
{
    return io != NULL && io->create_temporary != NULL &&
           io->write_bytes != NULL && io->sync_file != NULL &&
           io->close_file != NULL && io->replace_file != NULL &&
           io->remove_file != NULL;
}

static m3_status m3_wav_validate(const char *path,
                                 const m3_tensor_view *samples,
                                 const m3_wav_io *io,
                                 m3_wav_layout *layout,
                                 m3_error *error)
{
    const uint64_t maximum_frames =
        ((uint64_t)UINT32_MAX - M3_WAV_RIFF_BASE_SIZE) /
        M3_WAV_BLOCK_ALIGN;
    m3_tensor_view checked;
    uint64_t frame_count;
    uint64_t plane_bytes;
    uint64_t data_size;
    m3_status status;

    if (path == NULL || path[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "WAVE output path is empty");
    }
    if (!m3_wav_io_valid(io) || layout == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "WAVE output I/O is invalid");
    }
    if (samples == NULL || samples->metadata.dtype != M3_DTYPE_F32 ||
        samples->metadata.rank != 3U ||
        samples->metadata.shape[0] != 1U ||
        samples->metadata.shape[1] != M3_WAV_CHANNEL_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "WAVE samples must be planar F32 [1,2,N]");
    }
    frame_count = samples->metadata.shape[2];
    if (frame_count == 0U) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "WAVE frame count must be non-zero");
    }
    if (frame_count > maximum_frames) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE data exceeds RIFF size limit");
    }
    m3_tensor_view_init(&checked);
    status = m3_tensor_reshape(samples, 3U, samples->metadata.shape,
                               &checked, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (m3_storage_backend(samples->storage) == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "WAVE sample storage has no backend");
    }
    plane_bytes = frame_count * M3_WAV_FLOAT_SIZE;
    data_size = frame_count * M3_WAV_BLOCK_ALIGN;
    if (plane_bytes > SIZE_MAX ||
        samples->byte_offset > SIZE_MAX - (size_t)plane_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE planar sample offset overflows");
    }
    layout->data_size = (uint32_t)data_size;
    layout->frame_count = frame_count;
    layout->left_offset = samples->byte_offset;
    layout->right_offset = samples->byte_offset + (size_t)plane_bytes;
    return M3_STATUS_OK;
}

static void m3_wav_header(uint8_t header[M3_WAV_HEADER_SIZE],
                          uint32_t data_size)
{
    (void)memset(header, 0, M3_WAV_HEADER_SIZE);
    (void)memcpy(header, "RIFF", 4U);
    m3_wav_store_u32(header + 4U, M3_WAV_RIFF_BASE_SIZE + data_size);
    (void)memcpy(header + 8U, "WAVEfmt ", 8U);
    m3_wav_store_u32(header + 16U, 16U);
    m3_wav_store_u16(header + 20U, 3U);
    m3_wav_store_u16(header + 22U, M3_WAV_CHANNEL_COUNT);
    m3_wav_store_u32(header + 24U, M3_WAV_SAMPLE_RATE);
    m3_wav_store_u32(header + 28U, M3_WAV_BYTE_RATE);
    m3_wav_store_u16(header + 32U, M3_WAV_BLOCK_ALIGN);
    m3_wav_store_u16(header + 34U, 32U);
    (void)memcpy(header + 36U, "data", 4U);
    m3_wav_store_u32(header + 40U, data_size);
}

static m3_status m3_wav_write_all(const m3_wav_io *io, int descriptor,
                                  const uint8_t *data, size_t size,
                                  m3_error *error)
{
    size_t offset = 0U;

    while (offset < size) {
        ssize_t written = io->write_bytes(io->context, descriptor,
                                          data + offset, size - offset);

        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0 || (size_t)written > size - offset) {
            return m3_error_set(error, M3_STATUS_IO,
                                "failed to write WAVE output");
        }
        offset += (size_t)written;
    }
    return M3_STATUS_OK;
}

static void m3_wav_abort(const m3_wav_io *io, int descriptor,
                         const char *temporary_path)
{
    if (descriptor >= 0) {
        (void)io->close_file(io->context, descriptor);
    }
    if (temporary_path != NULL && temporary_path[0] != '\0') {
        (void)io->remove_file(io->context, temporary_path);
    }
}

static m3_status m3_wav_write_samples(
    const m3_tensor_view *samples, const m3_wav_layout *layout,
    const m3_wav_io *io, int descriptor, m3_error *error)
{
    uint32_t left[M3_WAV_FRAME_CAPACITY];
    uint32_t right[M3_WAV_FRAME_CAPACITY];
    uint8_t interleaved[M3_WAV_FRAME_CAPACITY * M3_WAV_BLOCK_ALIGN];
    uint64_t completed = 0U;

    while (completed < layout->frame_count) {
        uint64_t remaining = layout->frame_count - completed;
        size_t count = remaining > M3_WAV_FRAME_CAPACITY
                           ? M3_WAV_FRAME_CAPACITY
                           : (size_t)remaining;
        size_t plane_offset = (size_t)completed * M3_WAV_FLOAT_SIZE;
        size_t byte_count = count * M3_WAV_FLOAT_SIZE;
        size_t index;
        m3_status status = m3_storage_read(
            samples->storage, layout->left_offset + plane_offset,
            left, byte_count, error);

        if (status == M3_STATUS_OK) {
            status = m3_storage_read(
                samples->storage, layout->right_offset + plane_offset,
                right, byte_count, error);
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
        for (index = 0U; index < count; ++index) {
            size_t output = index * M3_WAV_BLOCK_ALIGN;

            m3_wav_store_u32(interleaved + output, left[index]);
            m3_wav_store_u32(interleaved + output + M3_WAV_FLOAT_SIZE,
                             right[index]);
        }
        status = m3_wav_write_all(io, descriptor, interleaved,
                                  count * M3_WAV_BLOCK_ALIGN, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        completed += (uint64_t)count;
    }
    return M3_STATUS_OK;
}

static m3_status m3_wav_temporary_path(const char *path, char **output,
                                       m3_error *error)
{
    static const char suffix[] = ".tmp.XXXXXX";
    size_t length = strlen(path);
    char *temporary;

    *output = NULL;
    if (length > SIZE_MAX - sizeof(suffix)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "WAVE temporary path length overflows");
    }
    temporary = malloc(length + sizeof(suffix));
    if (temporary == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate WAVE temporary path");
    }
    (void)memcpy(temporary, path, length);
    (void)memcpy(temporary + length, suffix, sizeof(suffix));
    *output = temporary;
    return M3_STATUS_OK;
}

m3_status m3_wav_write_f32_with_io(const char *path,
                                   const m3_tensor_view *samples,
                                   const m3_wav_io *io,
                                   m3_error *error)
{
    uint8_t header[M3_WAV_HEADER_SIZE];
    m3_wav_layout layout = {0};
    char *temporary_path = NULL;
    int descriptor = -1;
    m3_status status = m3_wav_validate(path, samples, io, &layout, error);

    if (status == M3_STATUS_OK) {
        status = m3_wav_temporary_path(path, &temporary_path, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    descriptor = io->create_temporary(io->context, temporary_path);
    if (descriptor < 0) {
        free(temporary_path);
        return m3_error_set(error, M3_STATUS_IO,
                            "failed to create WAVE temporary file");
    }
    m3_wav_header(header, layout.data_size);
    status = m3_wav_write_all(io, descriptor, header, sizeof(header), error);
    if (status == M3_STATUS_OK) {
        status = m3_wav_write_samples(samples, &layout, io, descriptor,
                                      error);
    }
    if (status == M3_STATUS_OK &&
        io->sync_file(io->context, descriptor) != 0) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "failed to synchronize WAVE output");
    }
    if (status == M3_STATUS_OK) {
        int close_result = io->close_file(io->context, descriptor);

        descriptor = -1;
        if (close_result != 0) {
            status = m3_error_set(error, M3_STATUS_IO,
                                  "failed to close WAVE output");
        }
    }
    if (status == M3_STATUS_OK &&
        io->replace_file(io->context, temporary_path, path) != 0) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "failed to replace WAVE output");
    }
    if (status != M3_STATUS_OK) {
        m3_wav_abort(io, descriptor, temporary_path);
        free(temporary_path);
        return status;
    }
    free(temporary_path);
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_wav_write_f32(const char *path,
                           const m3_tensor_view *samples,
                           m3_error *error)
{
    return m3_wav_write_f32_with_io(path, samples, m3_wav_default_io(),
                                    error);
}
