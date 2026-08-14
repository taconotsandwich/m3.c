/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend.h"
#include "m3_wav.h"
#include "m3_wav_internal.h"
#include "test_cases.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const uint32_t m3_wav_samples[] = {
    UINT32_C(0x3f800000), UINT32_C(0x80000000),
    UINT32_C(0x7fc12345), UINT32_C(0x7f800000),
    UINT32_C(0xbf000000), UINT32_C(0x00000000),
    UINT32_C(0x80000001), UINT32_C(0xff800000)
};

static const uint8_t m3_wav_expected[] = {
    0x52U, 0x49U, 0x46U, 0x46U, 0x44U, 0x00U, 0x00U, 0x00U,
    0x57U, 0x41U, 0x56U, 0x45U, 0x66U, 0x6dU, 0x74U, 0x20U,
    0x10U, 0x00U, 0x00U, 0x00U, 0x03U, 0x00U, 0x02U, 0x00U,
    0x44U, 0xacU, 0x00U, 0x00U, 0x20U, 0x62U, 0x05U, 0x00U,
    0x08U, 0x00U, 0x20U, 0x00U, 0x64U, 0x61U, 0x74U, 0x61U,
    0x20U, 0x00U, 0x00U, 0x00U,
    0x00U, 0x00U, 0x80U, 0x3fU, 0x00U, 0x00U, 0x00U, 0xbfU,
    0x00U, 0x00U, 0x00U, 0x80U, 0x00U, 0x00U, 0x00U, 0x00U,
    0x45U, 0x23U, 0xc1U, 0x7fU, 0x01U, 0x00U, 0x00U, 0x80U,
    0x00U, 0x00U, 0x80U, 0x7fU, 0x00U, 0x00U, 0x80U, 0xffU
};

static const uint8_t m3_wav_old_bytes[] = {
    0x6fU, 0x6cU, 0x64U, 0x2dU, 0x77U, 0x61U, 0x76U
};

static bool m3_wav_test_file_equals(const char *path,
                                    const uint8_t *expected, size_t size)
{
    uint8_t buffer[sizeof(m3_wav_expected)];
    FILE *stream;
    bool matches;

    if (size > sizeof(buffer)) {
        return false;
    }
    stream = fopen(path, "rb");
    if (stream == NULL) {
        return false;
    }
    matches = fread(buffer, 1U, size, stream) == size &&
              fgetc(stream) == EOF && ferror(stream) == 0 &&
              memcmp(buffer, expected, size) == 0;
    if (fclose(stream) != 0) {
        matches = false;
    }
    return matches;
}

static bool m3_wav_test_tensor(m3_backend *backend,
                               m3_storage **storage,
                               m3_tensor_view *view, m3_error *error)
{
    const uint64_t shape[] = {1U, 2U, 4U};

    *storage = NULL;
    m3_tensor_view_init(view);
    return m3_storage_allocate(backend, sizeof(m3_wav_samples), 16U,
                               storage, error) == M3_STATUS_OK &&
           m3_storage_write(*storage, 0U, m3_wav_samples,
                            sizeof(m3_wav_samples), error) == M3_STATUS_OK &&
           m3_tensor_view_contiguous(view, *storage, M3_DTYPE_F32, 3U,
                                     shape, 0U, error) == M3_STATUS_OK;
}

void m3_test_wav_planar_bytes(m3_test_context *test)
{
    static const m3_test_fixture old_fixture = {
        "old WAVE destination", m3_wav_old_bytes, sizeof(m3_wav_old_bytes)
    };
    m3_backend *backend = NULL;
    m3_storage *storage = NULL;
    m3_tensor_view samples;
    m3_test_temp_file file = {{0}};
    m3_error error = {M3_STATUS_INTERNAL, "stale"};

    m3_tensor_view_init(&samples);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_wav_test_tensor(backend, &storage, &samples,
                                          &error),
                   "create host planar WAVE tensor");
    M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &old_fixture),
                   "create existing WAVE destination");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &samples, &error) ==
                       M3_STATUS_OK,
                   "atomically replace destination with planar WAVE");
    M3_TEST_EXPECT(test, error.status == M3_STATUS_OK,
                   "successful planar WAVE write clears error");
    M3_TEST_EXPECT(test,
                   m3_wav_test_file_equals(file.path, m3_wav_expected,
                                           sizeof(m3_wav_expected)),
                   "WAVE header and planar interleave are byte exact");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &samples, NULL) ==
                       M3_STATUS_OK,
                   "write planar WAVE with null error output");
    M3_TEST_EXPECT(test,
                   m3_wav_test_file_equals(file.path, m3_wav_expected,
                                           sizeof(m3_wav_expected)),
                   "null-error write remains byte exact");

    M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                   "remove planar WAVE destination");
    m3_storage_free(storage);
    m3_backend_free(backend);
}

void m3_test_wav_validation(m3_test_context *test)
{
    static const m3_test_fixture old_fixture = {
        "old WAVE destination", m3_wav_old_bytes, sizeof(m3_wav_old_bytes)
    };
    const uint64_t strided_shape[] = {1U, 2U, 4U};
    const size_t strided_strides[] = {40U, 20U, 4U};
    const uint64_t large_frames = UINT64_C(536870908);
    const uint64_t large_shape[] = {1U, 2U, UINT64_C(536870908)};
    m3_backend *backend = NULL;
    m3_storage *storage = NULL;
    m3_storage *strided_storage = NULL;
    m3_tensor_view samples;
    m3_tensor_view invalid;
    m3_tensor_view strided;
    m3_test_temp_file file = {{0}};
    m3_error error = {0};
    char invalid_path[M3_TEST_TEMP_PATH_CAPACITY + 16U] = {0};
    int path_length;

    m3_tensor_view_init(&samples);
    m3_tensor_view_init(&strided);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_wav_test_tensor(backend, &storage, &samples,
                                          &error),
                   "create validation WAVE tensor");
    M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &old_fixture),
                   "create validation WAVE destination");

    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(NULL, &samples, &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       m3_wav_write_f32("", &samples, &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       m3_wav_write_f32(file.path, NULL, &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       m3_wav_write_f32(NULL, &samples, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT,
                   "reject invalid WAVE pointers with optional error");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32_with_io(file.path, &samples, NULL,
                                            &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null WAVE I/O contract");

    invalid = samples;
    invalid.metadata.dtype = M3_DTYPE_I32;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject non-F32 WAVE samples");
    invalid = samples;
    invalid.metadata.rank = 2U;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject non-rank-three WAVE samples");
    invalid = samples;
    invalid.metadata.shape[0] = 2U;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject WAVE batch other than one");
    invalid = samples;
    invalid.metadata.shape[1] = 1U;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject WAVE channels other than stereo");
    invalid = samples;
    invalid.metadata.shape[2] = 0U;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject empty WAVE tensor");

    M3_TEST_EXPECT(test,
                   m3_storage_allocate(backend, 40U, 16U,
                                       &strided_storage, &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_strided(
                           &strided, strided_storage, M3_DTYPE_F32, 3U,
                           strided_shape, strided_strides, 0U, &error) ==
                           M3_STATUS_OK,
                   "create valid non-contiguous WAVE tensor");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &strided, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject non-contiguous WAVE tensor");

    invalid = samples;
    invalid.storage = NULL;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject WAVE tensor without backend storage");
    invalid = samples;
    invalid.byte_offset = 1U;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject misaligned WAVE tensor offset");
    invalid = samples;
    invalid.byte_offset = m3_storage_size(storage);
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject WAVE tensor outside storage range");
    invalid = samples;
    --invalid.metadata.byte_count;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject inconsistent WAVE tensor metadata");

    invalid = samples;
    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&invalid.metadata,
                                           M3_DTYPE_F32, 3U, large_shape,
                                           &error) == M3_STATUS_OK,
                   "construct RIFF boundary metadata");
    invalid.byte_strides[2] = 4U;
    invalid.byte_strides[1] = (size_t)large_frames * 4U;
    invalid.byte_strides[0] = (size_t)large_frames * 8U;
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &invalid, &error) ==
                       M3_STATUS_OVERFLOW,
                   "reject WAVE data beyond classic RIFF limit");

    path_length = snprintf(invalid_path, sizeof(invalid_path),
                           "%s/child.wav", file.path);
    M3_TEST_EXPECT(test,
                   path_length > 0 &&
                       (size_t)path_length < sizeof(invalid_path),
                   "construct invalid WAVE destination path");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(invalid_path, &samples, &error) ==
                           M3_STATUS_IO &&
                       error.status == M3_STATUS_IO &&
                       m3_error_message(&error)[0] != '\0',
                   "report temporary-file creation failure");
    M3_TEST_EXPECT(test,
                   m3_wav_test_file_equals(file.path, m3_wav_old_bytes,
                                           sizeof(m3_wav_old_bytes)),
                   "validation failures preserve existing destination");

    M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                   "remove validation WAVE destination");
    m3_storage_free(strided_storage);
    m3_storage_free(storage);
    m3_backend_free(backend);
}

typedef enum {
    M3_WAV_FAULT_NONE = 0,
    M3_WAV_FAULT_CREATE,
    M3_WAV_FAULT_SHORT_WRITE,
    M3_WAV_FAULT_ZERO_WRITE,
    M3_WAV_FAULT_SYNC,
    M3_WAV_FAULT_CLOSE,
    M3_WAV_FAULT_RENAME
} m3_wav_fault_kind;

typedef struct {
    m3_wav_fault_kind kind;
    char temporary_path[M3_TEST_TEMP_PATH_CAPACITY + 32U];
    int descriptor;
    size_t write_calls;
} m3_wav_fault;

static int m3_wav_fault_create(void *context, char *path_template)
{
    m3_wav_fault *fault = context;
    int descriptor;

    if (fault->kind == M3_WAV_FAULT_CREATE) {
        errno = EIO;
        return -1;
    }
    descriptor = mkstemp(path_template);
    fault->descriptor = descriptor;
    if (descriptor >= 0) {
        size_t length = strlen(path_template);

        if (length < sizeof(fault->temporary_path)) {
            (void)memcpy(fault->temporary_path, path_template, length + 1U);
        }
    }
    return descriptor;
}

static ssize_t m3_wav_fault_write(void *context, int descriptor,
                                  const void *data, size_t size)
{
    m3_wav_fault *fault = context;

    ++fault->write_calls;
    if (fault->kind == M3_WAV_FAULT_ZERO_WRITE &&
        fault->write_calls == 1U) {
        return 0;
    }
    if (fault->kind == M3_WAV_FAULT_SHORT_WRITE &&
        fault->write_calls == 1U && size > 3U) {
        return write(descriptor, data, 3U);
    }
    return write(descriptor, data, size);
}

static int m3_wav_fault_sync(void *context, int descriptor)
{
    m3_wav_fault *fault = context;

    if (fault->kind == M3_WAV_FAULT_SYNC) {
        errno = EIO;
        return -1;
    }
    return fsync(descriptor);
}

static int m3_wav_fault_close(void *context, int descriptor)
{
    m3_wav_fault *fault = context;
    int result = close(descriptor);

    if (fault->kind == M3_WAV_FAULT_CLOSE) {
        errno = EIO;
        return -1;
    }
    return result;
}

static int m3_wav_fault_replace(void *context, const char *source,
                                const char *destination)
{
    m3_wav_fault *fault = context;

    if (fault->kind == M3_WAV_FAULT_RENAME) {
        errno = EIO;
        return -1;
    }
    return rename(source, destination);
}

static int m3_wav_fault_remove(void *context, const char *path)
{
    (void)context;
    return unlink(path);
}

static m3_wav_io m3_wav_fault_io(m3_wav_fault *fault)
{
    const m3_wav_io io = {
        fault,
        m3_wav_fault_create,
        m3_wav_fault_write,
        m3_wav_fault_sync,
        m3_wav_fault_close,
        m3_wav_fault_replace,
        m3_wav_fault_remove
    };

    return io;
}

static bool m3_wav_test_fault_clean(const m3_wav_fault *fault)
{
    bool descriptor_closed =
        fault->descriptor < 0 ||
        (fcntl(fault->descriptor, F_GETFD) == -1 && errno == EBADF);
    bool temporary_removed =
        fault->temporary_path[0] == '\0' ||
        (access(fault->temporary_path, F_OK) == -1 && errno == ENOENT);

    return descriptor_closed && temporary_removed;
}

void m3_test_wav_atomic_failures(m3_test_context *test)
{
    static const m3_test_fixture old_fixture = {
        "old WAVE destination", m3_wav_old_bytes, sizeof(m3_wav_old_bytes)
    };
    static const m3_wav_fault_kind failures[] = {
        M3_WAV_FAULT_CREATE, M3_WAV_FAULT_ZERO_WRITE,
        M3_WAV_FAULT_SYNC, M3_WAV_FAULT_CLOSE, M3_WAV_FAULT_RENAME
    };
    m3_backend *backend = NULL;
    m3_storage *storage = NULL;
    m3_tensor_view samples;
    m3_error error = {0};
    size_t index;

    m3_tensor_view_init(&samples);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_wav_test_tensor(backend, &storage, &samples,
                                          &error),
                   "create atomic WAVE tensor");
    for (index = 0U; index < sizeof(failures) / sizeof(failures[0]);
         ++index) {
        m3_test_temp_file file = {{0}};
        m3_wav_fault fault = {failures[index], {0}, -1, 0U};
        m3_wav_io io = m3_wav_fault_io(&fault);

        M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &old_fixture),
                       "create atomic-failure WAVE destination");
        M3_TEST_EXPECT(test,
                       m3_wav_write_f32_with_io(file.path, &samples, &io,
                                                &error) == M3_STATUS_IO,
                       "propagate injected pre-rename WAVE failure");
        M3_TEST_EXPECT(test, m3_wav_test_fault_clean(&fault),
                       "close descriptor and remove temporary WAVE file");
        M3_TEST_EXPECT(test,
                       m3_wav_test_file_equals(
                           file.path, m3_wav_old_bytes,
                           sizeof(m3_wav_old_bytes)),
                       "pre-rename failure preserves old WAVE destination");
        M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                       "remove atomic-failure WAVE destination");
    }
    {
        m3_test_temp_file file = {{0}};
        m3_wav_fault fault = {M3_WAV_FAULT_SHORT_WRITE, {0}, -1, 0U};
        m3_wav_io io = m3_wav_fault_io(&fault);

        M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &old_fixture),
                       "create short-write WAVE destination");
        M3_TEST_EXPECT(test,
                       m3_wav_write_f32_with_io(file.path, &samples, &io,
                                                &error) == M3_STATUS_OK &&
                           fault.write_calls >= 3U,
                       "complete WAVE after a short write");
        M3_TEST_EXPECT(test, m3_wav_test_fault_clean(&fault),
                       "close renamed short-write temporary descriptor");
        M3_TEST_EXPECT(test,
                       m3_wav_test_file_equals(file.path, m3_wav_expected,
                                               sizeof(m3_wav_expected)),
                       "short write retains exact WAVE bytes");
        M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                       "remove short-write WAVE destination");
    }

    m3_storage_free(storage);
    m3_backend_free(backend);
}

void m3_test_wav_metal_storage(m3_test_context *test)
{
    static const m3_test_fixture old_fixture = {
        "old WAVE destination", m3_wav_old_bytes, sizeof(m3_wav_old_bytes)
    };
    m3_backend *backend = NULL;
    m3_storage *storage = NULL;
    m3_tensor_view samples;
    m3_test_temp_file file = {{0}};
    m3_error error = {0};
    m3_status status = m3_backend_create_metal(&backend, &error);

    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, "Metal has no default device");
        return;
    }
    m3_tensor_view_init(&samples);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK && backend != NULL,
                   "create Metal backend for planar WAVE");
    if (backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_wav_test_tensor(backend, &storage, &samples, &error),
                   "create Metal planar WAVE tensor");
    M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &old_fixture),
                   "create Metal WAVE destination");
    M3_TEST_EXPECT(test,
                   m3_wav_write_f32(file.path, &samples, &error) ==
                       M3_STATUS_OK,
                   "stream planar WAVE from Metal storage");
    M3_TEST_EXPECT(test,
                   m3_wav_test_file_equals(file.path, m3_wav_expected,
                                           sizeof(m3_wav_expected)),
                   "Metal storage WAVE bytes are exact");
    M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                   "remove Metal WAVE destination");

    m3_storage_free(storage);
    m3_backend_free(backend);
}
