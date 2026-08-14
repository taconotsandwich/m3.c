/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "weight_stage_fixture.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    M3_STAGE_IO_EINTR = 0,
    M3_STAGE_IO_SHORT,
    M3_STAGE_IO_EOF,
    M3_STAGE_IO_ERROR,
    M3_STAGE_IO_WRITE_ERROR,
    M3_STAGE_IO_MUTATE,
    M3_STAGE_IO_PATH_SWAP,
    M3_STAGE_IO_CLOSE_ERROR
} m3_stage_io_mode;

typedef struct {
    m3_stage_io_mode mode;
    size_t trigger_call;
    size_t open_calls;
    size_t pread_calls;
    size_t maximum_request;
    size_t write_calls;
    size_t close_calls;
    bool triggered;
    bool flags_valid;
    bool observed_data;
    bool fail_cleanup_close;
    uint8_t first_byte;
    char target_path[M3_TEST_PATH_CAPACITY];
    char replacement_path[M3_TEST_PATH_CAPACITY];
} m3_stage_io_fault;

static int m3_stage_fault_open(void *context_pointer, const char *path,
                               int flags)
{
    m3_stage_io_fault *context = context_pointer;
    int expected_flags = O_RDONLY | O_NONBLOCK;

#ifdef O_CLOEXEC
    expected_flags |= O_CLOEXEC;
#endif
    ++context->open_calls;
    context->flags_valid = context->flags_valid && flags == expected_flags;
    return open(path, flags);
}

static bool m3_stage_fault_touch_descriptor(int descriptor)
{
    struct stat file_stat;
    struct timespec times[2];

    if (fstat(descriptor, &file_stat) != 0) {
        return false;
    }
    times[0].tv_sec = 0;
    times[0].tv_nsec = UTIME_OMIT;
#if defined(__APPLE__) && defined(_POSIX_C_SOURCE) && \
    !defined(_DARWIN_C_SOURCE)
    times[1].tv_sec = file_stat.st_mtime;
    times[1].tv_nsec = file_stat.st_mtimensec;
#elif defined(__APPLE__)
    times[1] = file_stat.st_mtimespec;
#else
    times[1] = file_stat.st_mtim;
#endif
    ++times[1].tv_sec;
    return futimens(descriptor, times) == 0;
}

static ssize_t m3_stage_fault_pread(void *context_pointer, int descriptor,
                                    void *buffer, size_t byte_count,
                                    off_t byte_offset)
{
    m3_stage_io_fault *context = context_pointer;
    size_t request = byte_count;
    bool trigger;

    ++context->pread_calls;
    if (byte_count > context->maximum_request) {
        context->maximum_request = byte_count;
    }
    trigger = !context->triggered &&
              context->pread_calls == context->trigger_call;
    if (trigger) {
        switch (context->mode) {
        case M3_STAGE_IO_EINTR:
            context->triggered = true;
            errno = EINTR;
            return -1;
        case M3_STAGE_IO_SHORT:
            context->triggered = true;
            if (request > 1U) {
                request /= 2U;
            }
            break;
        case M3_STAGE_IO_EOF:
            context->triggered = true;
            return 0;
        case M3_STAGE_IO_ERROR:
            context->triggered = true;
            errno = EIO;
            return -1;
        case M3_STAGE_IO_MUTATE:
            context->triggered = true;
            if (!m3_stage_fault_touch_descriptor(descriptor)) {
                errno = EIO;
                return -1;
            }
            break;
        case M3_STAGE_IO_PATH_SWAP:
            context->triggered = true;
            if (rename(context->replacement_path, context->target_path) !=
                0) {
                return -1;
            }
            break;
        case M3_STAGE_IO_WRITE_ERROR:
        case M3_STAGE_IO_CLOSE_ERROR:
            break;
        }
    }
    {
        ssize_t result = pread(descriptor, buffer, request, byte_offset);

        if (result > 0 && !context->observed_data) {
            context->observed_data = true;
            context->first_byte = *(const uint8_t *)buffer;
        }
        return result;
    }
}

static int m3_stage_fault_close(void *context_pointer, int descriptor)
{
    m3_stage_io_fault *context = context_pointer;
    int result;

    ++context->close_calls;
    result = close(descriptor);
    if (context->fail_cleanup_close && context->close_calls == 1U) {
        errno = EIO;
        return -1;
    }
    if (!context->triggered && context->mode == M3_STAGE_IO_CLOSE_ERROR &&
        context->close_calls == context->trigger_call) {
        context->triggered = true;
        errno = EIO;
        return -1;
    }
    return result;
}

static m3_status m3_stage_fault_write(
    void *context_pointer, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    m3_stage_io_fault *context = context_pointer;

    ++context->write_calls;
    if (!context->triggered && context->mode == M3_STAGE_IO_WRITE_ERROR &&
        context->write_calls == context->trigger_call) {
        context->triggered = true;
        return m3_error_set(error, M3_STATUS_IO,
                            "injected staged storage write failure");
    }
    return m3_storage_write(storage, byte_offset, source, byte_count, error);
}

static void m3_stage_fault_io(m3_weight_stage_io *io,
                              m3_stage_io_fault *context)
{
    m3_weight_stage_io_init(io);
    io->context = context;
    io->open_file = m3_stage_fault_open;
    io->pread_file = m3_stage_fault_pread;
    io->close_file = m3_stage_fault_close;
    io->write_storage = m3_stage_fault_write;
    io->maximum_chunk_bytes = 4U;
}

static bool m3_stage_prepare_swap(m3_weight_stage_test_fixture *fixture,
                                  m3_stage_io_fault *fault)
{
    static const char a_header[] =
        "{\"gamma\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[6,10]},"
        "\"beta\":{\"dtype\":\"F16\",\"shape\":[3],"
        "\"data_offsets\":[0,6]}}";
    uint8_t replacement[10];
    int descriptor;
    bool success;

    (void)memset(replacement, 0xee, sizeof(replacement));
    if (!m3_loader_test_path(fault->target_path, fixture->root,
                             "a-shard.safetensors") ||
        !m3_loader_test_path(fault->replacement_path, fixture->root,
                             "swap.safetensors") ||
        !m3_loader_test_write_safetensors(fault->replacement_path, a_header,
                                          sizeof(replacement))) {
        return false;
    }
    descriptor = open(fault->replacement_path, O_WRONLY);
    if (descriptor < 0) {
        return false;
    }
    success = pwrite(descriptor, replacement, sizeof(replacement),
                     (off_t)(8U + strlen(a_header))) ==
              (ssize_t)sizeof(replacement);
    if (close(descriptor) != 0) {
        success = false;
    }
    return success;
}

static bool m3_stage_fault_case(m3_stage_io_mode mode, size_t trigger_call,
                                m3_status expected_status, bool null_error)
{
    m3_weight_stage_test_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_stage_io_fault fault;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_weight_stage preserved;
    m3_backend *backend = NULL;
    m3_error error;
    m3_error load_error;
    m3_status load_status;
    bool success;

    (void)memset(&fault, 0, sizeof(fault));
    fault.mode = mode;
    fault.trigger_call = trigger_call;
    fault.flags_valid = true;
    fault.fail_cleanup_close = mode == M3_STAGE_IO_ERROR;
    m3_weight_stage_init(&stage);
    if (!m3_weight_stage_test_fixture_create(&fixture, &error) ||
        m3_backend_create_host(&backend, &error) != M3_STATUS_OK ||
        m3_weight_stage_load(&stage, &fixture.table, backend, NULL, NULL,
                             &error) != M3_STATUS_OK ||
        (mode == M3_STAGE_IO_PATH_SWAP &&
         !m3_stage_prepare_swap(&fixture, &fault))) {
        m3_weight_stage_dispose(&stage);
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return false;
    }
    preserved = stage;
    m3_stage_fault_io(&io, &fault);
    load_status = m3_weight_stage_load_with_io(
        &stage, &fixture.table, backend, NULL, NULL, &io,
        null_error ? NULL : &error);
    m3_error_reset(&load_error);
    if (!null_error) {
        load_error = error;
    }
    success = load_status == expected_status &&
              fault.triggered && fault.open_calls == 2U &&
              fault.flags_valid &&
              fault.maximum_request <= 4U &&
              fault.close_calls == fault.open_calls &&
              m3_backend_get_allocation_stats(backend, &stats, &error) ==
                  M3_STATUS_OK &&
              stats.live_allocated_bytes == 22U &&
              stats.live_storage_count == 2U &&
              stats.peak_allocated_bytes == 44U &&
              stats.peak_storage_count == 4U;
    if (expected_status == M3_STATUS_OK) {
        static const uint8_t expected_beta[] = {1U, 2U, 3U,
                                                4U, 5U, 6U};
        const m3_tensor_view *beta =
            m3_weight_stage_find_view(&stage, "beta");
        uint8_t actual_beta[sizeof(expected_beta)] = {0U};

        success = success && beta != NULL &&
                  m3_storage_read(beta->storage, beta->byte_offset,
                                  actual_beta, sizeof(actual_beta), &error) ==
                      M3_STATUS_OK &&
                  memcmp(actual_beta, expected_beta,
                         sizeof(expected_beta)) == 0 &&
                  (mode != M3_STAGE_IO_EINTR || fault.pread_calls == 7U) &&
                  (mode != M3_STAGE_IO_SHORT || fault.pread_calls == 7U);
    } else {
        success = success && memcmp(&stage, &preserved, sizeof(stage)) == 0;
        if (mode == M3_STAGE_IO_ERROR) {
            success = success &&
                      load_error.status == M3_STATUS_IO &&
                      strstr(m3_error_message(&load_error), "cannot read") !=
                          NULL;
        }
        if (mode == M3_STAGE_IO_CLOSE_ERROR) {
            success = success && load_error.status == M3_STATUS_IO &&
                      strstr(m3_error_message(&load_error),
                             "cannot close weight shard") != NULL &&
                      strstr(m3_error_message(&load_error), fixture.a_path) !=
                          NULL;
        }
        if (mode == M3_STAGE_IO_PATH_SWAP) {
            success = success && fault.observed_data &&
                      fault.first_byte == 1U;
        }
    }
    m3_weight_stage_dispose(&stage);
    m3_backend_free(backend);
    if (!m3_weight_stage_test_fixture_dispose(&fixture)) {
        success = false;
    }
    return success;
}

void m3_test_weight_stage_io_recovery(m3_test_context *test)
{
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_EINTR, 1U,
                                       M3_STATUS_OK, false),
                   "EINTR retries without duplicating a staged chunk");
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_SHORT, 1U,
                                       M3_STATUS_OK, false),
                   "positive short pread is accumulated exactly");
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_PATH_SWAP, 1U,
                                       M3_STATUS_INVALID_FORMAT, false),
                   "mid-read path swap cannot redirect retained descriptors");
}

void m3_test_weight_stage_io_failures(m3_test_context *test)
{
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_EOF, 3U, M3_STATUS_IO,
                                       true),
                   "premature EOF fails atomically without diagnostics");
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_ERROR, 3U, M3_STATUS_IO,
                                       false),
                   "mid-read I/O error closes every retained descriptor");
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_WRITE_ERROR, 2U,
                                       M3_STATUS_IO, false),
                   "storage write failure frees every temporary allocation");
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_MUTATE, 1U,
                                       M3_STATUS_INVALID_FORMAT, false),
                   "final descriptor snapshot detects mid-read mutation");
    M3_TEST_EXPECT(test,
                   m3_stage_fault_case(M3_STAGE_IO_CLOSE_ERROR, 1U,
                                       M3_STATUS_IO, false),
                   "close failure blocks publication and preserves old stage");
}

typedef struct {
    uint64_t cancel_at;
    size_t calls;
} m3_stage_cancel;

static bool m3_stage_cancel_progress(void *context_pointer,
                                     uint64_t completed, uint64_t total)
{
    m3_stage_cancel *context = context_pointer;

    (void)total;
    ++context->calls;
    return completed != context->cancel_at;
}

static bool m3_stage_cancel_case(uint64_t cancel_at, size_t expected_calls,
                                 bool opens_files)
{
    m3_weight_stage_test_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_stage_io_fault fault;
    m3_stage_cancel cancel;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_weight_stage preserved;
    m3_backend *backend = NULL;
    m3_error error;
    bool success;

    (void)memset(&fault, 0, sizeof(fault));
    fault.mode = M3_STAGE_IO_ERROR;
    fault.trigger_call = SIZE_MAX;
    fault.flags_valid = true;
    cancel.cancel_at = cancel_at;
    cancel.calls = 0U;
    m3_weight_stage_init(&stage);
    if (!m3_weight_stage_test_fixture_create(&fixture, &error) ||
        m3_backend_create_host(&backend, &error) != M3_STATUS_OK ||
        m3_weight_stage_load(&stage, &fixture.table, backend, NULL, NULL,
                             &error) != M3_STATUS_OK) {
        m3_weight_stage_dispose(&stage);
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return false;
    }
    preserved = stage;
    m3_stage_fault_io(&io, &fault);
    success = m3_weight_stage_load_with_io(
                  &stage, &fixture.table, backend, m3_stage_cancel_progress,
                  &cancel, &io, &error) == M3_STATUS_CANCELLED &&
              cancel.calls == expected_calls &&
              fault.open_calls == (opens_files ? 2U : 0U) &&
              fault.flags_valid &&
              fault.close_calls == fault.open_calls &&
              memcmp(&stage, &preserved, sizeof(stage)) == 0 &&
              m3_backend_get_allocation_stats(backend, &stats, &error) ==
                  M3_STATUS_OK &&
              stats.live_allocated_bytes == 22U &&
              stats.live_storage_count == 2U;
    m3_weight_stage_dispose(&stage);
    m3_backend_free(backend);
    if (!m3_weight_stage_test_fixture_dispose(&fixture)) {
        success = false;
    }
    return success;
}

void m3_test_weight_stage_cancellation(m3_test_context *test)
{
    M3_TEST_EXPECT(test, m3_stage_cancel_case(0U, 1U, false),
                   "initial cancellation opens no files or storage");
    M3_TEST_EXPECT(test, m3_stage_cancel_case(4U, 2U, true),
                   "middle cancellation releases all temporary resources");
    M3_TEST_EXPECT(test, m3_stage_cancel_case(22U, 7U, true),
                   "final cancellation still prevents publication");
}
