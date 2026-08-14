/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_file.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static bool m3_file_time_to_int64(time_t value, int64_t *converted)
{
    if ((time_t)-1 > (time_t)0 && (uintmax_t)value > (uintmax_t)INT64_MAX) {
        return false;
    }
    *converted = (int64_t)value;
    return (time_t)*converted == value;
}

m3_status m3_file_snapshot_descriptor(int descriptor,
                                       m3_file_snapshot *snapshot,
                                       m3_error *error)
{
    struct stat file_stat;
    int64_t modification_seconds;
    int64_t change_seconds;
    long modification_nanoseconds;
    long change_nanoseconds;

    if (snapshot == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "file snapshot argument is invalid");
    }
    (void)memset(snapshot, 0, sizeof(*snapshot));
    if (descriptor < 0) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "file snapshot argument is invalid");
    }
    if (fstat(descriptor, &file_stat) != 0) {
        return m3_error_set(error, M3_STATUS_IO,
                            "cannot snapshot file descriptor: %s",
                            strerror(errno));
    }
#if defined(__APPLE__)
#if defined(_POSIX_C_SOURCE) && !defined(_DARWIN_C_SOURCE)
    modification_nanoseconds = file_stat.st_mtimensec;
    change_nanoseconds = file_stat.st_ctimensec;
    if (!m3_file_time_to_int64(file_stat.st_mtime,
                               &modification_seconds) ||
        !m3_file_time_to_int64(file_stat.st_ctime, &change_seconds)) {
#else
    modification_nanoseconds = file_stat.st_mtimespec.tv_nsec;
    change_nanoseconds = file_stat.st_ctimespec.tv_nsec;
    if (!m3_file_time_to_int64(file_stat.st_mtimespec.tv_sec,
                               &modification_seconds) ||
        !m3_file_time_to_int64(file_stat.st_ctimespec.tv_sec,
                               &change_seconds)) {
#endif
#else
    modification_nanoseconds = file_stat.st_mtim.tv_nsec;
    change_nanoseconds = file_stat.st_ctim.tv_nsec;
    if (!m3_file_time_to_int64(file_stat.st_mtim.tv_sec,
                               &modification_seconds) ||
        !m3_file_time_to_int64(file_stat.st_ctim.tv_sec,
                               &change_seconds)) {
#endif
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "file timestamp does not fit snapshot");
    }
    if (file_stat.st_size < 0 ||
        (uintmax_t)file_stat.st_size > (uintmax_t)UINT64_MAX ||
        (uintmax_t)file_stat.st_dev > (uintmax_t)UINT64_MAX ||
        (uintmax_t)file_stat.st_ino > (uintmax_t)UINT64_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "file identity or size does not fit snapshot");
    }
    if (modification_nanoseconds < 0 ||
        modification_nanoseconds > 999999999L ||
        change_nanoseconds < 0 || change_nanoseconds > 999999999L) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "file timestamp nanoseconds are invalid");
    }
    snapshot->device = (uint64_t)file_stat.st_dev;
    snapshot->inode = (uint64_t)file_stat.st_ino;
    snapshot->file_size = (uint64_t)file_stat.st_size;
    snapshot->modification_time_seconds = modification_seconds;
    snapshot->modification_time_nanoseconds =
        (uint32_t)modification_nanoseconds;
    snapshot->change_time_seconds = change_seconds;
    snapshot->change_time_nanoseconds = (uint32_t)change_nanoseconds;
    snapshot->regular_file = S_ISREG(file_stat.st_mode);
    m3_error_reset(error);
    return M3_STATUS_OK;
}

bool m3_file_snapshot_equal(const m3_file_snapshot *left,
                            const m3_file_snapshot *right)
{
    return left != NULL && right != NULL && left->device == right->device &&
           left->inode == right->inode &&
           left->file_size == right->file_size &&
           left->modification_time_seconds ==
               right->modification_time_seconds &&
           left->modification_time_nanoseconds ==
               right->modification_time_nanoseconds &&
           left->change_time_seconds == right->change_time_seconds &&
           left->change_time_nanoseconds == right->change_time_nanoseconds &&
           left->regular_file == right->regular_file;
}

m3_status m3_file_read_bounded(const char *path, size_t maximum_size,
                               uint8_t **data, size_t *size,
                               m3_error *error)
{
    m3_file_snapshot before;
    m3_file_snapshot after;
    uint8_t *buffer;
    size_t total = 0U;
    size_t file_size;
    m3_status status;
    int descriptor;

    if (path == NULL || path[0] == '\0' || maximum_size == 0U ||
        data == NULL || size == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "bounded file read argument is invalid");
    }
    *data = NULL;
    *size = 0U;
    descriptor = open(path, O_RDONLY | O_NONBLOCK
#ifdef O_CLOEXEC
                                  | O_CLOEXEC
#endif
    );
    if (descriptor < 0) {
        return m3_error_set(error, M3_STATUS_IO, "cannot open '%s': %s",
                            path, strerror(errno));
    }
    status = m3_file_snapshot_descriptor(descriptor, &before, error);
    if (status != M3_STATUS_OK) {
        (void)close(descriptor);
        return status;
    }
    if (!before.regular_file || before.file_size == 0U ||
        before.file_size > (uint64_t)maximum_size ||
        before.file_size > (uint64_t)SIZE_MAX) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "file '%s' has invalid or excessive size",
                              path);
        (void)close(descriptor);
        return status;
    }
    file_size = (size_t)before.file_size;
    buffer = malloc(file_size);
    if (buffer == NULL) {
        status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                              "cannot allocate %zu bytes for '%s'",
                              file_size, path);
        (void)close(descriptor);
        return status;
    }
    while (total < file_size) {
        ssize_t count = read(descriptor, buffer + total, file_size - total);

        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            status = m3_error_set(error, M3_STATUS_IO,
                                  "cannot read '%s': %s", path,
                                  count < 0 ? strerror(errno) : "unexpected EOF");
            free(buffer);
            (void)close(descriptor);
            return status;
        }
        total += (size_t)count;
    }
    status = m3_file_snapshot_descriptor(descriptor, &after, error);
    if (status != M3_STATUS_OK) {
        free(buffer);
        (void)close(descriptor);
        return status;
    }
    if (!m3_file_snapshot_equal(&before, &after)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "file '%s' changed while being read", path);
        free(buffer);
        (void)close(descriptor);
        return status;
    }
    if (close(descriptor) != 0) {
        status = m3_error_set(error, M3_STATUS_IO, "cannot close '%s': %s",
                              path, strerror(errno));
        free(buffer);
        return status;
    }
    *data = buffer;
    *size = file_size;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
