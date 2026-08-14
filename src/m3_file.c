/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

m3_status m3_file_read_bounded(const char *path, size_t maximum_size,
                               uint8_t **data, size_t *size,
                               m3_error *error)
{
    struct stat file_stat;
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
    if (fstat(descriptor, &file_stat) != 0) {
        status = m3_error_set(error, M3_STATUS_IO, "cannot stat '%s': %s",
                              path, strerror(errno));
        (void)close(descriptor);
        return status;
    }
    if (!S_ISREG(file_stat.st_mode) || file_stat.st_size <= 0 ||
        (uint64_t)file_stat.st_size > (uint64_t)maximum_size ||
        (uint64_t)file_stat.st_size > (uint64_t)SIZE_MAX) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "file '%s' has invalid or excessive size",
                              path);
        (void)close(descriptor);
        return status;
    }
    file_size = (size_t)file_stat.st_size;
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
