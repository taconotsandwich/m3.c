/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_safetensors.h"

#include "m3_safetensors_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void m3_safetensors_metadata_init(m3_safetensors_metadata *metadata)
{
    if (metadata != NULL) {
        (void)memset(metadata, 0, sizeof(*metadata));
    }
}

void m3_safetensors_metadata_dispose(m3_safetensors_metadata *metadata)
{
    size_t index;

    if (metadata == NULL) {
        return;
    }
    for (index = 0U; index < metadata->tensor_count; ++index) {
        free(metadata->tensors[index].name);
    }
    free(metadata->tensors);
    m3_safetensors_metadata_init(metadata);
}

static m3_status m3_safetensors_read_exact(int descriptor, uint8_t *data,
                                           size_t size, off_t offset,
                                           m3_error *error)
{
    size_t total = 0U;

    while (total < size) {
        ssize_t count = pread(descriptor, data + total, size - total,
                              offset + (off_t)total);

        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            return m3_error_set(error, M3_STATUS_IO,
                                "cannot read Safetensors file: %s",
                                strerror(errno));
        }
        if (count == 0) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "truncated Safetensors file");
        }
        total += (size_t)count;
    }
    return M3_STATUS_OK;
}

static uint64_t m3_safetensors_decode_u64_le(const uint8_t bytes[8])
{
    uint64_t value = 0U;
    unsigned int index;

    for (index = 0U; index < 8U; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

m3_status m3_safetensors_inspect_file(const char *path,
                                      m3_safetensors_metadata *metadata,
                                      m3_error *error)
{
    uint8_t length_bytes[8];
    uint8_t *header = NULL;
    m3_file_snapshot before;
    m3_file_snapshot after;
    m3_safetensors_metadata parsed;
    uint64_t header_length = 0U;
    uint64_t payload_size = 0U;
    m3_status status;
    int descriptor;

    if (path == NULL || path[0] == '\0' || metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Safetensors inspection argument is null");
    }
    m3_safetensors_metadata_init(&parsed);
    descriptor = open(path, O_RDONLY | O_NONBLOCK
#ifdef O_CLOEXEC
                                  | O_CLOEXEC
#endif
    );
    if (descriptor < 0) {
        return m3_error_set(error, M3_STATUS_IO,
                            "cannot open Safetensors file: %s",
                            strerror(errno));
    }
    status = m3_file_snapshot_descriptor(descriptor, &before, error);
    if (status == M3_STATUS_OK &&
        (!before.regular_file || before.file_size < 8U)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Safetensors path is not a regular file");
    }
    if (status == M3_STATUS_OK) {
        status = m3_safetensors_read_exact(descriptor, length_bytes,
                                           sizeof(length_bytes), 0, error);
    }
    if (status == M3_STATUS_OK) {
        header_length = m3_safetensors_decode_u64_le(length_bytes);
        if (header_length == 0U ||
            header_length > (uint64_t)M3_SAFETENSORS_MAX_HEADER_BYTES ||
            header_length > before.file_size - 8U ||
            header_length > (uint64_t)SIZE_MAX) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "invalid Safetensors header length");
        }
    }
    if (status == M3_STATUS_OK) {
        header = malloc((size_t)header_length);
        if (header == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate Safetensors header");
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_safetensors_read_exact(descriptor, header,
                                           (size_t)header_length, 8, error);
    }
    if (status == M3_STATUS_OK) {
        if (header == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "Safetensors header allocation is null");
        } else if (header[0] != (uint8_t)'{') {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "Safetensors header must begin with '{'");
        }
    }
    if (status == M3_STATUS_OK) {
        payload_size = before.file_size - 8U - header_length;
        parsed.data_section_offset = 8U + header_length;
        status = m3_safetensors_parse_header(
            header, (size_t)header_length, payload_size, &parsed, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_safetensors_validate_ranges(&parsed, payload_size, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_file_snapshot_descriptor(descriptor, &after, error);
    }
    if (status == M3_STATUS_OK && !m3_file_snapshot_equal(&before, &after)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Safetensors file changed during inspection");
    }
    if (close(descriptor) != 0 && status == M3_STATUS_OK) {
        status = m3_error_set(error, M3_STATUS_IO,
                              "cannot close Safetensors file: %s",
                              strerror(errno));
    }
    free(header);
    if (status != M3_STATUS_OK) {
        m3_safetensors_metadata_dispose(&parsed);
        return status;
    }
    parsed.source_snapshot = before;
    m3_safetensors_metadata_dispose(metadata);
    *metadata = parsed;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
