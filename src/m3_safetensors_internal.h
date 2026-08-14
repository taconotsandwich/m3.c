/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_SAFETENSORS_INTERNAL_H
#define M3_SAFETENSORS_INTERNAL_H

#include "m3_safetensors.h"

m3_status m3_safetensors_parse_header(
    const uint8_t *header, size_t header_size, uint64_t payload_size,
    m3_safetensors_metadata *metadata, m3_error *error);
m3_status m3_safetensors_validate_ranges(
    m3_safetensors_metadata *metadata, uint64_t payload_size,
    m3_error *error);

#endif
