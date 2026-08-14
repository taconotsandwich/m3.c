/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_UNICODE_H
#define M3_UNICODE_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool m3_utf8_decode(const uint8_t *data, size_t size, uint32_t *code_point,
                    size_t *length);
size_t m3_utf8_encode(uint32_t code_point, uint8_t output[4]);
bool m3_unicode_is_letter(uint32_t code_point);
bool m3_unicode_is_number(uint32_t code_point);
bool m3_unicode_is_whitespace(uint32_t code_point);
uint32_t m3_unicode_simple_fold(uint32_t code_point);
m3_status m3_unicode_nfc(const uint8_t *data, size_t size, uint8_t **output,
                         size_t *output_size, m3_error *error);

#endif
