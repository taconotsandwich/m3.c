/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

static bool m3_byte_alphabet_visible(uint8_t byte)
{
    return (byte >= 33U && byte <= 126U) ||
           (byte >= 161U && byte <= 172U) ||
           (byte >= 174U);
}

uint32_t m3_byte_alphabet_code_point(uint8_t byte)
{
    uint32_t extra = 0U;
    unsigned int candidate;

    if (m3_byte_alphabet_visible(byte)) {
        return byte;
    }
    for (candidate = 0U; candidate < (unsigned int)byte; ++candidate) {
        if (!m3_byte_alphabet_visible((uint8_t)candidate)) {
            extra += 1U;
        }
    }
    return 256U + extra;
}

bool m3_byte_alphabet_byte(uint32_t code_point, uint8_t *byte)
{
    unsigned int candidate;

    if (byte == NULL) {
        return false;
    }
    if (code_point <= 255U &&
        m3_byte_alphabet_visible((uint8_t)code_point)) {
        *byte = (uint8_t)code_point;
        return true;
    }
    for (candidate = 0U; candidate <= 255U; ++candidate) {
        uint8_t value = (uint8_t)candidate;

        if (!m3_byte_alphabet_visible(value) &&
            m3_byte_alphabet_code_point(value) == code_point) {
            *byte = value;
            return true;
        }
    }
    return false;
}
