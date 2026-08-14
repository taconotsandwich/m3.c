/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_error.h"

#include <stdio.h>

const char *m3_status_string(m3_status status)
{
    switch (status) {
    case M3_STATUS_OK:
        return "ok";
    case M3_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case M3_STATUS_OUT_OF_RANGE:
        return "out of range";
    case M3_STATUS_OVERFLOW:
        return "overflow";
    case M3_STATUS_OUT_OF_MEMORY:
        return "out of memory";
    case M3_STATUS_IO:
        return "I/O error";
    case M3_STATUS_INVALID_FORMAT:
        return "invalid format";
    case M3_STATUS_UNSUPPORTED:
        return "unsupported";
    case M3_STATUS_CANCELLED:
        return "cancelled";
    case M3_STATUS_INTERNAL:
        return "internal error";
    }

    return "unknown status";
}

void m3_error_reset(m3_error *error)
{
    if (error == NULL) {
        return;
    }

    error->status = M3_STATUS_OK;
    error->message[0] = '\0';
}

m3_status m3_error_set_v(m3_error *error, m3_status status,
                         const char *format, va_list arguments)
{
    int result;

    if (error == NULL) {
        return status;
    }

    error->status = status;
    if (status == M3_STATUS_OK) {
        error->message[0] = '\0';
        return status;
    }

    if (format == NULL) {
        format = m3_status_string(status);
        (void)snprintf(error->message, sizeof(error->message), "%s", format);
        return status;
    }

    result = vsnprintf(error->message, sizeof(error->message), format,
                       arguments);
    if (result < 0) {
        (void)snprintf(error->message, sizeof(error->message), "%s",
                       m3_status_string(status));
    }
    error->message[sizeof(error->message) - 1U] = '\0';
    return status;
}

m3_status m3_error_set(m3_error *error, m3_status status,
                       const char *format, ...)
{
    m3_status result;
    va_list arguments;

    va_start(arguments, format);
    result = m3_error_set_v(error, status, format, arguments);
    va_end(arguments);
    return result;
}

const char *m3_error_message(const m3_error *error)
{
    if (error == NULL) {
        return "";
    }
    return error->message;
}
