/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_ERROR_H
#define M3_ERROR_H

#include "m3.h"

#include <stdarg.h>

m3_status m3_error_set(m3_error *error, m3_status status,
                       const char *format, ...);
m3_status m3_error_set_v(m3_error *error, m3_status status,
                         const char *format, va_list arguments);

#endif
