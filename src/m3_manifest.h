/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MANIFEST_H
#define M3_MANIFEST_H

#include "m3_error.h"

m3_status m3_manifest_validate_file(const char *path, m3_error *error);

#endif
