/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
    const char *version = m3_version();

    if (strcmp(version, M3_VERSION_STRING) != 0) {
        (void)fprintf(stderr, "version mismatch: expected %s, got %s\n",
                      M3_VERSION_STRING, version);
        return 1;
    }

    (void)printf("m3 %s\n", version);
    return 0;
}
