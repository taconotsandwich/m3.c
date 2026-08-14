/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_METAL_CONVOLUTION_INTERNAL_H
#define M3_BACKEND_METAL_CONVOLUTION_INTERNAL_H

#include "m3_error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>

NSString *m3_metal_convolution_kernel_source(void);
#endif

m3_status m3_metal_convolution_work(uint64_t count, size_t *work,
                                    m3_error *error);

#endif
