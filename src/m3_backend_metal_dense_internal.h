/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_METAL_DENSE_INTERNAL_H
#define M3_BACKEND_METAL_DENSE_INTERNAL_H

#include "m3_error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>

NSString *m3_metal_dense_kernel_source(void);
#endif

m3_status m3_metal_dense_work(uint64_t count, size_t *work,
                              m3_error *error);
m3_status m3_metal_dense_product(uint64_t left, uint64_t right,
                                 size_t *work, m3_error *error);

#endif
