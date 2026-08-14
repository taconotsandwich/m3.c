/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include <stdint.h>

m3_status m3_metal_dispatch_1d(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline, size_t work_count,
    m3_error *error)
{
    NSUInteger threads;

    if (work_count == 0U) {
        return M3_STATUS_OK;
    }
    if (work_count > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Metal dispatch grid exceeds 32-bit indexing");
    }
    if (encoder == nil || pipeline == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal dispatch resources are unavailable");
    }
    threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256U) {
        threads = 256U;
    }
    if (threads == 0U) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal pipeline has no dispatch capacity");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)work_count, 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    return M3_STATUS_OK;
}
