/* SPDX-License-Identifier: GPL-2.0-only */

#include "metal_basic.inc"
#include "metal_dense.inc"
#include "metal_attention.inc"
#include "metal_sampling.inc"
#include "metal_convolution.inc"

M3_TEST_CASE(m3_test_metal_generation_foundations,
             "Metal Qwen and RVQ complete-frame chain")
