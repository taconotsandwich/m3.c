/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TEST_CASES_H
#define M3_TEST_CASES_H

#include "m3_test.h"

#define M3_TEST_CASE(function, description) \
    void function(m3_test_context *test);

#include "cases/core.inc"
#include "cases/model_loader.inc"
#include "cases/metal_runtime.inc"
#include "cases/prompt_tokenizer.inc"
#include "cases/host_utilities.inc"
#include "cases/transformer_precision.inc"
#include "cases/metal_operations.inc"
#include "cases/qwen_runtime.inc"
#include "cases/rvq_condition.inc"
#include "cases/flow_runtime.inc"
#include "cases/vocoder_runtime.inc"

#undef M3_TEST_CASE

#endif
