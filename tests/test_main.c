/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3.h"
#include "m3_test.h"
#include "test_cases.h"

#include <stddef.h>

int main(void)
{
    static const m3_test_case cases[] = {
#define M3_TEST_CASE(function, description) {description, function},
#include "cases/core.inc"
#include "cases/model_loader.inc"
#include "cases/metal_runtime.inc"
#include "cases/prompt_tokenizer.inc"
#include "cases/host_utilities.inc"
#include "cases/wav_output.inc"
#include "cases/runtime_workspace.inc"
#include "cases/guided_sampling.inc"
#include "cases/transformer_precision.inc"
#include "cases/metal_operations.inc"
#include "cases/qwen_runtime.inc"
#include "cases/rvq_condition.inc"
#include "cases/flow_runtime.inc"
#include "cases/vocoder_runtime.inc"
#include "cases/music3_engine.inc"
#undef M3_TEST_CASE
    };
    static const size_t case_count = sizeof(cases) / sizeof(cases[0]);

    return m3_test_run("m3 " M3_VERSION_STRING, cases, case_count);
}
