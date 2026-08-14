/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"
#include "m3_qwen_internal.h"
#include "m3_test.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void m3_test_qwen_official_plan(m3_test_context *test)
{
    static const char *const suffixes[] = {
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight"
    };
    const m3_qwen_dimensions *dimensions = m3_qwen_official_dimensions();
    m3_qwen_weight_plan plan;
    m3_error error;
    size_t layer;
    size_t suffix;
    bool ordered = true;

    M3_TEST_EXPECT(test,
                   dimensions->vocab_size == 200000U &&
                       dimensions->hidden_size == 4096U &&
                       dimensions->layer_count == 36U &&
                       dimensions->query_heads == 32U &&
                       dimensions->key_value_heads == 8U &&
                       dimensions->head_dimension == 128U &&
                       dimensions->intermediate_size == 12288U,
                   "lock official Qwen architecture dimensions");
    M3_TEST_EXPECT(test,
                   dimensions->eos_token_id == 151670U &&
                       dimensions->semantic_token_start == 151675U &&
                       dimensions->semantic_token_count == 16384U &&
                       dimensions->rms_epsilon == 1.0e-6F &&
                       dimensions->rope_theta == 1000000.0F,
                   "lock official Qwen token and numeric constants");
    M3_TEST_EXPECT(test,
                   m3_qwen_weight_plan_init(&plan, &error) ==
                           M3_STATUS_OK &&
                       plan.count == 399U,
                   "build all 399 official Qwen requirements");
    M3_TEST_EXPECT(test,
                   strcmp(plan.requirements[0].name,
                          "model.embed_tokens.weight") == 0 &&
                       strcmp(plan.requirements[1].name,
                              "model.norm.weight") == 0 &&
                       strcmp(plan.requirements[2].name,
                              "lm_head.weight") == 0,
                   "bind global Qwen weights in official order");
    for (layer = 0U; layer < 36U; ++layer) {
        for (suffix = 0U; suffix < 11U; ++suffix) {
            char expected[M3_QWEN_WEIGHT_NAME_CAPACITY];
            size_t index = 3U + layer * 11U + suffix;
            int written = snprintf(expected, sizeof(expected),
                                   "model.layers.%zu.%s", layer,
                                   suffixes[suffix]);

            if (written < 0 || (size_t)written >= sizeof(expected) ||
                strcmp(plan.requirements[index].name, expected) != 0 ||
                plan.requirements[index].tensor.dtype != M3_DTYPE_BF16) {
                ordered = false;
            }
        }
    }
    M3_TEST_EXPECT(test, ordered,
                   "bind every layer name and BF16 requirement exactly");
    M3_TEST_EXPECT(test,
                   plan.requirements[0].tensor.rank == 2U &&
                       plan.requirements[0].tensor.shape[0] == 200000U &&
                       plan.requirements[0].tensor.shape[1] == 4096U &&
                       plan.requirements[398].tensor.shape[0] == 4096U &&
                       plan.requirements[398].tensor.shape[1] == 12288U,
                   "lock first and last official weight shapes");
}

void m3_test_qwen_cache_layout(m3_test_context *test)
{
    const m3_qwen_dimensions *dimensions = m3_qwen_official_dimensions();
    m3_runtime_tensor_spec first;
    m3_runtime_tensor_spec last;
    size_t count = 17U;
    size_t bytes = 19U;
    m3_error error;
    m3_status status = m3_qwen_cache_measure(
        dimensions, 14000U, &count, &bytes, &error);

    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OK && count == 72U &&
                       bytes == (size_t)4128768000ULL,
                   "measure exact 72-storage 14000-token cache bytes");
    M3_TEST_EXPECT(test,
                   m3_qwen_cache_spec(dimensions, 14000U, 0U, &first,
                                      &error) == M3_STATUS_OK &&
                       m3_qwen_cache_spec(dimensions, 14000U, 71U, &last,
                                          &error) == M3_STATUS_OK &&
                       first.dtype == M3_DTYPE_BF16 && first.rank == 4U &&
                       first.shape[0] == 14000U && first.shape[1] == 2U &&
                       first.shape[2] == 8U && first.shape[3] == 128U &&
                       memcmp(&first, &last, sizeof(first)) == 0,
                   "lock token-major [capacity,2,8,128] K/V layout");
    count = 17U;
    bytes = 19U;
    status = m3_qwen_cache_measure(dimensions, UINT64_MAX, &count, &bytes,
                                   NULL);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OVERFLOW && count == 17U &&
                       bytes == 19U,
                   "reject cache overflow with null error atomically");
    M3_TEST_EXPECT(test,
                   m3_qwen_cache_spec(dimensions, 14000U, 72U, &first,
                                      NULL) == M3_STATUS_OUT_OF_RANGE &&
                       m3_qwen_cache_measure(dimensions, 0U, &count, &bytes,
                                             NULL) ==
                           M3_STATUS_INVALID_ARGUMENT,
                   "reject cache index and zero-capacity bounds");
}

void m3_test_qwen_runtime_api_validation(m3_test_context *test)
{
    m3_backend *backend = NULL;
    m3_weight_stage stage;
    m3_qwen_runtime *runtime = (m3_qwen_runtime *)(uintptr_t)1U;
    m3_qwen_logits logits;
    m3_qwen_step_result step;
    m3_error error;

    m3_weight_stage_init(&stage);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK,
                   "create host backend for Qwen API validation");
    stage.backend = backend;
    M3_TEST_EXPECT(test,
                   m3_qwen_runtime_create(&runtime, NULL, 1U, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       runtime == NULL,
                   "reject absent staged Qwen model with null error");
    runtime = (m3_qwen_runtime *)(uintptr_t)1U;
    M3_TEST_EXPECT(test,
                   m3_qwen_runtime_create(&runtime, &stage, 14001U, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       runtime == NULL,
                   "reject runtime cache capacity above 14000 atomically");
    runtime = (m3_qwen_runtime *)(uintptr_t)1U;
    M3_TEST_EXPECT(test,
                   m3_qwen_runtime_create(&runtime, &stage, 1U, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       runtime == NULL,
                   "reject a stage without exact official weights before "
                   "cache allocation");
    M3_TEST_EXPECT(test,
                   m3_qwen_runtime_prefill(NULL, NULL, NULL, NULL, &logits,
                                           NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       m3_qwen_runtime_step(NULL,
                                            M3_QWEN_SEMANTIC_TOKEN_START,
                                            NULL, NULL, &step, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       m3_qwen_runtime_token_count(NULL) == 0U &&
                       m3_qwen_runtime_cache_capacity(NULL) == 0U &&
                       m3_qwen_runtime_cache_bytes(NULL) == 0U,
                   "make null runtime calls and queries safe");
    m3_qwen_runtime_free(NULL);
    m3_backend_free(backend);
}

static bool m3_qwen_rope_oracle_execute(
    m3_backend *backend, m3_runtime_workspace *rope,
    m3_runtime_workspace *io, uint64_t position, m3_error *error)
{
    static const size_t indices[] = {0U, 1U, 63U, 64U, 65U, 127U};
    static const float values[] = {1.0F, 3.0F, -1.0F,
                                   2.0F, 4.0F, 0.5F};
    m3_command_executor executor;
    m3_command command;
    size_t index;
    m3_status status;

    (void)memset(m3_storage_data(io->storages[0]), 0,
                 m3_storage_size(io->storages[0]));
    for (index = 0U; index < sizeof(indices) / sizeof(indices[0]); ++index) {
        m3_op_store_float(&io->views[0],
                          m3_op_element_offset(&io->views[0], indices[index]),
                          values[index]);
    }
    m3_command_executor_init(&executor, backend);
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ROPE;
    command.descriptor.rope.input = &io->views[0];
    command.descriptor.rope.cosines = &rope->views[0];
    command.descriptor.rope.sines = &rope->views[1];
    command.descriptor.rope.output = &io->views[1];
    command.descriptor.rope.position_offset = position;
    command.descriptor.rope.rotary_dimension = 128U;
    command.descriptor.rope.mode = M3_ROPE_HALF_SPLIT;
    status = m3_command_executor_execute(&executor, &command, 1U, error);
    m3_command_executor_dispose(&executor);
    return status == M3_STATUS_OK;
}

void m3_test_qwen_rope_oracles(m3_test_context *test)
{
    static const size_t pairs[] = {0U, 1U, 2U, 31U, 32U, 63U};
    static const uint16_t position_one[][2] = {
        {0x3f0aU, 0x3f57U}, {0x3f31U, 0x3f39U},
        {0x3f4cU, 0x3f1bU}, {0x3f80U, 0x3aa3U},
        {0x3f80U, 0x3a83U}, {0x3f80U, 0x35a7U}
    };
    static const uint16_t position_last[][2] = {
        {0x3f7fU, 0x3d81U}, {0xbf64U, 0x3eeaU},
        {0x3ef3U, 0xbf61U}, {0x3dbeU, 0xbf7fU},
        {0x3e0dU, 0x3f7eU}, {0x3f80U, 0x3c8eU}
    };
    static const uint16_t rotated_one[] = {
        0xbf92U, 0xbf51U, 0xbf80U, 0x3ff6U, 0x409eU, 0x3f00U
    };
    static const uint16_t rotated_last[] = {
        0x3f5fU, 0xc090U, 0xbf81U, 0x4004U, 0xc00cU, 0x3ef7U
    };
    static const size_t rotated_indices[] = {0U, 1U, 63U,
                                             64U, 65U, 127U};
    m3_backend *backend = NULL;
    m3_runtime_workspace rope;
    m3_runtime_workspace io;
    m3_runtime_tensor_spec specs[2];
    const uint16_t *cosines;
    const uint16_t *sines;
    const uint16_t *output;
    m3_error error;
    size_t index;
    bool exact_zero = true;
    bool exact_one = true;
    bool exact_last = true;
    bool split_one = true;
    bool split_last = true;

    m3_runtime_workspace_init(&rope);
    m3_runtime_workspace_init(&io);
    if (m3_backend_create_host(&backend, &error) != M3_STATUS_OK ||
        m3_qwen_rope_build(&rope, backend,
                           m3_qwen_official_dimensions(), 14000U,
                           &error) != M3_STATUS_OK) {
        M3_TEST_EXPECT(test, false, "build official Qwen RoPE table");
        m3_runtime_workspace_dispose(&rope);
        m3_backend_free(backend);
        return;
    }
    cosines = m3_storage_const_data(rope.views[0].storage);
    sines = m3_storage_const_data(rope.views[1].storage);
    for (index = 0U; index < 64U; ++index) {
        exact_zero = exact_zero && cosines[index] == 0x3f80U &&
                     sines[index] == 0x0000U;
    }
    for (index = 0U; index < sizeof(pairs) / sizeof(pairs[0]); ++index) {
        size_t one = 64U + pairs[index];
        size_t last = 13999U * 64U + pairs[index];

        exact_one = exact_one &&
                    cosines[one] == position_one[index][0] &&
                    sines[one] == position_one[index][1];
        exact_last = exact_last &&
                     cosines[last] == position_last[index][0] &&
                     sines[last] == position_last[index][1];
    }
    M3_TEST_EXPECT(test, exact_zero && exact_one && exact_last,
                   "lock BF16 RoPE oracles at positions 0, 1, and 13999");
    (void)memset(specs, 0, sizeof(specs));
    specs[0].dtype = M3_DTYPE_BF16;
    specs[0].rank = 4U;
    specs[0].shape[0] = 1U;
    specs[0].shape[1] = 1U;
    specs[0].shape[2] = 1U;
    specs[0].shape[3] = 128U;
    specs[0].alignment = 64U;
    specs[1] = specs[0];
    if (m3_runtime_workspace_build(&io, backend, specs, 2U, &error) !=
        M3_STATUS_OK) {
        M3_TEST_EXPECT(test, false, "build Qwen half-split oracle tensors");
    } else {
        split_one = m3_qwen_rope_oracle_execute(
            backend, &rope, &io, 1U, &error);
        output = m3_storage_const_data(io.views[1].storage);
        for (index = 0U;
             index < sizeof(rotated_indices) / sizeof(rotated_indices[0]);
             ++index) {
            split_one = split_one &&
                        output[rotated_indices[index]] == rotated_one[index];
        }
        split_last = m3_qwen_rope_oracle_execute(
            backend, &rope, &io, 13999U, &error);
        output = m3_storage_const_data(io.views[1].storage);
        for (index = 0U;
             index < sizeof(rotated_indices) / sizeof(rotated_indices[0]);
             ++index) {
            split_last = split_last &&
                         output[rotated_indices[index]] ==
                             rotated_last[index];
        }
        M3_TEST_EXPECT(test, split_one && split_last,
                       "apply official tables in exact half-split order");
    }
    m3_runtime_workspace_dispose(&io);
    m3_runtime_workspace_dispose(&rope);
    m3_backend_free(backend);
}
