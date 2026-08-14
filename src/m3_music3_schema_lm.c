/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_schema_internal.h"

typedef struct {
    const char *suffix;
    uint64_t rows;
    uint64_t columns;
    uint8_t rank;
} m3_lm_layer_tensor;

static const m3_lm_layer_tensor m3_lm_layer_tensors[] = {
    {"input_layernorm.weight", 4096U, 0U, 1U},
    {"self_attn.q_proj.weight", 4096U, 4096U, 2U},
    {"self_attn.k_proj.weight", 1024U, 4096U, 2U},
    {"self_attn.v_proj.weight", 1024U, 4096U, 2U},
    {"self_attn.o_proj.weight", 4096U, 4096U, 2U},
    {"self_attn.q_norm.weight", 128U, 0U, 1U},
    {"self_attn.k_norm.weight", 128U, 0U, 1U},
    {"post_attention_layernorm.weight", 4096U, 0U, 1U},
    {"mlp.gate_proj.weight", 12288U, 4096U, 2U},
    {"mlp.up_proj.weight", 12288U, 4096U, 2U},
    {"mlp.down_proj.weight", 4096U, 12288U, 2U}
};

static m3_status m3_lm_emit(m3_music3_schema_emitter *emitter,
                            const char *name, uint64_t rows,
                            uint64_t columns, uint8_t rank)
{
    const uint64_t shape[] = {rows, columns};

    return m3_music3_schema_emit(emitter, name, M3_DTYPE_BF16, rank, shape);
}

m3_status m3_music3_schema_generate_lm(m3_music3_schema_emitter *emitter)
{
    char name[128];
    unsigned int layer;
    size_t tensor_index;
    m3_status status;

    status = m3_lm_emit(emitter, "model.embed_tokens.weight", 200000U,
                        4096U, 2U);
    if (status == M3_STATUS_OK) {
        status = m3_lm_emit(emitter, "model.norm.weight", 4096U, 0U, 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_lm_emit(emitter, "lm_head.weight", 200000U, 4096U, 2U);
    }
    for (layer = 0U; layer < 36U && status == M3_STATUS_OK; ++layer) {
        for (tensor_index = 0U;
             tensor_index < sizeof(m3_lm_layer_tensors) /
                                    sizeof(m3_lm_layer_tensors[0]) &&
             status == M3_STATUS_OK;
             ++tensor_index) {
            const m3_lm_layer_tensor *tensor =
                &m3_lm_layer_tensors[tensor_index];

            status = m3_music3_schema_format(
                name, sizeof(name), emitter->error, "model.layers.%u.%s",
                layer, tensor->suffix);
            if (status == M3_STATUS_OK) {
                status = m3_lm_emit(emitter, name, tensor->rows,
                                    tensor->columns, tensor->rank);
            }
        }
    }
    return status;
}
