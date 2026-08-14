/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_schema_internal.h"

typedef struct {
    const char *suffix;
    uint64_t first;
    uint64_t second;
} m3_rvq_layer_tensor;

static const m3_rvq_layer_tensor m3_rvq_layer_tensors[] = {
    {"input_layernorm.weight", 4096U, 0U},
    {"attn.to_q.weight", 4096U, 4096U},
    {"attn.to_k.weight", 4096U, 4096U},
    {"attn.to_v.weight", 4096U, 4096U},
    {"attn.to_out.weight", 4096U, 4096U},
    {"post_attention_layernorm.weight", 4096U, 0U},
    {"gate_proj.weight", 6144U, 4096U},
    {"up_proj.weight", 6144U, 4096U},
    {"down_proj.weight", 4096U, 6144U}
};

static m3_status m3_rvq_emit(m3_music3_schema_emitter *emitter,
                             const char *name, uint64_t first,
                             uint64_t second)
{
    const uint64_t shape[] = {first, second};
    uint8_t rank = second == 0U ? 1U : 2U;

    return m3_music3_schema_emit(emitter, name, M3_DTYPE_BF16, rank, shape);
}

m3_status m3_music3_schema_generate_rvq(m3_music3_schema_emitter *emitter)
{
    char name[128];
    unsigned int layer;
    unsigned int head;
    size_t tensor_index;
    m3_status status = m3_rvq_emit(
        emitter, "audio_embeddings.weight", 7168U, 4096U);

    if (status == M3_STATUS_OK) {
        status = m3_rvq_emit(emitter, "projection.weight", 4096U, 4096U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_emit(emitter, "pos_embedding.weight", 16U, 4096U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_emit(emitter, "norm.weight", 4096U, 0U);
    }
    for (layer = 0U; layer < 4U && status == M3_STATUS_OK; ++layer) {
        for (tensor_index = 0U;
             tensor_index < sizeof(m3_rvq_layer_tensors) /
                                    sizeof(m3_rvq_layer_tensors[0]) &&
             status == M3_STATUS_OK;
             ++tensor_index) {
            const m3_rvq_layer_tensor *tensor =
                &m3_rvq_layer_tensors[tensor_index];

            status = m3_music3_schema_format(
                name, sizeof(name), emitter->error, "layers.%u.%s", layer,
                tensor->suffix);
            if (status == M3_STATUS_OK) {
                status = m3_rvq_emit(emitter, name, tensor->first,
                                     tensor->second);
            }
        }
    }
    for (head = 0U; head < 7U && status == M3_STATUS_OK; ++head) {
        status = m3_music3_schema_format(
            name, sizeof(name), emitter->error, "audio_heads.%u.weight",
            head);
        if (status == M3_STATUS_OK) {
            status = m3_rvq_emit(emitter, name, 1024U, 4096U);
        }
    }
    return status;
}

m3_status m3_music3_schema_generate_condition(
    m3_music3_schema_emitter *emitter)
{
    m3_status status = m3_music3_schema_emit1(
        emitter, "layer_weight_logits", 8U);

    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(emitter, "layer_scale", 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(emitter, "proj.weight", 2048U,
                                        4096U, 3U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(emitter, "proj.bias", 2048U);
    }
    return status;
}
