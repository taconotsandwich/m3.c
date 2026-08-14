/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_schema_internal.h"

typedef struct {
    const char *suffix;
    uint64_t first;
    uint64_t second;
} m3_flow_layer_tensor;

static const m3_flow_layer_tensor m3_flow_layer_tensors[] = {
    {"norm1.weight", 2048U, 0U},
    {"norm1.bias", 2048U, 0U},
    {"norm2.weight", 2048U, 0U},
    {"norm2.bias", 2048U, 0U},
    {"attn.to_q.weight", 2048U, 2048U},
    {"attn.to_k.weight", 2048U, 2048U},
    {"attn.to_v.weight", 2048U, 2048U},
    {"attn.to_out.0.weight", 2048U, 2048U},
    {"ff_in.weight", 16384U, 2048U},
    {"ff_in.bias", 16384U, 0U},
    {"ff_out.weight", 2048U, 8192U},
    {"ff_out.bias", 2048U, 0U}
};

static m3_status m3_flow_emit_roots(m3_music3_schema_emitter *emitter)
{
    m3_status status = m3_music3_schema_emit2(
        emitter, "time_proj.weight", 128U, 1U);

    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit2(
            emitter, "time_embed.linear_1.weight", 2048U, 256U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(
            emitter, "time_embed.linear_1.bias", 2048U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit2(
            emitter, "time_embed.linear_2.weight", 2048U, 2048U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(
            emitter, "time_embed.linear_2.bias", 2048U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(
            emitter, "preprocess_conv.weight", 2304U, 2304U, 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit2(emitter, "proj_in.weight", 2048U,
                                        2304U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit2(emitter, "proj_out.weight", 128U,
                                        2048U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(
            emitter, "postprocess_conv.weight", 128U, 128U, 1U);
    }
    return status;
}

m3_status m3_music3_schema_generate_flow(m3_music3_schema_emitter *emitter)
{
    char name[128];
    unsigned int layer;
    size_t tensor_index;
    m3_status status = m3_flow_emit_roots(emitter);

    for (layer = 0U; layer < 36U && status == M3_STATUS_OK; ++layer) {
        for (tensor_index = 0U;
             tensor_index < sizeof(m3_flow_layer_tensors) /
                                    sizeof(m3_flow_layer_tensors[0]) &&
             status == M3_STATUS_OK;
             ++tensor_index) {
            const m3_flow_layer_tensor *tensor =
                &m3_flow_layer_tensors[tensor_index];

            status = m3_music3_schema_format(
                name, sizeof(name), emitter->error,
                "transformer_blocks.%u.%s", layer, tensor->suffix);
            if (status == M3_STATUS_OK && tensor->second == 0U) {
                status = m3_music3_schema_emit1(emitter, name, tensor->first);
            } else if (status == M3_STATUS_OK) {
                status = m3_music3_schema_emit2(
                    emitter, name, tensor->first, tensor->second);
            }
        }
    }
    return status;
}
