/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_schema_internal.h"

static m3_status m3_vocoder_emit_block_root(
    m3_music3_schema_emitter *emitter, unsigned int block, uint64_t input,
    uint64_t output, uint64_t stride)
{
    char name[128];
    m3_status status = m3_music3_schema_format(
        name, sizeof(name), emitter->error, "blocks.%u.snake1.alpha", block);

    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(emitter, name, 1U, input, 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_format(
            name, sizeof(name), emitter->error, "blocks.%u.conv_t1.weight_g",
            block);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(emitter, name, input, 1U, 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_format(
            name, sizeof(name), emitter->error, "blocks.%u.conv_t1.weight_v",
            block);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(emitter, name, input, output,
                                        2U * stride);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_format(
            name, sizeof(name), emitter->error, "blocks.%u.conv_t1.bias",
            block);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(emitter, name, output);
    }
    return status;
}

static m3_status m3_vocoder_emit_residual(
    m3_music3_schema_emitter *emitter, unsigned int block,
    unsigned int residual, uint64_t channels)
{
    static const char *const suffixes[] = {
        "snake1.alpha", "conv1.weight_g", "conv1.weight_v", "conv1.bias",
        "snake2.alpha", "conv2.weight_g", "conv2.weight_v", "conv2.bias"
    };
    char name[128];
    size_t index;
    m3_status status = M3_STATUS_OK;

    for (index = 0U; index < sizeof(suffixes) / sizeof(suffixes[0]) &&
                     status == M3_STATUS_OK;
         ++index) {
        status = m3_music3_schema_format(
            name, sizeof(name), emitter->error, "blocks.%u.res_unit%u.%s",
            block, residual, suffixes[index]);
        if (status != M3_STATUS_OK) {
            break;
        }
        switch (index) {
        case 0U:
        case 4U:
            status = m3_music3_schema_emit3(emitter, name, 1U, channels, 1U);
            break;
        case 1U:
        case 5U:
            status = m3_music3_schema_emit3(emitter, name, channels, 1U, 1U);
            break;
        case 2U:
            status = m3_music3_schema_emit3(emitter, name, channels,
                                            channels, 7U);
            break;
        case 6U:
            status = m3_music3_schema_emit3(emitter, name, channels,
                                            channels, 1U);
            break;
        case 3U:
        case 7U:
            status = m3_music3_schema_emit1(emitter, name, channels);
            break;
        default:
            break;
        }
    }
    return status;
}

static m3_status m3_vocoder_emit_final(m3_music3_schema_emitter *emitter)
{
    m3_status status = m3_music3_schema_emit3(
        emitter, "snake_out.alpha", 1U, 96U, 1U);

    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(
            emitter, "conv_out.weight_g", 1U, 1U, 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(
            emitter, "conv_out.weight_v", 1U, 96U, 7U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(emitter, "conv_out.bias", 1U);
    }
    return status;
}

m3_status m3_music3_schema_generate_vocoder(m3_music3_schema_emitter *emitter)
{
    static const uint64_t strides[] = {8U, 8U, 4U, 2U};
    uint64_t input = 1536U;
    unsigned int block;
    m3_status status = m3_music3_schema_emit3(
        emitter, "dec_in_proj.weight", 1024U, 64U, 1U);

    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(emitter, "dec_in_proj.bias", 1024U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(emitter, "conv_in.weight_g", 1536U,
                                        1U, 1U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit3(emitter, "conv_in.weight_v", 1536U,
                                        1024U, 7U);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_emit1(emitter, "conv_in.bias", 1536U);
    }
    for (block = 0U; block < 4U && status == M3_STATUS_OK; ++block) {
        uint64_t output = input / 2U;
        unsigned int residual;

        status = m3_vocoder_emit_block_root(
            emitter, block, input, output, strides[block]);
        for (residual = 1U; residual <= 3U && status == M3_STATUS_OK;
             ++residual) {
            status = m3_vocoder_emit_residual(
                emitter, block, residual, output);
        }
        input = output;
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_emit_final(emitter);
    }
    return status;
}
