/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_vocoder_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static m3_status m3_vocoder_name(char *destination, const char *format,
                                 m3_error *error, ...)
{
    va_list arguments;
    int length;

    va_start(arguments, error);
    length = vsnprintf(destination, M3_VOCODER_NAME_CAPACITY, format,
                       arguments);
    va_end(arguments);
    if (length < 0 || (size_t)length >= M3_VOCODER_NAME_CAPACITY) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder weight name is too long");
    }
    return M3_STATUS_OK;
}

void m3_vocoder_plan_init(m3_vocoder_plan *plan)
{
    if (plan != NULL) {
        (void)memset(plan, 0, sizeof(*plan));
    }
}

void m3_vocoder_plan_dispose(m3_vocoder_plan *plan)
{
    if (plan != NULL) {
        free(plan->entries);
        m3_vocoder_plan_init(plan);
    }
}

void m3_vocoder_plan_official_config(m3_vocoder_plan_config *config)
{
    if (config == NULL) {
        return;
    }
    (void)memset(config, 0, sizeof(*config));
    config->latent_channels = 128U;
    config->maximum_latent_length = M3_VOCODER_MAXIMUM_LATENT_LENGTH;
    config->decoder_input_channels = 64U;
    config->decoder_output_channels = 1024U;
    config->initial_channels = 1536U;
    config->block_count = M3_VOCODER_BLOCK_COUNT;
    config->residual_count = M3_VOCODER_RESIDUAL_COUNT;
    config->strides[0] = 8U;
    config->strides[1] = 8U;
    config->strides[2] = 4U;
    config->strides[3] = 2U;
}

static m3_status m3_vocoder_metadata(m3_tensor_metadata *metadata,
                                     uint8_t rank, const uint64_t *shape,
                                     m3_error *error)
{
    return m3_tensor_metadata_init(metadata, M3_DTYPE_F32, rank, shape,
                                   error);
}

static m3_status m3_vocoder_add_copy(
    m3_vocoder_plan *plan, size_t *entry_index, const char *name,
    uint8_t rank, const uint64_t *shape, m3_error *error)
{
    m3_vocoder_plan_entry *entry = &plan->entries[*entry_index];
    m3_status status = m3_vocoder_name(
        entry->output_name, "%s", error, name);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_name(
            entry->source_names[0], "%s", error, name);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_metadata(
            &entry->source_metadata[0], rank, shape, error);
    }
    if (status == M3_STATUS_OK) {
        entry->output_metadata = entry->source_metadata[0];
        entry->kind = M3_VOCODER_MATERIAL_COPY;
        ++*entry_index;
    }
    return status;
}

static m3_status m3_vocoder_add_norm(
    m3_vocoder_plan *plan, size_t *entry_index, const char *root,
    const uint64_t *value_shape, m3_error *error)
{
    m3_vocoder_plan_entry *entry = &plan->entries[*entry_index];
    uint64_t gain_shape[] = {value_shape[0], 1U, 1U};
    m3_status status = m3_vocoder_name(
        entry->output_name, "%s.weight", error, root);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_name(
            entry->source_names[0], "%s.weight_g", error, root);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_name(
            entry->source_names[1], "%s.weight_v", error, root);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_metadata(
            &entry->source_metadata[0], 3U, gain_shape, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_metadata(
            &entry->source_metadata[1], 3U, value_shape, error);
    }
    if (status == M3_STATUS_OK) {
        entry->output_metadata = entry->source_metadata[1];
        entry->kind = M3_VOCODER_MATERIAL_WEIGHT_NORM;
        ++*entry_index;
    }
    return status;
}

static m3_status m3_vocoder_add_named_copy(
    m3_vocoder_plan *plan, size_t *entry_index, m3_error *error,
    const char *format, unsigned int block, unsigned int residual,
    const char *suffix, uint8_t rank, const uint64_t *shape)
{
    char name[M3_VOCODER_NAME_CAPACITY];
    m3_status status = m3_vocoder_name(
        name, format, error, block, residual, suffix);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            plan, entry_index, name, rank, shape, error);
    }
    return status;
}

static m3_status m3_vocoder_add_named_norm(
    m3_vocoder_plan *plan, size_t *entry_index, m3_error *error,
    const char *format, unsigned int block, unsigned int residual,
    const char *suffix, const uint64_t *shape)
{
    char root[M3_VOCODER_NAME_CAPACITY];
    m3_status status = m3_vocoder_name(
        root, format, error, block, residual, suffix);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_norm(
            plan, entry_index, root, shape, error);
    }
    return status;
}

static m3_status m3_vocoder_add_residual(
    m3_vocoder_plan *plan, size_t *entry_index, unsigned int block,
    unsigned int residual, uint64_t channels, m3_error *error)
{
    uint64_t alpha_shape[] = {1U, channels, 1U};
    uint64_t conv1_shape[] = {channels, channels, 7U};
    uint64_t conv2_shape[] = {channels, channels, 1U};
    uint64_t bias_shape[] = {channels};
    const char *format = "blocks.%u.res_unit%u.%s";
    m3_status status = m3_vocoder_add_named_copy(
        plan, entry_index, error, format, block, residual,
        "snake1.alpha", 3U, alpha_shape);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_named_norm(
            plan, entry_index, error, format, block, residual,
            "conv1", conv1_shape);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_named_copy(
            plan, entry_index, error, format, block, residual,
            "conv1.bias", 1U, bias_shape);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_named_copy(
            plan, entry_index, error, format, block, residual,
            "snake2.alpha", 3U, alpha_shape);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_named_norm(
            plan, entry_index, error, format, block, residual,
            "conv2", conv2_shape);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_named_copy(
            plan, entry_index, error, format, block, residual,
            "conv2.bias", 1U, bias_shape);
    }
    return status;
}

static m3_status m3_vocoder_add_block(
    m3_vocoder_plan *plan, size_t *entry_index,
    const m3_vocoder_plan_config *config, unsigned int block,
    uint64_t input, uint64_t output, m3_error *error)
{
    uint64_t alpha_shape[] = {1U, input, 1U};
    uint64_t value_shape[] = {
        input, output, 2U * config->strides[block]
    };
    uint64_t bias_shape[] = {output};
    char name[M3_VOCODER_NAME_CAPACITY];
    size_t residual;
    m3_status status = m3_vocoder_name(
        name, "blocks.%u.snake1.alpha", error, block);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            plan, entry_index, name, 3U, alpha_shape, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_name(
            name, "blocks.%u.conv_t1", error, block);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_norm(
            plan, entry_index, name, value_shape, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_name(
            name, "blocks.%u.conv_t1.bias", error, block);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            plan, entry_index, name, 1U, bias_shape, error);
    }
    for (residual = 1U; residual <= config->residual_count &&
                         status == M3_STATUS_OK;
         ++residual) {
        status = m3_vocoder_add_residual(
            plan, entry_index, block, (unsigned int)residual, output,
            error);
    }
    return status;
}

m3_status m3_vocoder_plan_config_validate(
    const m3_vocoder_plan_config *config, m3_error *error)
{
    uint64_t channels;
    size_t block;

    if (config == NULL || config->latent_channels == 0U ||
        config->maximum_latent_length == 0U ||
        config->decoder_input_channels == 0U ||
        config->decoder_output_channels == 0U ||
        config->initial_channels == 0U ||
        config->block_count > M3_VOCODER_BLOCK_COUNT ||
        config->residual_count > M3_VOCODER_RESIDUAL_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder plan configuration is invalid");
    }
    if (config->decoder_input_channels > UINT64_MAX / 2U) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder latent channel count overflows");
    }
    if (config->latent_channels !=
        config->decoder_input_channels * 2U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder folded channel configuration is invalid");
    }
    channels = config->initial_channels;
    for (block = 0U; block < config->block_count; ++block) {
        if (config->strides[block] == 0U ||
            config->strides[block] > UINT64_MAX / 2U ||
            channels < 2U || (channels & 1U) != 0U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "vocoder block configuration is invalid");
        }
        channels /= 2U;
    }
    return M3_STATUS_OK;
}

m3_status m3_vocoder_plan_build(
    const m3_vocoder_plan_config *config, m3_vocoder_plan *plan,
    m3_error *error)
{
    m3_vocoder_plan built;
    uint64_t shape3[3];
    uint64_t shape1[1];
    uint64_t channels;
    size_t entry_index = 0U;
    size_t block;
    m3_status status;

    if (plan == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder plan output is null");
    }
    status = m3_vocoder_plan_config_validate(config, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_vocoder_plan_init(&built);
    built.entry_count = 7U + config->block_count *
        (3U + 6U * config->residual_count);
    built.source_count = 9U + config->block_count *
        (4U + 8U * config->residual_count);
    built.block_count = config->block_count;
    built.residual_count = config->residual_count;
    built.config = *config;
    built.entries = calloc(built.entry_count, sizeof(*built.entries));
    if (built.entries == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate vocoder materialization plan");
    }
    shape3[0] = config->decoder_output_channels;
    shape3[1] = config->decoder_input_channels;
    shape3[2] = 1U;
    status = m3_vocoder_add_copy(
        &built, &entry_index, "dec_in_proj.weight", 3U, shape3, error);
    shape1[0] = config->decoder_output_channels;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            &built, &entry_index, "dec_in_proj.bias", 1U, shape1, error);
    }
    shape3[0] = config->initial_channels;
    shape3[1] = config->decoder_output_channels;
    shape3[2] = 7U;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_norm(
            &built, &entry_index, "conv_in", shape3, error);
    }
    shape1[0] = config->initial_channels;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            &built, &entry_index, "conv_in.bias", 1U, shape1, error);
    }
    channels = config->initial_channels;
    for (block = 0U; block < config->block_count &&
                         status == M3_STATUS_OK;
         ++block) {
        uint64_t output = channels / 2U;

        status = m3_vocoder_add_block(
            &built, &entry_index, config, (unsigned int)block,
            channels, output, error);
        channels = output;
    }
    shape3[0] = 1U;
    shape3[1] = channels;
    shape3[2] = 1U;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            &built, &entry_index, "snake_out.alpha", 3U, shape3, error);
    }
    shape3[0] = 1U;
    shape3[1] = channels;
    shape3[2] = 7U;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_norm(
            &built, &entry_index, "conv_out", shape3, error);
    }
    shape1[0] = 1U;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_add_copy(
            &built, &entry_index, "conv_out.bias", 1U, shape1, error);
    }
    if (status == M3_STATUS_OK && entry_index != built.entry_count) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "vocoder plan count is inconsistent");
    }
    if (status != M3_STATUS_OK) {
        m3_vocoder_plan_dispose(&built);
        return status;
    }
    m3_vocoder_plan_dispose(plan);
    *plan = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
