/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MUSIC3_CONFIG_H
#define M3_MUSIC3_CONFIG_H

#include "m3_model.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    m3_component_id id;
    bool valid;
    uint32_t hidden_size;
    uint32_t condition_layers;
    uint32_t codebooks;
    uint32_t condition_dim;
    uint32_t input_channels;
    uint32_t latent_channels;
    uint32_t output_hop_length;
    uint32_t output_sampling_rate;
    uint32_t upsampling_product;
    uint32_t sampling_rate;
} m3_music3_component_config;

void m3_music3_component_config_init(m3_music3_component_config *config);
m3_status m3_music3_config_read_file(
    const char *path, m3_component_id id,
    m3_music3_component_config *config, m3_error *error);
m3_status m3_music3_config_validate_cross(
    const m3_music3_component_config configs[M3_COMPONENT_COUNT],
    m3_error *error);

#endif
