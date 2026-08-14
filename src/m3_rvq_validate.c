/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static m3_status m3_rvq_check(
    m3_backend *backend, const m3_tensor_view *view, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, const char *name,
    m3_error *error)
{
    return m3_rvq_condition_check_view(
        backend, view, dtype, rank, shape, name, error);
}

static m3_status m3_rvq_validate_config(const m3_rvq_config *config,
                                        m3_error *error)
{
    if (config == NULL || config->hidden_size == 0U ||
        config->layer_count == 0U ||
        config->layer_count > M3_RVQ_LAYER_CAPACITY ||
        config->attention_head_count == 0U ||
        config->hidden_size % config->attention_head_count != 0U ||
        config->intermediate_size == 0U ||
        config->position_count < M3_RVQ_CODEBOOK_COUNT ||
        config->dtype != M3_DTYPE_BF16) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ runtime configuration is invalid");
    }
    return M3_STATUS_OK;
}

static m3_status m3_rvq_validate_layer(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_layer_weights *layer, size_t index, m3_error *error)
{
    const uint64_t hidden[] = {config->hidden_size};
    const uint64_t hidden_matrix[] = {
        config->hidden_size, config->hidden_size
    };
    const uint64_t expanded[] = {
        config->intermediate_size, config->hidden_size
    };
    const uint64_t contracted[] = {
        config->hidden_size, config->intermediate_size
    };
    const m3_tensor_view *views[] = {
        layer->input_norm, layer->query, layer->key, layer->value,
        layer->attention_out, layer->post_attention_norm,
        layer->gate, layer->up, layer->down
    };
    const uint8_t ranks[] = {1U, 2U, 2U, 2U, 2U, 1U, 2U, 2U, 2U};
    const uint64_t *shapes[] = {
        hidden, hidden_matrix, hidden_matrix, hidden_matrix,
        hidden_matrix, hidden, expanded, expanded, contracted
    };
    size_t tensor;

    for (tensor = 0U; tensor < 9U; ++tensor) {
        char name[64];
        int length = snprintf(name, sizeof(name),
                              "RVQ layer %zu weight %zu", index, tensor);
        m3_status status;

        if (length < 0 || (size_t)length >= sizeof(name)) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "RVQ validation name overflows");
        }
        status = m3_rvq_check(backend, views[tensor], config->dtype,
                              ranks[tensor], shapes[tensor], name, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_rvq_validate_weights(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights, m3_error *error)
{
    const uint64_t audio[] = {
        (M3_RVQ_CODEBOOK_COUNT - 1U) * M3_RVQ_CODEBOOK_SIZE,
        config->hidden_size
    };
    const uint64_t matrix[] = {config->hidden_size, config->hidden_size};
    const uint64_t positions[] = {
        config->position_count, config->hidden_size
    };
    const uint64_t hidden[] = {config->hidden_size};
    const uint64_t head[] = {M3_RVQ_CODEBOOK_SIZE, config->hidden_size};
    size_t index;
    m3_status status;

    if (weights == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ weights are required");
    }
    status = m3_rvq_check(backend, weights->audio_embeddings,
                          config->dtype, 2U, audio,
                          "RVQ audio embeddings", error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_check(backend, weights->projection,
                              config->dtype, 2U, matrix,
                              "RVQ projection", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_check(backend, weights->position_embeddings,
                              config->dtype, 2U, positions,
                              "RVQ position embeddings", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_check(backend, weights->norm, config->dtype, 1U,
                              hidden, "RVQ final norm", error);
    }
    for (index = 0U; index < config->layer_count &&
                     status == M3_STATUS_OK; ++index) {
        status = m3_rvq_validate_layer(
            backend, config, &weights->layers[index], index, error);
    }
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT &&
                     status == M3_STATUS_OK; ++index) {
        status = m3_rvq_check(backend, weights->heads[index],
                              config->dtype, 2U, head,
                              "RVQ audio head", error);
    }
    return status;
}

m3_status m3_rvq_validate(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights, const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, const m3_rvq_frame *frame,
    size_t *output_bytes, m3_error *error)
{
    uint64_t last_shape[2];
    uint64_t semantic_shape[1];
    m3_tensor_metadata output_metadata;
    uint64_t output_shape[3];
    size_t index;
    m3_status status;

    if (backend == NULL || config == NULL || frame == NULL ||
        output_bytes == NULL ||
        uniforms == NULL || uniform_count != M3_RVQ_RESIDUAL_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ runtime arguments are invalid");
    }
    last_shape[0] = 2U;
    last_shape[1] = config->hidden_size;
    semantic_shape[0] = config->hidden_size;
    status = m3_rvq_validate_config(config, error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_validate_weights(backend, config, weights, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_check(backend, last_hidden, config->dtype, 2U,
                              last_shape, "RVQ last hidden", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_check(
            backend, semantic_embedding, config->dtype, 1U,
            semantic_shape, "RVQ semantic embedding", error);
    }
    for (index = 0U; index < uniform_count && status == M3_STATUS_OK;
         ++index) {
        if (!isfinite(uniforms[index]) || uniforms[index] < 0.0F ||
            uniforms[index] >= 1.0F) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                  "RVQ uniform is outside [0,1)");
        }
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    output_shape[0] = 1U;
    output_shape[1] = M3_RVQ_CODEBOOK_COUNT;
    output_shape[2] = config->hidden_size;
    status = m3_tensor_metadata_init(&output_metadata, config->dtype, 3U,
                                     output_shape, error);
    if (status == M3_STATUS_OK) {
        *output_bytes = output_metadata.byte_count;
        m3_error_reset(error);
    }
    return status;
}

static void m3_rvq_spec(m3_runtime_tensor_spec *spec, m3_dtype dtype,
                        uint8_t rank, const uint64_t *shape)
{
    uint8_t axis;

    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = dtype;
    spec->rank = rank;
    spec->alignment = 64U;
    for (axis = 0U; axis < rank; ++axis) {
        spec->shape[axis] = shape[axis];
    }
}

void m3_rvq_workspace_specs(const m3_rvq_config *config,
                            m3_runtime_tensor_spec *specs)
{
    const uint64_t sequence[] = {
        M3_RVQ_CODEBOOK_COUNT, 2U, config->hidden_size
    };
    const uint64_t positions[] = {M3_RVQ_CODEBOOK_COUNT};
    const uint64_t code_ids[] = {2U};
    const uint64_t token[] = {2U, config->hidden_size};
    const uint64_t position_vectors[] = {
        M3_RVQ_CODEBOOK_COUNT, config->hidden_size
    };
    const uint64_t hidden[] = {
        2U, M3_RVQ_CODEBOOK_COUNT, config->hidden_size
    };
    const uint64_t attention[] = {
        2U, config->attention_head_count, M3_RVQ_CODEBOOK_COUNT,
        config->hidden_size / config->attention_head_count
    };
    const uint64_t expanded[] = {
        2U, M3_RVQ_CODEBOOK_COUNT, config->intermediate_size
    };
    const uint64_t logits[] = {2U, M3_RVQ_CODEBOOK_SIZE};

    m3_rvq_spec(&specs[M3_RVQ_WS_SEQUENCE], config->dtype, 3U,
                sequence);
    m3_rvq_spec(&specs[M3_RVQ_WS_POSITION_IDS], M3_DTYPE_I32, 1U,
                positions);
    m3_rvq_spec(&specs[M3_RVQ_WS_CODE_IDS], M3_DTYPE_I32, 1U, code_ids);
    m3_rvq_spec(&specs[M3_RVQ_WS_TOKEN], config->dtype, 2U, token);
    m3_rvq_spec(&specs[M3_RVQ_WS_POSITIONS], config->dtype, 2U,
                position_vectors);
    m3_rvq_spec(&specs[M3_RVQ_WS_HIDDEN], config->dtype, 3U, hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_HIDDEN_TEMP], config->dtype, 3U,
                hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_NORM], config->dtype, 3U, hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_QUERY], config->dtype, 3U, hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_KEY], config->dtype, 3U, hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_VALUE], config->dtype, 3U, hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_ATTENTION], config->dtype, 4U,
                attention);
    m3_rvq_spec(&specs[M3_RVQ_WS_REORDER], config->dtype, 3U, hidden);
    m3_rvq_spec(&specs[M3_RVQ_WS_GATE], config->dtype, 3U, expanded);
    m3_rvq_spec(&specs[M3_RVQ_WS_UP], config->dtype, 3U, expanded);
    m3_rvq_spec(&specs[M3_RVQ_WS_LOGITS], config->dtype, 2U, logits);
}
