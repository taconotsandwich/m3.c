/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_guided_sampling.h"

#include "m3_op_internal.h"

#include <math.h>
#include <stddef.h>

#define M3_GUIDED_TOP_K 50U
#define M3_GUIDED_CFG_SCALE 1.5F
#define M3_GUIDED_FINITE_LIMIT 1.0e9F
#define M3_GUIDED_SUM_FLOOR 1.0e-12F

typedef struct {
    const m3_tensor_view *eos;
    const m3_tensor_view *codes;
    size_t candidate_count;
    bool semantic;
} m3_guided_source;

static float m3_guided_load(const m3_guided_source *source, size_t row,
                            size_t candidate)
{
    const m3_tensor_view *view = source->codes;
    size_t flat_index;

    if (source->semantic && candidate == 0U) {
        view = source->eos;
        flat_index = row;
    } else {
        size_t code = source->semantic ? candidate - 1U : candidate;

        flat_index = row * (size_t)view->metadata.shape[1] + code;
    }
    return m3_op_load_float(view, m3_op_element_offset(view, flat_index));
}

#pragma STDC FP_CONTRACT OFF
static float m3_guided_cfg(const m3_guided_source *source, size_t candidate)
{
    float conditional = m3_guided_load(source, 0U, candidate);
    float unconditional = m3_guided_load(source, 1U, candidate);
    float difference = conditional - unconditional;
    float scaled = difference * M3_GUIDED_CFG_SCALE;

    return unconditional + scaled;
}
#pragma STDC FP_CONTRACT DEFAULT

static float m3_guided_sanitize(float value)
{
    if (isnan(value) || value == -INFINITY) {
        return -M3_GUIDED_FINITE_LIMIT;
    }
    if (value == INFINITY) {
        return M3_GUIDED_FINITE_LIMIT;
    }
    return value;
}

static float m3_guided_threshold(const m3_guided_source *source,
                                 bool conditional, float first_threshold)
{
    m3_top_pair pairs[M3_GUIDED_TOP_K];
    size_t pair_count = 0U;
    size_t k = source->candidate_count < M3_GUIDED_TOP_K
                   ? source->candidate_count
                   : M3_GUIDED_TOP_K;
    size_t index;

    for (index = 0U; index < source->candidate_count; ++index) {
        float value = conditional ? m3_guided_load(source, 0U, index)
                                  : m3_guided_cfg(source, index);

        if (!conditional && source->semantic &&
            m3_guided_load(source, 0U, index) < first_threshold) {
            value = -INFINITY;
        }
        if (!conditional) {
            value = m3_guided_sanitize(value);
        }
        m3_top_consider(pairs, &pair_count, k, value, (int32_t)index);
    }
    return pairs[k - 1U].value;
}

static float m3_guided_value(const m3_guided_source *source,
                             size_t candidate, float first_threshold,
                             float threshold)
{
    float value = m3_guided_cfg(source, candidate);

    if (source->semantic &&
        m3_guided_load(source, 0U, candidate) < first_threshold) {
        value = -INFINITY;
    }
    value = m3_guided_sanitize(value);
    if (value < threshold) {
        value = -INFINITY;
    }
    return value;
}

static size_t m3_guided_sample(const m3_guided_source *source,
                               float uniform)
{
    float first_threshold = source->semantic
                                ? m3_guided_threshold(source, true, 0.0F)
                                : -INFINITY;
    float threshold = m3_guided_threshold(source, false, first_threshold);
    float maximum = -INFINITY;
    float sum = 0.0F;
    float cumulative = 0.0F;
    float denominator;
    size_t selected = 0U;
    size_t index;

    for (index = 0U; index < source->candidate_count; ++index) {
        float value = m3_guided_value(source, index, first_threshold,
                                      threshold);

        if (value > maximum) {
            maximum = value;
        }
    }
    for (index = 0U; index < source->candidate_count; ++index) {
        float value = m3_guided_value(source, index, first_threshold,
                                      threshold);
        float probability = expf(value - maximum);

        if (isnan(probability)) {
            probability = 0.0F;
        }
        sum = sum + probability;
    }
    denominator = fmaxf(sum, M3_GUIDED_SUM_FLOOR);
    for (index = 0U; index < source->candidate_count; ++index) {
        float value = m3_guided_value(source, index, first_threshold,
                                      threshold);
        float probability = expf(value - maximum);

        if (isnan(probability)) {
            probability = 0.0F;
        }
        probability = probability / denominator;
        if (probability > 0.0F) {
            selected = index;
        }
        cumulative = cumulative + probability;
        if (uniform < cumulative) {
            selected = index;
            break;
        }
    }
    return selected;
}

static m3_status m3_guided_validate_view(
    const m3_backend *backend, const m3_tensor_view *view,
    uint64_t candidate_count, const char *name, m3_error *error)
{
    m3_status status = m3_op_check_view(backend, view, false, name, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!m3_op_dtype_float(view->metadata.dtype) ||
        view->metadata.rank != 2U || view->metadata.shape[0] != 2U ||
        view->metadata.shape[1] != candidate_count) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s must be a float [2,%llu] tensor", name,
                            (unsigned long long)candidate_count);
    }
    return M3_STATUS_OK;
}

static m3_status m3_guided_validate_common(const m3_backend *backend,
                                           float uniform,
                                           const void *output,
                                           m3_error *error)
{
    if (backend == NULL || output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "guided backend and output are required");
    }
    if (!isfinite(uniform) || uniform < 0.0F || uniform >= 1.0F) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "guided uniform is outside [0,1)");
    }
    return M3_STATUS_OK;
}

static m3_status m3_guided_validate_semantic_conditional(
    const m3_guided_source *source, m3_error *error)
{
    size_t index;

    for (index = 0U; index < source->candidate_count; ++index) {
        if (isnan(m3_guided_load(source, 0U, index))) {
            return m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "semantic conditional logits contain NaN");
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_guided_sample_semantic(
    const m3_backend *backend, const m3_tensor_view *eos_logits,
    const m3_tensor_view *semantic_logits, float uniform,
    m3_guided_semantic_sample *sample, m3_error *error)
{
    m3_guided_source source;
    m3_guided_semantic_sample result;
    size_t selected;
    m3_status status = m3_guided_validate_common(backend, uniform, sample,
                                                 error);

    if (status == M3_STATUS_OK) {
        status = m3_guided_validate_view(backend, eos_logits, 1U,
                                         "semantic EOS logits", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_guided_validate_view(
            backend, semantic_logits, M3_GUIDED_SEMANTIC_CODE_COUNT,
            "semantic code logits", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    source.eos = eos_logits;
    source.codes = semantic_logits;
    source.candidate_count = M3_GUIDED_SEMANTIC_CODE_COUNT + 1U;
    source.semantic = true;
    status = m3_guided_validate_semantic_conditional(&source, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    selected = m3_guided_sample(&source, uniform);
    result.eos = selected == 0U;
    result.code = result.eos ? 0U : (uint32_t)(selected - 1U);
    *sample = result;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_guided_sample_residual(
    const m3_backend *backend, const m3_tensor_view *logits, float uniform,
    uint32_t *code, m3_error *error)
{
    m3_guided_source source;
    size_t selected;
    m3_status status = m3_guided_validate_common(backend, uniform, code,
                                                 error);

    if (status == M3_STATUS_OK) {
        status = m3_guided_validate_view(
            backend, logits, M3_GUIDED_RESIDUAL_CODE_COUNT,
            "residual logits", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    source.eos = NULL;
    source.codes = logits;
    source.candidate_count = M3_GUIDED_RESIDUAL_CODE_COUNT;
    source.semantic = false;
    selected = m3_guided_sample(&source, uniform);
    *code = (uint32_t)selected;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
