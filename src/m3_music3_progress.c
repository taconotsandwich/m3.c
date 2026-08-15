/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

static m3_status m3_music3_progress_emit(
    m3_music3_progress_state *state, uint64_t completed,
    m3_error *error)
{
    if (!state->active || completed < state->last_completed ||
        completed > state->total) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 progress is not monotonic");
    }
    if (completed == state->last_completed) {
        return M3_STATUS_OK;
    }
    if (state->callback != NULL &&
        !state->callback(state->context, state->phase, completed,
                         state->total)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "Music3 generation was cancelled");
    }
    state->last_completed = completed;
    return M3_STATUS_OK;
}

void m3_music3_progress_init(
    m3_music3_progress_state *state,
    m3_music3_progress_callback callback, void *context)
{
    state->callback = callback;
    state->context = context;
    state->phase = M3_MUSIC3_PHASE_COUNT;
    state->total = 0U;
    state->last_completed = 0U;
    state->active = false;
}

m3_status m3_music3_progress_begin(
    m3_music3_progress_state *state, m3_music3_phase phase,
    uint64_t total, m3_error *error)
{
    bool first;

    if (state == NULL || state->active || total == 0U ||
        phase < M3_MUSIC3_PHASE_PREPARE ||
        phase >= M3_MUSIC3_PHASE_COUNT) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 progress phase is invalid");
    }
    first = state->phase == M3_MUSIC3_PHASE_COUNT;
    if ((first && phase != M3_MUSIC3_PHASE_PREPARE) ||
        (!first && (int)phase != (int)state->phase + 1)) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 progress phase order is invalid");
    }
    state->phase = phase;
    state->total = total;
    state->last_completed = 0U;
    state->active = true;
    if (state->callback != NULL &&
        !state->callback(state->context, phase, 0U, total)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "Music3 generation was cancelled");
    }
    return M3_STATUS_OK;
}

m3_status m3_music3_progress_finish(
    m3_music3_progress_state *state, m3_error *error)
{
    m3_status status;

    if (state == NULL || !state->active) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 progress phase is not active");
    }
    status = m3_music3_progress_emit(state, state->total, error);
    state->active = false;
    return status;
}

void m3_music3_progress_bridge_init(
    m3_music3_progress_bridge *bridge,
    m3_music3_progress_state *state, uint64_t offset,
    uint64_t child_total)
{
    bridge->state = state;
    bridge->offset = offset;
    bridge->child_total = child_total;
    bridge->last_child_completed = 0U;
    bridge->saw_total = false;
    bridge->cancelled = false;
    bridge->invalid = false;
}

bool m3_music3_progress_bridge_report(
    void *context, uint64_t completed, uint64_t total)
{
    m3_music3_progress_bridge *bridge = context;
    m3_music3_progress_state *state;
    uint64_t mapped;
    m3_error ignored;
    m3_status status;

    if (bridge == NULL || bridge->state == NULL) {
        return false;
    }
    state = bridge->state;
    if (!state->active || total != bridge->child_total ||
        completed < bridge->last_child_completed || completed > total ||
        bridge->offset > state->total ||
        completed > state->total - bridge->offset) {
        bridge->invalid = true;
        return false;
    }
    bridge->last_child_completed = completed;
    if (completed == total) {
        bridge->saw_total = true;
        return true;
    }
    mapped = bridge->offset + completed;
    m3_error_reset(&ignored);
    status = m3_music3_progress_emit(state, mapped, &ignored);
    if (status == M3_STATUS_CANCELLED) {
        bridge->cancelled = true;
    } else if (status != M3_STATUS_OK) {
        bridge->invalid = true;
    }
    return status == M3_STATUS_OK;
}

m3_status m3_music3_progress_bridge_complete(
    m3_music3_progress_bridge *bridge, m3_status child_status,
    m3_error *error)
{
    uint64_t mapped;
    m3_status status;

    if (bridge == NULL || bridge->state == NULL || bridge->invalid) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 child progress is invalid");
    }
    if (bridge->cancelled) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "Music3 generation was cancelled");
    }
    if (child_status != M3_STATUS_OK) {
        return child_status;
    }
    if (!bridge->saw_total ||
        bridge->child_total > UINT64_MAX - bridge->offset) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Music3 child progress did not complete");
    }
    mapped = bridge->offset + bridge->child_total;
    if (mapped == bridge->state->total) {
        return M3_STATUS_OK;
    }
    status = m3_music3_progress_emit(bridge->state, mapped, error);
    return status;
}
