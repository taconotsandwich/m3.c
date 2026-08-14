/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WEIGHT_STAGE_FIXTURE_H
#define M3_WEIGHT_STAGE_FIXTURE_H

#include "m3_backend_internal.h"
#include "m3_safetensors.h"
#include "m3_weight_stage_internal.h"

#include "model_loader_fixture.h"

typedef struct {
    char root[M3_TEST_PATH_CAPACITY];
    char a_path[M3_TEST_PATH_CAPACITY];
    char z_path[M3_TEST_PATH_CAPACITY];
    m3_safetensors_metadata a_metadata;
    m3_safetensors_metadata z_metadata;
    m3_weight_table table;
} m3_weight_stage_test_fixture;

typedef struct {
    size_t allocation_calls;
    size_t free_calls;
    size_t fail_allocation_call;
} m3_weight_stage_fake_context;

bool m3_weight_stage_test_fixture_create(
    m3_weight_stage_test_fixture *fixture, m3_error *error);
bool m3_weight_stage_test_fixture_dispose(
    m3_weight_stage_test_fixture *fixture);
bool m3_weight_stage_test_fake_backend_create(
    uint64_t maximum_storage_bytes,
    uint64_t recommended_working_set_bytes,
    size_t fail_allocation_call, m3_backend **backend,
    m3_weight_stage_fake_context **context, m3_error *error);

#endif
