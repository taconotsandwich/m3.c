/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MODEL_LOADER_FIXTURE_H
#define M3_MODEL_LOADER_FIXTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "m3_model.h"

#define M3_TEST_PATH_CAPACITY 256U

extern const char m3_loader_test_manifest[];

bool m3_loader_test_path(char path[M3_TEST_PATH_CAPACITY],
                         const char *directory, const char *name);
bool m3_loader_test_write_file(const char *path, const uint8_t *data,
                               size_t size);
bool m3_loader_test_write_json(const char *path, const char *json);
bool m3_loader_test_write_safetensors(const char *path, const char *header,
                                      size_t payload_size);
bool m3_loader_test_make_root(char root[M3_TEST_PATH_CAPACITY]);
bool m3_loader_test_remove_tree(const char *path);
bool m3_loader_test_create_layout(char root[M3_TEST_PATH_CAPACITY]);
bool m3_test_music3_write_config(const char *path, m3_component_id id);

#endif
