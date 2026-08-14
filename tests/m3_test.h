/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TEST_H
#define M3_TEST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_TEST_TEMP_PATH_CAPACITY 64U

typedef struct m3_test_context {
    const char *case_name;
    size_t check_count;
    size_t failure_count;
    bool skipped;
} m3_test_context;

typedef void (*m3_test_function)(m3_test_context *test);

typedef struct m3_test_case {
    const char *name;
    m3_test_function function;
} m3_test_case;

typedef struct m3_test_fixture {
    const char *name;
    const uint8_t *data;
    size_t size;
} m3_test_fixture;

typedef struct m3_test_temp_file {
    char path[M3_TEST_TEMP_PATH_CAPACITY];
} m3_test_temp_file;

int m3_test_run(const char *suite_name, const m3_test_case *cases,
                size_t case_count);
void m3_test_expect(m3_test_context *test, bool condition,
                    const char *message, const char *file, int line);
void m3_test_skip(m3_test_context *test, const char *reason,
                  const char *file, int line);
bool m3_test_f32_close(float actual, float expected, float absolute_tolerance,
                       float relative_tolerance);
void m3_test_expect_f32(m3_test_context *test, float actual, float expected,
                        float absolute_tolerance, float relative_tolerance,
                        const char *message, const char *file, int line);
bool m3_test_temp_file_create(m3_test_temp_file *file,
                              const m3_test_fixture *fixture);
bool m3_test_temp_file_remove(m3_test_temp_file *file);

#define M3_TEST_EXPECT(test, condition, message) \
    m3_test_expect((test), (condition), (message), __FILE__, __LINE__)

#define M3_TEST_SKIP(test, reason) \
    m3_test_skip((test), (reason), __FILE__, __LINE__)

#define M3_TEST_EXPECT_F32(test, actual, expected, absolute_tolerance, \
                           relative_tolerance, message) \
    m3_test_expect_f32((test), (actual), (expected), (absolute_tolerance), \
                       (relative_tolerance), (message), __FILE__, __LINE__)

#endif
