/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_test.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void m3_test_report_failure(const m3_test_context *test,
                                   const char *message, const char *file,
                                   int line)
{
    (void)fprintf(stderr, "%s:%d: FAIL [%s] %s\n", file, line,
                  test->case_name, message);
}

int m3_test_run(const char *suite_name, const m3_test_case *cases,
                size_t case_count)
{
    size_t total_checks = 0U;
    size_t total_failures = 0U;
    size_t index;

    if (suite_name == NULL || cases == NULL || case_count == 0U) {
        (void)fprintf(stderr, "invalid test suite\n");
        return 1;
    }

    for (index = 0U; index < case_count; ++index) {
        m3_test_context test = {cases[index].name, 0U, 0U};

        if (cases[index].name == NULL || cases[index].function == NULL) {
            (void)fprintf(stderr, "invalid test case at index %zu\n", index);
            total_failures += 1U;
            continue;
        }
        cases[index].function(&test);
        total_checks += test.check_count;
        total_failures += test.failure_count;
    }

    if (total_failures != 0U) {
        (void)fprintf(stderr, "%s: %zu of %zu checks failed\n", suite_name,
                      total_failures, total_checks);
        return 1;
    }

    (void)printf("%s: %zu cases, %zu checks passed\n", suite_name,
                 case_count, total_checks);
    return 0;
}

void m3_test_expect(m3_test_context *test, bool condition,
                    const char *message, const char *file, int line)
{
    if (test == NULL) {
        return;
    }
    test->check_count += 1U;
    if (!condition) {
        test->failure_count += 1U;
        m3_test_report_failure(test, message, file, line);
    }
}

bool m3_test_f32_close(float actual, float expected, float absolute_tolerance,
                       float relative_tolerance)
{
    double difference;
    double scale;
    double relative_limit;

    if (!isfinite(absolute_tolerance) || !isfinite(relative_tolerance) ||
        absolute_tolerance < 0.0F || relative_tolerance < 0.0F) {
        return false;
    }
    if (actual == expected) {
        return true;
    }
    if (!isfinite(actual) || !isfinite(expected)) {
        return false;
    }

    difference = fabs((double)actual - (double)expected);
    scale = fmax(fabs((double)actual), fabs((double)expected));
    relative_limit = (double)relative_tolerance * scale;
    return difference <= (double)absolute_tolerance ||
           difference <= relative_limit;
}

void m3_test_expect_f32(m3_test_context *test, float actual, float expected,
                        float absolute_tolerance, float relative_tolerance,
                        const char *message, const char *file, int line)
{
    if (test == NULL) {
        return;
    }
    test->check_count += 1U;
    if (!m3_test_f32_close(actual, expected, absolute_tolerance,
                           relative_tolerance)) {
        test->failure_count += 1U;
        m3_test_report_failure(test, message, file, line);
        (void)fprintf(stderr,
                      "  actual %.9g, expected %.9g, abs_tol %.9g, "
                      "rel_tol %.9g\n",
                      (double)actual, (double)expected,
                      (double)absolute_tolerance,
                      (double)relative_tolerance);
    }
}

static bool m3_test_write_all(int descriptor, const uint8_t *data, size_t size)
{
    size_t offset = 0U;

    while (offset < size) {
        ssize_t written = write(descriptor, data + offset, size - offset);

        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        offset += (size_t)written;
    }
    return true;
}

bool m3_test_temp_file_create(m3_test_temp_file *file,
                              const m3_test_fixture *fixture)
{
    char path[] = "/tmp/m3-test-XXXXXX";
    size_t path_length;
    int descriptor;
    bool success;

    if (file == NULL) {
        return false;
    }
    file->path[0] = '\0';
    if (fixture == NULL || fixture->name == NULL ||
        (fixture->size != 0U && fixture->data == NULL)) {
        return false;
    }

    descriptor = mkstemp(path);
    if (descriptor < 0) {
        return false;
    }
    success = m3_test_write_all(descriptor, fixture->data, fixture->size);
    if (close(descriptor) != 0) {
        success = false;
    }
    if (!success) {
        (void)unlink(path);
        return false;
    }

    path_length = strlen(path);
    if (path_length >= sizeof(file->path)) {
        (void)unlink(path);
        return false;
    }
    (void)memcpy(file->path, path, path_length + 1U);
    return true;
}

bool m3_test_temp_file_remove(m3_test_temp_file *file)
{
    if (file == NULL || file->path[0] == '\0') {
        return false;
    }
    if (unlink(file->path) != 0) {
        return false;
    }
    file->path[0] = '\0';
    return true;
}
