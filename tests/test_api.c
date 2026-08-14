/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3.h"
#include "m3_error.h"

#include <string.h>

void m3_test_api_contract(m3_test_context *test)
{
    char long_message[512];
    m3_error error;
    m3_status status;

    M3_TEST_EXPECT(test, strcmp(m3_version(), M3_VERSION_STRING) == 0,
                   "version string");
    M3_TEST_EXPECT(test, strcmp(m3_status_string(M3_STATUS_OK), "ok") == 0,
                   "success status string");
    M3_TEST_EXPECT(test,
                   strcmp(m3_status_string(M3_STATUS_INVALID_FORMAT),
                          "invalid format") == 0,
                   "format status string");
    M3_TEST_EXPECT(test,
                   strcmp(m3_status_string(M3_STATUS_CANCELLED),
                          "cancelled") == 0,
                   "cancelled status string");
    M3_TEST_EXPECT(test,
                   strcmp(m3_status_string((m3_status)99),
                          "unknown status") == 0,
                   "unknown status string");

    m3_error_reset(&error);
    M3_TEST_EXPECT(test, error.status == M3_STATUS_OK, "reset error status");
    M3_TEST_EXPECT(test, strcmp(m3_error_message(&error), "") == 0,
                   "reset error message");

    status = m3_error_set(&error, M3_STATUS_IO, "cannot read shard %u", 3U);
    M3_TEST_EXPECT(test, status == M3_STATUS_IO,
                   "formatted error return status");
    M3_TEST_EXPECT(test, error.status == M3_STATUS_IO,
                   "formatted error stored status");
    M3_TEST_EXPECT(test, strcmp(error.message, "cannot read shard 3") == 0,
                   "formatted error message");

    status = m3_error_set(&error, M3_STATUS_OVERFLOW, NULL);
    M3_TEST_EXPECT(test, status == M3_STATUS_OVERFLOW,
                   "default error return status");
    M3_TEST_EXPECT(test, strcmp(error.message, "overflow") == 0,
                   "default error message");

    (void)memset(long_message, 'x', sizeof(long_message));
    long_message[sizeof(long_message) - 1U] = '\0';
    (void)m3_error_set(&error, M3_STATUS_INTERNAL, "%s", long_message);
    M3_TEST_EXPECT(test,
                   strlen(error.message) == M3_ERROR_MESSAGE_CAPACITY - 1U,
                   "long error truncation");
    M3_TEST_EXPECT(test,
                   error.message[M3_ERROR_MESSAGE_CAPACITY - 1U] == '\0',
                   "long error termination");

    status = m3_error_set(NULL, M3_STATUS_IO, "ignored");
    M3_TEST_EXPECT(test, status == M3_STATUS_IO,
                   "null error storage return status");
    M3_TEST_EXPECT(test, strcmp(m3_error_message(NULL), "") == 0,
                   "null error message");

    (void)m3_error_set(&error, M3_STATUS_OK, "ignored");
    M3_TEST_EXPECT(test,
                   error.status == M3_STATUS_OK && error.message[0] == '\0',
                   "success clears error");
}
