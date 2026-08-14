# SPDX-License-Identifier: GPL-2.0-only

CC = clang
AR = ar

CPPFLAGS = -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc -Itests
COMMON_CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
CFLAGS = $(COMMON_CFLAGS) -O3
SANITIZER_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
SANITIZER_CFLAGS = $(COMMON_CFLAGS) -O1 -g $(SANITIZER_FLAGS)
LDLIBS = -lm

BUILD_DIR = build
SANITIZER_BUILD_DIR = build-sanitize
LIBRARY = $(BUILD_DIR)/libm3.a
TEST_BINARY = $(BUILD_DIR)/test_m3
LIBRARY_SOURCES = src/m3.c \
	src/m3_error.c \
	src/m3_model.c \
	src/m3_tensor.c
TEST_SOURCES = tests/m3_test.c \
	tests/test_api.c \
	tests/test_fixture.c \
	tests/test_main.c \
	tests/test_model.c \
	tests/test_tensor.c
LIBRARY_OBJECTS = $(patsubst src/%.c,$(BUILD_DIR)/src/%.o,$(LIBRARY_SOURCES))
TEST_OBJECTS = $(patsubst tests/%.c,$(BUILD_DIR)/tests/%.o,$(TEST_SOURCES))
DEPENDENCY_FILES = $(LIBRARY_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)

.PHONY: all clean test test-sanitize

all: $(LIBRARY)

test: $(TEST_BINARY)
	$(TEST_BINARY)

test-sanitize:
	$(MAKE) BUILD_DIR=$(SANITIZER_BUILD_DIR) \
		CFLAGS="$(SANITIZER_CFLAGS)" \
		LDFLAGS="$(SANITIZER_FLAGS)" test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/src: | $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/src

$(BUILD_DIR)/tests: | $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/src/%.o: src/%.c | $(BUILD_DIR)/src
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.c | $(BUILD_DIR)/tests
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(LIBRARY): $(LIBRARY_OBJECTS)
	$(AR) rcs $@ $(LIBRARY_OBJECTS)

$(TEST_BINARY): $(TEST_OBJECTS) $(LIBRARY)
	$(CC) $(CFLAGS) $(LDFLAGS) $(TEST_OBJECTS) $(LIBRARY) $(LDLIBS) -o $@

clean:
	rm -rf build build-sanitize

-include $(DEPENDENCY_FILES)
