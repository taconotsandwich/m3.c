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
LIBRARY_C_SOURCES := $(sort $(wildcard src/*.c))
LIBRARY_OBJECTIVE_C_SOURCES := $(sort $(wildcard src/*.m))
TEST_SOURCES := $(sort $(wildcard tests/*.c))
LIBRARY_C_OBJECTS = $(patsubst src/%.c,$(BUILD_DIR)/src/%.o,$(LIBRARY_C_SOURCES))
LIBRARY_OBJECTIVE_C_OBJECTS = $(patsubst src/%.m,$(BUILD_DIR)/src/%.m.o,$(LIBRARY_OBJECTIVE_C_SOURCES))
LIBRARY_OBJECTS = $(LIBRARY_C_OBJECTS) $(LIBRARY_OBJECTIVE_C_OBJECTS)
TEST_OBJECTS = $(patsubst tests/%.c,$(BUILD_DIR)/tests/%.o,$(TEST_SOURCES))
DEPENDENCY_FILES = $(LIBRARY_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)

ifneq ($(strip $(LIBRARY_OBJECTIVE_C_SOURCES)),)
OBJECTIVE_C_FLAGS = -fobjc-arc
APPLE_FRAMEWORKS = -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders \
	-framework MetalPerformanceShadersGraph -framework Accelerate
endif

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

$(BUILD_DIR)/src/%.m.o: src/%.m | $(BUILD_DIR)/src
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OBJECTIVE_C_FLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.c | $(BUILD_DIR)/tests
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(LIBRARY): $(LIBRARY_OBJECTS)
	$(AR) rcs $@ $(LIBRARY_OBJECTS)

$(TEST_BINARY): $(TEST_OBJECTS) $(LIBRARY)
	$(CC) $(CFLAGS) $(LDFLAGS) $(TEST_OBJECTS) $(LIBRARY) $(LDLIBS) \
		$(APPLE_FRAMEWORKS) -o $@

clean:
	rm -rf build build-sanitize

-include $(DEPENDENCY_FILES)
