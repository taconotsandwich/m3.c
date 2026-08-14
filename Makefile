# SPDX-License-Identifier: GPL-2.0-only

CC = clang
AR = ar

CPPFLAGS = -Iinclude
CFLAGS = -std=c11 -O3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion

BUILD_DIR = build
LIBRARY = $(BUILD_DIR)/libm3.a
TEST_BINARY = $(BUILD_DIR)/test_m3
LIBRARY_OBJECTS = $(BUILD_DIR)/m3.o
TEST_OBJECTS = $(BUILD_DIR)/test_m3.o

.PHONY: all clean test

all: $(LIBRARY)

test: $(TEST_BINARY)
	$(TEST_BINARY)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/m3.o: src/m3.c include/m3.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/m3.c -o $@

$(BUILD_DIR)/test_m3.o: tests/test_m3.c include/m3.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c tests/test_m3.c -o $@

$(LIBRARY): $(LIBRARY_OBJECTS)
	$(AR) rcs $@ $(LIBRARY_OBJECTS)

$(TEST_BINARY): $(TEST_OBJECTS) $(LIBRARY)
	$(CC) $(CFLAGS) $(TEST_OBJECTS) $(LIBRARY) -o $@

clean:
	rm -rf $(BUILD_DIR)
