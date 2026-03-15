CC        := cc
CFLAGS    := -O2 -Wall -Wextra -Iinclude -std=c11
LDFLAGS   := -lm
BUILD_DIR := build

SRCS := src/compute_backend.c
OBJS := $(SRCS:src/%.c=$(BUILD_DIR)/%.o)

.PHONY: all clean

all: $(OBJS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
