CC        := cc
CFLAGS    := -O2 -Wall -Wextra -Iinclude -std=c11
BUILD_DIR := build

SRCS := src/compute_backend.c
OBJS := $(SRCS:src/%.c=$(BUILD_DIR)/%.o)

UNAME_S := $(shell uname -s)

HAS_OPENCL := 0
OPENCL_HEADER_DIR :=

ifneq ($(wildcard /usr/include/CL/cl.h),)
    OPENCL_HEADER_DIR := /usr/include
else ifneq ($(wildcard /usr/local/include/CL/cl.h),)
    OPENCL_HEADER_DIR := /usr/local/include
else ifneq ($(wildcard /opt/rocm/include/CL/cl.h),)
    OPENCL_HEADER_DIR := /opt/rocm/include
endif

ifeq ($(UNAME_S),Darwin)
    HAS_OPENCL := 1
    LDFLAGS := -framework OpenCL
else
    ifneq ($(shell pkg-config --exists OpenCL 2>/dev/null && echo yes),)
        HAS_OPENCL := 1
        CFLAGS += $(shell pkg-config --cflags OpenCL)
        LDFLAGS := $(shell pkg-config --libs OpenCL) -lm
    else ifneq ($(and $(OPENCL_HEADER_DIR),$(wildcard /usr/lib/libOpenCL.*)),)
        HAS_OPENCL := 1
        CFLAGS += -I$(OPENCL_HEADER_DIR)
        LDFLAGS := -lOpenCL -lm
    else ifneq ($(and $(OPENCL_HEADER_DIR),$(wildcard /usr/lib/x86_64-linux-gnu/libOpenCL.*)),)
        HAS_OPENCL := 1
        CFLAGS += -I$(OPENCL_HEADER_DIR)
        LDFLAGS := -lOpenCL -lm
    else ifneq ($(wildcard /opt/rocm/lib/libOpenCL.*),)
        HAS_OPENCL := 1
        CFLAGS += -I/opt/rocm/include
        LDFLAGS := -L/opt/rocm/lib -lOpenCL -lm
    else
        LDFLAGS := -lm
    endif
endif

ifeq ($(HAS_OPENCL),1)
    CFLAGS += -DMG_HAS_OPENCL
endif

ifeq ($(FORCE_CPU),1)
    HAS_OPENCL := 0
    CFLAGS := $(filter-out -DMG_HAS_OPENCL,$(CFLAGS))
    LDFLAGS := -lm
endif

.PHONY: all clean info

all: info $(OBJS)

info:
ifeq ($(HAS_OPENCL),1)
	@echo "[minigrad] OpenCL detected - building with GPU support"
else
	@echo "[minigrad] OpenCL not found - building CPU-only"
endif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
