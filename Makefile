CC        := cc
CFLAGS    := -O2 -Wall -Wextra -std=c11
TEDDY_CFLAGS := -Iinclude -D_POSIX_C_SOURCE=200809L
BUILD_DIR := build

SRCS := src/compute_backend.c src/gpu_compute.c src/math_ops.c src/matrix_ops.c src/computation_engine.c src/dataset_ops.c src/model_train.c
OBJS := $(SRCS:src/%.c=$(BUILD_DIR)/%.o)

KERNEL := $(shell uname -s)
SYSTEM_HAS_OPENCL := 0
OPENCL_HEADER_DIRECTORY :=

ifneq ($(wildcard /usr/include/CL/cl.h),)
	OPENCL_HEADER_DIRECTORY := /usr/include
else ifneq ($(wildcard /usr/local/include/CL/cl.h),)
	OPENCL_HEADER_DIRECTORY := /usr/local/include
else ifneq ($(wildcard /opt/rocm/include/CL/cl.h),)
	OPENCL_HEADER_DIRECTORY := /opt/rocm/include
endif

ifeq ($(KERNEL),Darwin)
	SYSTEM_HAS_OPENCL := 1
	LDFLAGS := -framework OpenCL
else
	ifneq ($(shell pkg-config --exists OpenCL 2>/dev/null && echo yes),)
    SYSTEM_HAS_OPENCL := 1
    TEDDY_CFLAGS += $(shell pkg-config --cflags OpenCL)
    LDFLAGS := $(shell pkg-config --libs OpenCL) -lm
  else ifneq ($(and $(OPENCL_HEADER_DIRECTORY),$(wildcard /usr/lib/libOpenCL.*)),)
    SYSTEM_HAS_OPENCL := 1
    TEDDY_CFLAGS += -I$(OPENCL_HEADER_DIRECTORY)
    LDFLAGS := -lOpenCL -lm
  else ifneq ($(and $(OPENCL_HEADER_DIRECTORY),$(wildcard /usr/lib/x86_64-linux-gnu/libOpenCL.*)),)
    SYSTEM_HAS_OPENCL := 1
    TEDDY_CFLAGS += -I$(OPENCL_HEADER_DIRECTORY)
    LDFLAGS := -lOpenCL -lm
  else ifneq ($(wildcard /opt/rocm/lib/libOpenCL.*),)
    SYSTEM_HAS_OPENCL := 1
    TEDDY_CFLAGS += -I/opt/rocm/include
    LDFLAGS := -L/opt/rocm/lib -lOpenCL -lm
  else
    LDFLAGS := -lm
  endif
endif

ifeq ($(SYSTEM_HAS_OPENCL),1)
	TEDDY_CFLAGS += -DSYSTEM_HAS_OPENCL
endif

ifeq ($(FORCE_CPU),1)
	SYSTEM_HAS_OPENCL := 0
	TEDDY_CFLAGS := $(filter-out -DSYSTEM_HAS_OPENCL,$(TEDDY_CFLAGS))
	LDFLAGS := -lm
endif

TARGET := $(BUILD_DIR)/teddy_train

.PHONY: all clean data info

all: $(TARGET)

info:
ifeq ($(SYSTEM_HAS_OPENCL),1)
	@echo "Teddy: OpenCL detected, using GPU for Model training..."
else
	@echo "Teddy: OpenCL not detected, using CPU for Model training..."
endif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(TEDDY_CFLAGS) -c $< -o $@

$(BUILD_DIR)/teddy_train.o: train_teddy/train_teddy.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(TEDDY_CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(BUILD_DIR)/teddy_train.o
	$(CC) $(CFLAGS) $(TEDDY_CFLAGS) $^ $(LDFLAGS) -o $@

data:
	python3 dwnldr/download_dataset.py

clean:
	rm -rf $(BUILD_DIR)
