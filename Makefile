###############################################################################
#    Filter example Makefile
###############################################################################

BUILD_DIR     := build
OBJECT_DIR    := $(BUILD_DIR)/out
SOURCE_DIR    := src

OUT   := $(BUILD_DIR)/filter_example

# Source code variables
SOURCE_FILES := \
  main.cpp \
  util.cpp \
  jpeg/JPEGReader.cpp \
  jpeg/JPEGWriter.cpp \
  image_operations.cpp \
  compute_features.cpp

HALIDE_SRC := \
  to_conv_patch.cpp

OBJECTS := $(SOURCE_FILES:%.cpp=$(OBJECT_DIR)/%.o)

# Halide variables
HALIDE_INC_PATH=`echo ~`/repos/Halide/include
HALIDE_LIB_PATH=`echo ~`/repos/Halide/bin

CAFFE_INC_PATH=`echo ~`/repos/caffe/include
CAFFE_LIB_PATH=`echo ~`/repos/caffe/build/lib

HDF5_INC_PATH=/usr/include/hdf5/serial
HDF5_LIB_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial

# GCS library variables
GCS_INC_PATH=./go_gcs/src/gcsbindings
GCS_LIB_PATH=$(GCS_INC_PATH)

GCC = g++

# Includes
INCLUDE_FLAGS += \
  -I$(GCS_INC_PATH) \
  -I$(CAFFE_INC_PATH) \
  -I$(HDF5_INC_PATH)

# Compiler flags
GCC_FLAGS   ?=
GCC_FLAGS += -DCPU_ONLY

# Linker flags
LD_FLAGS += \
  -L$(GCS_LIB_PATH) -lgcs \
  -ljpeg \
  -lz \
  -L$(CAFFE_LIB_PATH) -lcaffe -L$(HDF5_LIB_PATH) -lhdf5 -lglog \
  -fopenmp

HALIDE_GEN := $(HALIDE_SRC:%.cpp=src/halide/%_gen)
HALIDE_OBJS := $(HALIDE_SRC:%.cpp=src/halide/%.o)


.PHONY: default
default: $(OUT)

###############################################################################
#
#    Legion Configuration
#
###############################################################################

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0	  # Use the shared low level
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE		?=
# List all the application source files here
GEN_SRC		?= # .cc files
GEN_GPU_SRC	?=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
CC_FLAGS	?= -std=c++11 -fPIC -DLEGION_PROF
NVCC_FLAGS	:=
GASNET_FLAGS	:=

GASNET=/usr/local/GASNet
USE_CUDA=0
CONDUIT=udp

# Include the legion Makefile so we can link in their dependencies
include $(LG_RT_DIR)/runtime.mk

###############################################################################
#
#    Final build rules
#      * We need to put these after including runtime.mk so that we have all of
#        Legion's dependencies
#
###############################################################################

GCC_FLAGS += $(CC_FLAGS)
INCLUDE_FLAGS += $(INC_FLAGS)

$(OUT): dirs $(HALIDE_OBJS) $(OBJECTS) gcs_go $(SLIB_LEGION) $(SLIB_REALM) $(SLIB_SHAREDLLR)
	$(GCC) -o $@ -std=c++11 -I./src $(HALIDE_OBJS) $(OBJECTS) $(LEGION_LD_FLAGS) $(LD_FLAGS) $(GASNET_FLAGS) $(LEGION_LIBS) 

dirs: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OBJECT_DIR)
	mkdir -p $(OBJECT_DIR)/jpeg

$(OBJECTS): $(OBJECT_DIR)/%.o : $(SOURCE_DIR)/%.cpp
	$(GCC) -o $@ -c $< $(GCC_FLAGS) $(INCLUDE_FLAGS)

gcs_go: $(GCS_LIB_PATH)/libgcs.a
	cd $(GCS_LIB_PATH) && GOPATH=`pwd`../../../ go get
	GOPATH=`pwd`/go_gcs $(MAKE) -C $(GCS_LIB_PATH) -f Makefile

$(HALIDE_OBJS) : %.o : %.cpp
	$(GCC) -o $(@:%.o=%_gen) $< -g -ggdb -std=c++11 \
	-I$(HALIDE_INC_PATH) -L$(HALIDE_LIB_PATH) -lHalide && \
	cd ./src/halide && ../../$(@:%.o=%_gen)

clean::
	@$(RM) -rf $(BUILD_DIR)

veryclean: clean
	$(MAKE) -C $(LG_RT_DIR) -f runtime.mk clean
