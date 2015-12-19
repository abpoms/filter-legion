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
  jpeg/JPEGWriter.cpp

OBJECTS := $(SOURCE_FILES:%.cpp=$(OBJECT_DIR)/%.o)

# Halide variables
# HALIDE_INC_PATH=`echo ~`/repos/Halide/include
# HALIDE_LIB_PATH=`echo ~`/repos/Halide/bin

# GCS library variables
GCS_INC_PATH=./go_gcs/src/gcsbindings
GCS_LIB_PATH=$(GCS_INC_PATH)

GCC = g++

# Includes
INCLUDE_FLAGS := \
  -I$(GCS_INC_PATH)

# Compiler flags
GCC_FLAGS   ?=

# Linker flags
LD_FLAGS    := \
  -L$(GCS_LIB_PATH) -lgcs \
  -ljpeg



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
CC_FLAGS	?= -std=c++11 -fPIC
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

$(OUT): dirs $(OBJECTS) gcs_go $(SLIB_LEGION) $(SLIB_REALM) $(SLIB_SHAREDLLR)
	$(GCC) -o $@ -std=c++11 -I./src $(OBJECTS) $(LEGION_LD_FLAGS) $(LD_FLAGS) $(GASNET_FLAGS) $(LEGION_LIBS) 

dirs: 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OBJECT_DIR)
	mkdir -p $(OBJECT_DIR)/jpeg

$(OBJECTS): $(OBJECT_DIR)/%.o : $(SOURCE_DIR)/%.cpp
	$(GCC) -o $@ -c $< $(GCC_FLAGS) $(INCLUDE_FLAGS)

gcs_go: $(GCS_LIB_PATH)/libgcs.a
	cd $(GCS_LIB_PATH) && GOPATH=`pwd`../../../ go get
	GOPATH=`pwd`/go_gcs $(MAKE) -C $(GCS_LIB_PATH) -f Makefile

clean::
	@$(RM) -rf $(BUILD_DIR)

veryclean: clean
	$(MAKE) -C $(LG_RT_DIR) -f runtime.mk clean
