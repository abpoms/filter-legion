ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0	  # Use the shared low level
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE		?= bug_out
# List all the application source files here
GEN_SRC		?= main.cc # .cc files
GEN_GPU_SRC	?=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
CC_FLAGS	?= -std=c++11 -Wno-deprecated-register
NVCC_FLAGS	:=
GASNET_FLAGS	:=

GASNET=/usr/local/GASNet
USE_CUDA=0
CONDUIT=udp

# Include the legion Makefile so we can link in their dependencies
include $(LG_RT_DIR)/runtime.mk
