# set OS to your operating system (SOLARIS, WINDOWS, LINUX64_LEGACY, MACOS)

  OS		= MACIN64

# C standard

  STD           = -std=c99

# set CC to your compiler (gcc)

  CC		= gcc $(STD)

#
#
#  End of Customization Area
#
#########################################################################################

CFI		= moon_sun-ephemerides
LANGUAGE	= c
EXT		= _c

#------------------------------------------------------------------------------
#       Set silent mode
#------------------------------------------------------------------------------

.SILENT :

#------------------------------------------------------------------------------
#       Bash shell
#------------------------------------------------------------------------------

SHELL = /bin/bash

#------------------------------------------------------------------------------
#	Compiler flags
#	--------------
#
#	 I	  = header files directories
#	 D$(OS)	  = conditional code for SOLARIS/AIX
#
#------------------------------------------------------------------------------

CFLAGS = \
        -m64 \
	-Iinclude/$(OS) \
	-D$(OS) -Wall

DEBUG_FLAGS = \
		-g -DDEBUG

#------------------------------------------------------------------------------
#       Linker tool, flags and libraries 
#------------------------------------------------------------------------------

LIBS_DIR = \
	-Llib/$(OS)

LIBS = 	\
	-lexplorer_orbit \
	-lexplorer_visibility \
	-lexplorer_pointing \
	-lexplorer_lib \
	-lexplorer_data_handling \
	-lexplorer_file_handling \
	-lgeotiff -ltiff -lproj -lxml2 -lm -lc -lpthread


ARCH = $(shell arch)

BIN_NAME = get_positions_darwin

ifeq ($(ARCH), arm64)
BIN_NAME = get_positions_darwin_arm
endif

default : executable

executable:
	echo "--------------------"
	echo "$(CFI): ... creating the executable"
	echo "--------------------"
	$(CC) $(CFLAGS) -O3 code/get_positions.c $(LIBS_DIR) $(LIBS) -o bin/$(BIN_NAME)

debug:
	echo "--------------------"
	echo "$(CFI): ... creating the debug-executable"
	echo "--------------------"
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) code/get_positions.c $(LIBS_DIR) $(LIBS) -o bin/get_positions

shared:
	echo "--------------------"
	echo "$(CFI): ... creating shared library"
	echo "--------------------"
	$(CC) -fPIC -shared $(CFLAGS) -O3 code/get_positions.c $(LIBS_DIR) $(LIBS) -o bin/$(BIN_NAME).so

shared_debug:
	echo "--------------------"
	echo "$(CFI): ... creating shared debug library"
	echo "--------------------"
	$(CC) -fPIC -shared $(CFLAGS) $(DEBUG_FLAGS) code/get_positions.c $(LIBS_DIR) $(LIBS) -o bin/$(BIN_NAME).so

spice:
	echo "--------------------"
	echo "SPICE example: ... creating the executable"
	echo "--------------------"
	$(CC) $(CFLAGS) -g -Wall -o ./bin/spice_example ./code/spice_example.c -I/opt/spice/cspice/include /opt/spice/cspice/lib/csupport.a /opt/spice/cspice/lib/cspice.a -lm


clean: 
	\rm -f *.o *.out core *% bin/*

