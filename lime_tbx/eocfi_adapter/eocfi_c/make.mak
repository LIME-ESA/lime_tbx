# set OS to your operating system (SOLARIS, WINDOWS, LINUX64_LEGACY, MACOS)

OS		= WINDOWS64

# set CC to your compiler (gcc)

CC		= cl.exe

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
#       Compiler flags
#       ==============
#
#
#        /c              = suppress linking
#        /ML             = ...
#        /D "WIN64"      = conditional code for Windows 95 (compiler)
#        /D "NDEBUG"     = do not include conditional code for debugging (compiler)
#        /D "_DEBUG"     = include conditional code for debugging (compiler)
#        /D "_WINDOWS"   = ... (compiler)
#        /D "WINDOWS" = include conditional code (CFI libraries)
#
#        /I              = header files directories
#
#------------------------------------------------------------------------------

CFLAGS = \
        /c /nologo /W0 /EHsc /Od /RTC1 \
        /D "_LIB"  /D "NDEBUG" /D "_CONSOLE" /D "_MBCS"\
        /D "WIN64"\
        /D "_WINDOWS"\
        /D "WINDOWS"\
        /I "include"

#------------------------------------------------------------------------------
#       Linker tool, flags and libraries
#------------------------------------------------------------------------------

LINK_STA = link.exe /nologo /stack:0xb71b00 /subsystem:console /incremental:yes /machine:X64
LINK_DLL = link.exe /nologo /stack:0xb71b00 /subsystem:console /incremental:yes /machine:X64 /NODEFAULTLIB:LIBCMT.lib

LIBS_DIR = /libpath:".\lib\WINDOWS64"\
           /libpath:"$(EXPCFI_TOOLS)\pthread\WINDOWS\lib" \
           /libpath:"cfi_tools"

LIBS = 	libexplorer_data_handling.lib\
		libexplorer_file_handling.lib\
		libexplorer_lib.lib\
		libexplorer_orbit.lib\
        libgeotiff.lib libtiff.lib libproj.lib libxml2.lib pthread.lib Ws2_32.lib pthreadVC2.lib
		
EXAMPLE = $(CFI)

#------------------------------------------------------------------------------
#	Makefile default target
#------------------------------------------------------------------------------

WIN_STA: example_sta run
WIN_DLL: example_dll run

#------------------------------------------------------------------------------
#	Compilation after preprocessing rules
#------------------------------------------------------------------------------


default : example

example:
	echo "--------------------"
	echo "$(CFI): ... creating the example (WIN_STA)"
	echo "--------------------"
  
	$(CC) $(CFLAGS) code\get_positions.c
	$(LINK_STA) get_positions.obj /out:bin\get_positions_win64.exe $(LIBS_DIR) $(LIBS)
	-erase *.obj
	-erase *.ilk

shared:	
	echo "--------------------"
	echo "$(CFI): ... creating the example (WIN_DLL)"
	echo "--------------------"
  
	$(CC) $(CFLAGS) code\get_positions.c
	$(LINK_STA) /DLL /OUT:bin\get_positions_win64.dll get_positions.obj $(LIBS_DIR) $(LIBS)
	-erase *.obj
	-erase bin\*.ilk
#-erase *.obj
#-erase *.ilk

# gcc -std=c99 -fPIC -shared -m64 -Iinclude -DWINDOWS64 code/get_positions.c -Llib/WINDOWS64 -lexplorer_orbit -lexplorer_lib -lexplorer_data_handling -lexplorer_file_handling -lgeotiff -ltiff -lproj -lxml2 -lm -lc -lpthread -o bin/get_position_win64.so

clean: 
	
