cmake_minimum_required(VERSION 2.6)

if (POLICY CMP0020)
	cmake_policy (SET CMP0020 NEW)
endif(POLICY CMP0020)

if (POLICY CMP0021)
	cmake_policy (SET CMP0021 NEW)
endif(POLICY CMP0021)

project(ParameterOptimization)

#-----------------------------------------------------------------------
# CPM configuration
#-----------------------------------------------------------------------
set(CPM_MODULE_NAME ParameterOptimization)

set(CPM_BIN_TARGET_NAME ${CPM_MODULE_NAME})
set(CPM_LIB_TARGET_NAME ${CPM_MODULE_NAME}Lib)

if ((DEFINED CPM_DIR) AND (DEFINED CPM_UNIQUE_ID) AND (DEFINED CPM_TARGET_NAME))
  set(CPM_LIB_TARGET_NAME ${CPM_TARGET_NAME})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CPM_DIR})
  include(CPM)
else()
  set(CPM_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpm-packages" CACHE TYPE STRING)
  find_package(Git)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "CPM requires Git.")
  endif()
  if (NOT EXISTS ${CPM_DIR}/CPM.cmake)
    message(STATUS "Cloning repo (https://github.com/iauns/cpm)")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" clone https://github.com/iauns/cpm ${CPM_DIR}
      RESULT_VARIABLE error_code
      OUTPUT_QUIET ERROR_QUIET)
    if(error_code)
      message(FATAL_ERROR "CPM failed to get the hash for HEAD")
    endif()
  endif()
  include(${CPM_DIR}/CPM.cmake)
endif()

if(NOT DEFINED BIOTRACKER_CORE_PATH)
	CPM_AddModule("biotracker_core"
		GIT_REPOSITORY "https://github.com/BioroboticsLab/biotracker_core.git"
		GIT_TAG "master"
	)
else()
	CPM_AddModule("biotracker_core"
		SOURCE_DIR "${BIOTRACKER_CORE_PATH}"
	)
endif()

CPM_AddModule("cpm_bayesopt"
    GIT_REPOSITORY git@github.com:BioroboticsLab/cpm_bayesopt.git
    GIT_TAG master
)

set(USE_DEEPLOCALIZER "1")

if(NOT DEFINED PIPELINE_PATH)
	CPM_AddModule("pipeline"
		GIT_REPOSITORY "https://github.com/BioroboticsLab/pipeline.git"
		GIT_TAG "cpm"
	)
else()
	CPM_AddModule("pipeline"
		SOURCE_DIR "${PIPELINE_PATH}"
	)
endif()

CPM_InitModule(${CPM_MODULE_NAME})

# project dependecies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
find_package(PythonLibs REQUIRED)

# remove all previously set compiler and linker flags for debug build
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_EXE_LINKER_FLAGS "")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using Clang or GCC
    # minimal optimmization and debug symbols for debug builds
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
    # enable optimization for release builds
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG")
    # enable C++1y
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -fPIC")
    # enable all warnings
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Woverloaded-virtual -Wold-style-cast -Wnon-virtual-dtor -Wsign-promo -Wno-missing-braces")
    # warnings are errors
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=unused-variable -Wno-error=unused-parameter")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    # Force to always compile with W3 and treat warnings as errors
    # W4 would be preferable, but causes to many warnings in included files
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    endif()
	# warnings are errors
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
	# disable warning 4503 on visual studio (boost)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4503")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	# determine clang version
	EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
	string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
	if (CLANG_VERSION_STRING VERSION_GREATER 3.5)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=inconsistent-missing-override")
	endif()
	# gcc doesn't know about unused-private-field warning
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-private-field")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=absolute-value")
    # use Wdocumentation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdocumentation")
    # enable thread safety analysis
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wthread-safety")
    # enable thread safety analysis
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimplicit-fallthrough")
    # implicit conversion warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion")
    # osx clang3.6 throw additional warnings
    if(${APPLE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override -Wno-deprecated-declarations")
        set(CMAKE_CXX_LINK_FLAGS_DEBUG "-lc++abi")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_CXX_LINK_FLAGS_DEBUG}")
    endif()
endif()

message("${CLANG_VERSION}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # clang doesnt know about unused-but-set-variable warning
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-but-set-variable")

	execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
					OUTPUT_VARIABLE GCC_VERSION)
	if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
		# implicit conversion warnings
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion")
	endif()
endif()

# use runtime analyzers when using clang/gcc in debug mode
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
		(GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9))
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-optimize-sibling-calls")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined")
	endif()

	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-optimize-sibling-calls")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=integer")
	endif()
endif()

# include external dependecies as SYSTEM headers to prevent warnings from
# external files. unfortunately, this has no effect when using MSVC.
include_directories(
    ${PROJECT_SOURCE_DIR}
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${Boost_INCLUDE_DIRS}
)

add_subdirectory(parameteroptimization)