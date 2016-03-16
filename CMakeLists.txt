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

if(DEFINED CMAKECONFIG_PATH)
    CPM_AddModule("cmakeconfig"
        SOURCE_DIR "${CMAKECONFIG_PATH}"
    )
else()
    CPM_AddModule("cmakeconfig"
    GIT_REPOSITORY "https://github.com/BioroboticsLab/cmakeconfig.git"
    GIT_TAG "master"
    )
endif()

include_biotracker_core(master)
include_pipeline(use-mxnet)
include_deeplocalizer_models(use-mxnet)

CPM_AddModule("cpm_bayesopt"
    GIT_REPOSITORY "https://github.com/BioroboticsLab/cpm_bayesopt.git"
    GIT_TAG master
)

CPM_InitModule(${CPM_MODULE_NAME})

biorobotics_set_compiler_flags()

# project dependecies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
find_package(PythonLibs REQUIRED)

# include external dependecies as SYSTEM headers to prevent warnings from
# external files. unfortunately, this has no effect when using MSVC.
include_directories(
    ${PROJECT_SOURCE_DIR}
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${Boost_INCLUDE_DIRS}
)

add_subdirectory(parameteroptimization)
