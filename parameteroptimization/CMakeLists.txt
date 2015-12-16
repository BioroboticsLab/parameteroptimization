file(GLOB_RECURSE src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

set(main_src ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

set(lib_src ${src})
list(REMOVE_ITEM lib_src ${main_src})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DUSE_DEEPLOCALIZER)

add_library(${CPM_LIB_TARGET_NAME} STATIC
	${lib_src} ${hdr}
)

add_executable(${CPM_BIN_TARGET_NAME}
    ${main_src} ${hdr}
)

target_link_libraries(${CPM_LIB_TARGET_NAME}
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    ${CPM_LIBRARIES}
)

target_link_libraries(${CPM_BIN_TARGET_NAME}
    ${CPM_LIB_TARGET_NAME}
)
