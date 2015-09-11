file(GLOB_RECURSE src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)
file(GLOB_RECURSE ui RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ui)
file(GLOB_RECURSE qrc RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.qrc)

add_definitions(-DUSE_DEEPLOCALIZER)

qt5_add_resources(UI_RESOURCES ${qrc})
qt5_wrap_ui(UI_HEADERS ${ui})

link_directories(${DEEPLOCALIZER_CLASSIFIER_LIBRARY_DIR})

get_target_property(PIPELINE_INCLUDES Pipeline PIPELINE_INCLUDES)

include_directories(${PIPELINE_INCLUDES})

add_executable(pipelineParameterOptimization
    ${src} ${hdr} ${UI_RESOURCES} ${UI_HEADERS}
)

target_link_libraries(pipelineParameterOptimization
    BioTrackerLib
    "deeplocalizer-classifier"
    Pipeline
    ${ALGORITHM_LIBRARIES}
    Qt5::Widgets
    Qt5::OpenGL
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    -Wl,--whole-archive ${Caffe_LIBRARIES} -Wl,--no-whole-archive
    ${LMDB_LIBRARIES}
    ${CUDA_LIBRARIES}
    bayesopt
    nlopt
)
