cmake_minimum_required(VERSION 3.20)
project(tnn LANGUAGES C)

file(GLOB_RECURSE TARGET_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
add_library(tnn ${TARGET_SOURCES})
target_include_directories(tnn
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(tnn_example__cifar_mlp__train example/cifar_mlp/train.c)
target_link_libraries(tnn_example__cifar_mlp__train PRIVATE tnn)
