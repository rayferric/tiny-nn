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
target_link_libraries(tnn PRIVATE m)

add_executable(tnn_example__mnist_mlp__train example/mnist_mlp/train.c)
target_link_libraries(tnn_example__mnist_mlp__train PRIVATE tnn)

add_executable(tnn_example__cifar10_resnet__train example/cifar10_resnet/train.c)
target_link_libraries(tnn_example__cifar10_resnet__train PRIVATE tnn)
