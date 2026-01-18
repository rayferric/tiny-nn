cmake_minimum_required(VERSION 3.20)
project(tnn LANGUAGES C)

option(ENABLE_VULKAN_BACKEND "Enable Vulkan backend support" ON)
option(BUILD_EXAMPLES "Build example executables" ON)
file(GLOB_RECURSE TARGET_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
# exclude backend directory sources
list(FILTER TARGET_SOURCES EXCLUDE REGEX ".*/src/backend/.*")

add_library(tnn ${TARGET_SOURCES})
target_include_directories(tnn
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(tnn PRIVATE m)

# re-add cpu backend files
file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu/*")
target_sources(tnn PRIVATE ${CPU_SOURCES})

if(ENABLE_VULKAN_BACKEND)
    message(STATUS "Vulkan backend enabled")
    find_package(Vulkan REQUIRED)
    file(GLOB_RECURSE VULKAN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/*")
    target_sources(tnn PRIVATE ${VULKAN_SOURCES})
    target_link_libraries(tnn PRIVATE Vulkan::Vulkan)
    target_compile_definitions(tnn PRIVATE ENABLE_VULKAN_BACKEND)
endif()

if(BUILD_EXAMPLES)
    add_executable(tnn_example_mnist_mlp_train example/mnist_mlp/train.c)
    target_link_libraries(tnn_example_mnist_mlp_train PRIVATE tnn)

    add_executable(tnn_example_cifar10_resnet_train example/cifar10_resnet/train.c)
    target_link_libraries(tnn_example_cifar10_resnet_train PRIVATE tnn)

    add_executable(tnn_example_vulkan example/vulkan/main.c)
    target_link_libraries(tnn_example_vulkan PRIVATE tnn)
endif()
