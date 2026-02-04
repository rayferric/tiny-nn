cmake_minimum_required(VERSION 3.20)
project(tnn LANGUAGES C)

option(ENABLE_AVX2 "Enable AVX2 optimizations in CPU backend" ON)
option(ENABLE_VULKAN_BACKEND "Enable Vulkan backend support" ON)
option(ENABLE_VULKAN_VALIDATION "Enable Vulkan validation layers" OFF)
option(BUILD_EXAMPLES "Build example executables" ON)
option(ENABLE_TRACY "Enable Tracy profiler instrumentation" ON)
set(TRACY_VERSION "v0.13.1" CACHE STRING "Tracy profiler version/tag to checkout")

file(GLOB_RECURSE TARGET_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
# exclude backend directory sources
list(FILTER TARGET_SOURCES EXCLUDE REGEX ".*/src/backend/.*")

add_library(tnn ${TARGET_SOURCES})
target_include_directories(tnn
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(tnn PRIVATE m)

# re-add cpu backend files
file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu/*")
target_sources(tnn PRIVATE ${CPU_SOURCES})

if(ENABLE_AVX2)
    message(STATUS "AVX2 enabled for CPU backend")
    target_compile_definitions(tnn PRIVATE ENABLE_AVX2)
    target_compile_options(tnn PRIVATE -mavx2 -mfma)
endif()

if(ENABLE_VULKAN_BACKEND)
    message(STATUS "Vulkan backend enabled")
    
    # add vulkan backend sources
    file(GLOB_RECURSE VULKAN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/*")
    target_sources(tnn PRIVATE ${VULKAN_SOURCES})
    target_compile_definitions(tnn PRIVATE ENABLE_VULKAN_BACKEND)
    
    # link vulkan 
    find_package(Vulkan REQUIRED)
    target_link_libraries(tnn PRIVATE Vulkan::Vulkan)

    # enable validation layers
    if(ENABLE_VULKAN_VALIDATION)
        message(STATUS "Vulkan validation layers enabled")
        target_compile_definitions(tnn PRIVATE ENABLE_VULKAN_VALIDATION)
    endif()

    # compile shaders

    find_program(GLSLANG glslangValidator REQUIRED)
    find_program(XXD xxd REQUIRED)
    file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/devices/vulkan/shaders/*.comp")

    set(GENERATED_HEADERS_DIR "${CMAKE_BINARY_DIR}/generated/vulkan_shaders")
    file(MAKE_DIRECTORY ${GENERATED_HEADERS_DIR})

    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        
        set(SPV_FILE "${GENERATED_HEADERS_DIR}/${SHADER_NAME}.spv")
        set(HEADER_FILE "${GENERATED_HEADERS_DIR}/${SHADER_NAME}.h")
        
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLANG} -V ${SHADER} -o ${SPV_FILE}
            DEPENDS ${SHADER}
        )
        
        add_custom_command(
            OUTPUT ${HEADER_FILE}
            COMMAND ${XXD} -i ${SHADER_NAME}.spv > ${HEADER_FILE}
            DEPENDS ${SPV_FILE}
            WORKING_DIRECTORY ${GENERATED_HEADERS_DIR}
        )
        
        list(APPEND GENERATED_SHADER_HEADERS ${HEADER_FILE})
    endforeach()

    target_include_directories(tnn PRIVATE "${CMAKE_BINARY_DIR}/generated")
    target_sources(tnn PRIVATE ${GENERATED_SHADER_HEADERS})
endif()

if(BUILD_EXAMPLES)
    add_executable(tnn_example_mnist_mlp_train example/mnist_mlp/train.c)
    target_link_libraries(tnn_example_mnist_mlp_train PRIVATE tnn)

    add_executable(tnn_example_cifar10_resnet_train example/cifar10_resnet/train.c)
    target_link_libraries(tnn_example_cifar10_resnet_train PRIVATE tnn)

    add_executable(tnn_example_vulkan_backend_tests example/vulkan_backend_tests/main.c)
    target_link_libraries(tnn_example_vulkan_backend_tests PRIVATE tnn)
endif()

if(ENABLE_TRACY)
    message(STATUS "Tracy profiler enabled (version: ${TRACY_VERSION})")
    target_compile_definitions(tnn PUBLIC ENABLE_TRACY)

    enable_language(CXX)
    include(FetchContent)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG ${TRACY_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)

    target_compile_definitions(tnn PRIVATE TRACY_ENABLE)
    target_link_libraries(tnn PUBLIC TracyClient)
endif()
