cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0104 NEW)
project(neural_textures)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

set(NEURAL_TEXTURES_SOURCES
    main.cpp
)

set(NT_COMPILE_DEFINITIONS)
if (WIN32)
    set(NT_COMPILE_DEFINITIONS NOMINMAX)
endif()

set(NT_TINYEXR_INCLUDE_DIRECTORY "")
if (EXISTS "C:/Users/maven/workspace/NvRTX/UnrealEngine/Engine/Source/ThirdParty/astcenc/4.2.0/Source/tinyexr.h")
    set(NT_TINYEXR_INCLUDE_DIRECTORY
        "C:/Users/maven/workspace/NvRTX/UnrealEngine/Engine/Source/ThirdParty/astcenc/4.2.0/Source"
    )
elseif(EXISTS "C:/Users/maven/workspace/openpgl/third-party/tinyexr/tinyexr.h")
    set(NT_TINYEXR_INCLUDE_DIRECTORY
        "C:/Users/maven/workspace/openpgl/third-party/tinyexr"
    )
endif()

if (NOT NT_TINYEXR_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "Could not find a local tinyexr.h. Expected one in the NvRTX or openpgl workspaces.")
endif()

set(NT_INCLUDE_DIRECTORIES
    third_party/tiny-cuda-nn/include
    third_party/tiny-cuda-nn/dependencies
    third_party/tiny-cuda-nn/dependencies/cutlass/include
    third_party/tiny-cuda-nn/dependencies/cutlass/tools/util/include
    ${NT_TINYEXR_INCLUDE_DIRECTORY}
)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND NEURAL_TEXTURES_SOURCES
        tests/forward_pass_test.cu
        train.cu
    )
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA compiler not found; building without train.cu")
endif()

add_executable(neural_textures ${NEURAL_TEXTURES_SOURCES})

if(CMAKE_CUDA_COMPILER)
    set_target_properties(neural_textures PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(neural_textures PRIVATE ${NT_COMPILE_DEFINITIONS})
    target_compile_definitions(neural_textures PRIVATE
        NT_HAS_CUDA=1
        TCNN_MIN_GPU_ARCH=75
        TCNN_HALF_PRECISION=1
    )
    target_include_directories(neural_textures PRIVATE
        ${NT_INCLUDE_DIRECTORIES}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
else()
    target_compile_definitions(neural_textures PRIVATE ${NT_COMPILE_DEFINITIONS} NT_HAS_CUDA=0)
    target_include_directories(neural_textures PRIVATE ${NT_INCLUDE_DIRECTORIES})
endif()
