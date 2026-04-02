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

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND NEURAL_TEXTURES_SOURCES train.cu)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA compiler not found; building without train.cu")
endif()

add_executable(neural_textures ${NEURAL_TEXTURES_SOURCES})

if(CMAKE_CUDA_COMPILER)
    set_target_properties(neural_textures PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()
