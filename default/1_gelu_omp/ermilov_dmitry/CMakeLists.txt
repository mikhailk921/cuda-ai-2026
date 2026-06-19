cmake_minimum_required(VERSION 3.10)

project(GeluOMP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(gelu_app
    main.cpp
    gelu_omp.cpp
)

target_include_directories(gelu_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gelu_app PRIVATE
    OpenMP::OpenMP_CXX
)