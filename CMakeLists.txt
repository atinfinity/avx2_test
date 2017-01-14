cmake_minimum_required(VERSION 2.8)
project(avx2_test)

file(GLOB SOURCES "*.cpp" "*.c" "*.h" "*.hpp")
find_package(OpenCV REQUIRED)

set(EXTRA_C_FLAGS "")
set(EXTRA_CXX_FLAGS "")

if(MSVC)
    set(EXTRA_C_FLAGS "/arch:AVX2")
    set(EXTRA_CXX_FLAGS "/arch:AVX2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(EXTRA_C_FLAGS "-mavx2")
    set(EXTRA_CXX_FLAGS "-mavx2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(EXTRA_C_FLAGS "-mavx2")
    set(EXTRA_CXX_FLAGS "-mavx2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(EXTRA_C_FLAGS "-xCORE-AVX2")
    set(EXTRA_CXX_FLAGS "-xCORE-AVX2")
endif()

message(STATUS "EXTRA_C_FLAGS  : ${EXTRA_C_FLAGS}")
message(STATUS "EXTRA_CXX_FLAGS: ${EXTRA_CXX_FLAGS}")

# add compiler option
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")

if(OpenCV_FOUND)
    add_executable(avx2_test ${SOURCES})
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(avx2_test ${OpenCV_LIBS})
endif(OpenCV_FOUND)
