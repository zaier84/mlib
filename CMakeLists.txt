cmake_minimum_required(VERSION 3.22.0)
project(mlib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MLIB_BUILD_TESTS "Build the mlib tests" ON)

add_subdirectory(src)

if(MLIB_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

