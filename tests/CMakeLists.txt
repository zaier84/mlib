
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest ${CMAKE_BINARY_DIR}/googletest)
add_executable(mlib_tests
	main.cpp
	core/test_tensor.cpp
	core/test_operations.cpp
)

target_link_libraries(mlib_tests PRIVATE mlib_core gtest_main)

include(GoogleTest)
gtest_discover_tests(mlib_tests)

