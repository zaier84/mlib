add_library(mlib_core INTERFACE)

target_include_directories(mlib_core INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	$<INSTALL_INTERFACE:include>
)

target_sources(mlib_core PRIVATE
	# core/tensor.cpp
	# core/operations.cpp
)

