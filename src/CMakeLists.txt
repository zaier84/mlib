add_library(mlib_core)

target_include_directories(mlib_core PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
$<INSTALL_INTERFACE:include>
)

target_sources(mlib_core PRIVATE
	core/tensor.cpp
)

