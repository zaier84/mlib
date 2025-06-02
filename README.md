# mlib: A Modern C++ Machine Learning/Deep Learning Library

**mlib** is an in-development, header-only (currently) C++ library designed for machine learning and deep learning tasks. It aims to provide a clean, efficient, and user-friendly interface inspired by popular Python libraries like NumPy and PyTorch, but with the performance benefits and control of C++.

This project is a learning endeavor and a work in progress, focusing on building core ML/DL components from the ground up.

## Vision
The long-term vision for mlib is to offer:
*   A powerful and flexible `Tensor` class as the fundamental data structure.
*   A comprehensive suite of mathematical operations for tensor manipulation.
*   An automatic differentiation engine (autograd) for building and training neural networks.
*   Common neural network layers, activation functions, and optimizers.
*   A focus on modern C++ (C++17 and later) features and best practices.
*   Clear documentation and examples.

## Current Status
mlib is in its early stages of development. Currently, it features:
*   A foundational `Tensor` class supporting:
    *   Multiple data types.
    *   Dynamic N-dimensional shapes.
    *   Basic constructors (shape-based, data-based, fill-value).
    *   Element access via `at()` and variadic `operator()`.
    *   Copy and move semantics.
    *   Reshaping capabilities.
*   Initial element-wise arithmetic operations:
    *   Addition (`+`)
    *   Subtraction (`-`)
    *   Unary Plus (`+`)
    *   Unary Minus (`-`)
*   A CMake-based build system.
*   Unit tests using Google Test.

## Getting Started (Preliminary)

**Prerequisites:**
*   A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
*   CMake (version 3.22.0 or higher recommended).
*   Git (for cloning).

**Building (Example for a Unix-like environment):**
```bash
# Clone the repository
git clone https://github.com/zaier84/mlib.git mlib
cd mlib

# Configure the build
mkdir build && cd build
cmake ..

# Build the library (if applicable) and tests
cmake --build . # or make

# Run tests
ctest -V
