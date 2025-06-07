# mlib: A Modern C++ Machine Learning/Deep Learning Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**mlib** is a header-only C++ library (in development) for machine learning and deep learning tasks. Built with modern C++ (C++17+), it offers a clean, efficient, and user-friendly interface inspired by NumPy and PyTorch, delivering high performance and fine-grained control.

## Table of Contents

- [Vision](#vision)
- [Current Status](#current-status)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Features Highlight](#features-highlight)
- [Future Development](#future-development)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## Vision

The long-term vision for mlib is to offer:

- A powerful and flexible `Tensor` class as the fundamental data structure.
- A comprehensive suite of mathematical operations for tensor manipulation.
- An automatic differentiation engine (autograd) for building and training neural networks.
- Common neural network layers, activation functions, and optimizers.
- A focus on modern C++ (C++17 and later) features and best practices.
- Clear documentation and examples.

### Design Philosophy

mlib is currently header-only for ease of use and integration, with plans to evaluate compiled components for performance-critical features in the future.

## Current Status

mlib is in its early stages of development, with significant progress on its core components. Currently, it features:

- **Tensor Class**:
  - Multiple data types (`int`, `float`, `double`, `bool`).
  - Dynamic N-dimensional shapes, including scalar (0-dim) and zero-sized dimensions.
  - Basic constructors (shape-based, data-based, fill-value).
  - Element access via `at()` and variadic `operator()`.
  - Correct copy and move semantics.
  - Reshaping capabilities.
  - Contiguous data handling.
- **Element-wise Arithmetic Operations**:
  - Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`).
  - Tensor-Tensor and Tensor-Scalar operations.
  - Unary Plus (`+`) and Unary Minus (`-`).
- **Unary Mathematical Functions** (element-wise):
  - Exponential (`exp`), Natural Logarithm (`log`), Square Root (`sqrt`), Absolute Value (`abs`).
- **Element-wise Comparison Operations**:
  - Equality (`==`), Inequality (`!=`), Greater (`>`), Less (`<`), Greater Equal (`>=`), Less Equal (`<=`).
  - Results in `Tensor<bool>` (boolean masks).
- **Matrix Multiplication**:
  - `matmul()` for 2D tensors.
- **Full Reduction Operations**:
  - `sum()`, `mean()`, `max_val()`, `min_val()`, `prod()`.
- **Axis-wise Reduction Operations**:
  - `sum()`, `mean()`, `max_val()`, `min_val()`, `prod()` along a specified axis, with `keep_dims` option.
- CMake-based build system for cross-platform compatibility.
- Comprehensive unit tests (Google Test).

### Known Limitations

- Advanced indexing/slicing is not yet implemented.
- Autograd and neural network layers are under development.
- Limited optimization for large-scale tensors (e.g., no GPU support yet).

## Getting Started

**Prerequisites:**

- C++17 compliant compiler (e.g., GCC, Clang, MSVC).
- CMake (version 3.22.0 or higher recommended).
- Git (for cloning).

**Building (Unix-like environment):**

```bash
# Clone the repository
git clone https://github.com/zaier84/mlib.git
cd mlib
# Configure the build
mkdir build && cd build
cmake ..
# Build the library and tests
cmake --build .
# Run unit tests
ctest -V
```

### Quick Start

1. Include the library:
   ```cpp
   #include <mlib/core/tensor.hpp>
   using namespace mlib::core;
   ```
2. Write a simple program:
   ```cpp
   int main() {
       Tensor<float> t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
       Tensor<float> result = t + 2.0f;
       std::cout << result << std::endl; // Tensor(shape: {2, 2}, data: [3.0, 4.0, 5.0, 6.0])
       return 0;
   }
   ```
3. Compile with your project.

## Documentation

Detailed documentation is under development. For now, refer to:

- Code comments in `include/mlib/`.
- Example code in the [Features Highlight](#features-highlight) section.
- Unit tests in the `tests/` directory.

## Features Highlight

```cpp
// Example: Basic Tensor operations
mlib::core::Tensor<float> A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
mlib::core::Tensor<float> B({2, 3}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

// Element-wise addition
mlib::core::Tensor<float> C = A + B;
// C: Tensor<float>({2, 3}, {8.0, 10.0, 12.0, 14.0, 16.0, 18.0})

// Tensor-scalar multiplication
mlib::core::Tensor<float> D = A * 2.0f;
// D: Tensor<float>({2, 3}, {2.0, 4.0, 6.0, 8.0, 10.0, 12.0})

// Matrix multiplication
mlib::core::Tensor<float> M1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
mlib::core::Tensor<float> M2({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
mlib::core::Tensor<float> P = mlib::core::matmul(M1, M2);
// P: Tensor<float>({2, 2}, {19.0, 22.0, 43.0, 50.0})

// Sum reduction along an axis
mlib::core::Tensor<float> axis_sum = mlib::core::sum(A, 1);
// axis_sum: Tensor<float>({2}, {6.0, 15.0})

// Element-wise comparison
mlib::core::Tensor<bool> mask = A > 3.5f;
// mask: Tensor<bool>({2, 3}, {false, false, false, true, true, true})
```

## Future Development

- Advanced indexing and slicing (tensor views).
- Complex linear algebra operations (transpose, N-D matmul).
- Automatic differentiation engine (autograd).
- Neural network layers and activation functions.
- Serialization capabilities.

## Contributing

mlib is a personal learning project, but feedback is welcome:

- Open issues on the [GitHub Issues page](https://github.com/zaier84/mlib/issues).
- Provide detailed bug reports or feature suggestions.
  Contribution guidelines will be added as the project matures.

## Contact

For questions or feedback:

- Open an issue on [GitHub Issues](https://github.com/zaier84/mlib/issues).
- Join [GitHub Discussions](https://github.com/zaier84/mlib/discussions) (if enabled).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

