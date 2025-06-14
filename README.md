# mlib: A Modern C++ Machine Learning/Deep Learning Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

**mlib** is a header-only C++ library (in development) for machine learning and deep learning tasks. Built with modern C++ (C++20+), it offers a clean, efficient, and user-friendly interface inspired by NumPy and PyTorch, delivering high performance and fine-grained control.

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
- A focus on modern C++ (C++20 and later) features and best practices.
- Clear documentation and examples.

### Design Philosophy

mlib is currently header-only for ease of use and integration, with plans to evaluate compiled components for performance-critical features in the future.

## Current Status

mlib is in its early stages of development, with significant progress on its core components. Currently, it features:

- **Robust Tensor Class**:
  - Multiple data types (`int`, `float`, `double`, `bool`).
  - Dynamic N-dimensional shapes, including scalar (0-dim) and zero-sized dimensions.
  - Basic constructors (shape-based, data-based, fill-value).
  - Element access via `at()` and variadic `operator()`.
  - Correct copy and move semantics.
  - Reshaping capabilities, including inference of one dimension (`-1` support).
  - `squeeze()`: Removes dimensions of size 1 (removes all or specific axis).
  - `unsqueeze()`: Inserts a new dimension of size 1 at a specified axis.
  - Contiguous data handling.
  - Stream insertion (`operator<<`) for easy printing.
- **Extensive Element-wise Arithmetic Operations**:
  - Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`).
  - Supports Tensor-Tensor and Tensor-Scalar interactions.
  - Unary Plus (`+`) and Unary Minus (`-`).
- **Unary Mathematical Functions** (element-wise):
  - Exponential (`exp`), Natural Logarithm (`log`), Square Root (`sqrt`), Absolute Value (`abs`).
- **Element-wise Comparison Operations**:
  - Equality (`==`), Inequality (`!=`), Greater (`>`), Less (`<`), Greater Equal (`>=`), Less Equal (`<=`).
  - Results in `Tensor<bool>` (boolean masks).
  - Supports Tensor-Tensor and Tensor-Scalar comparisons.
- **Element-wise Logical Operations**:
  - `logical_and()`, `logical_or()`, `logical_not()` (`operator!`).
  - Operates on `Tensor<bool>` inputs, returns `Tensor<bool>`.
- **Comprehensive Matrix Multiplication (`matmul`)**:
  - Performs `matmul` for 2D matrices (`(m,k) @ (k,n)` -> `(m,n)`).
  - Automatically handles common vector-matrix interactions:
    - 1D Vector @ 2D Matrix (`(k,) @ (k,n)` -> `(n,)`)
    - 2D Matrix @ 1D Vector (`(m,k) @ (k,)` -> `(m,)`)
    - 1D Vector @ 1D Vector (Dot Product: `(k,) @ (k,)` -> scalar `()`)
  - Correctly handles zero-sized dimensions in matrices/vectors within multiplication rules.
- **2D Matrix Transpose**:
  - `transpose()` for 2D tensors, creating a copied transposed matrix.
- **Full Reduction Operations**:
  - `sum()`, `mean()`, `max_val()`, `min_val()`, `prod()` across the entire tensor.
- **Axis-wise Reduction Operations**:
  - `sum()`, `mean()`, `max_val()`, `min_val()`, `prod()` along a specified axis, with `keep_dims` option.
- **Tensor Creation Routines**:
  - `zeros()`: Creates tensors filled with zeros.
  - `ones()`: Creates tensors filled with ones.
  - `full()`: Creates tensors filled with a specified value.
  - `eye()`: Creates 2D identity matrices.
  - `arange()`: Creates 1D tensors with evenly spaced values (like NumPy's `np.arange`).
  - `linspace()`: Creates 1D tensors with a specified number of evenly spaced samples over an interval (like NumPy's `np.linspace`).
- CMake-based build system for cross-platform compatibility.
- Comprehensive unit tests (Google Test) for all implemented features, ensuring correctness and robustness.

### Known Limitations

- Advanced indexing/slicing is not yet implemented (requires views).
- More complex linear algebra (e.g., QR decomposition, SVD) is not available.
- Automatic differentiation engine (autograd) is under development.
- Limited optimization for large-scale tensors (e.g., no GPU support, SIMD acceleration).

## Getting Started

**Prerequisites:**

- C++20 compliant compiler (e.g., GCC, Clang, MSVC).
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
   int main()
   {
       Tensor<float> t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
       Tensor<float> result = t + 2.0f;
       std::cout << result << std::endl;
       /* Expected Output for 'result':
       Tensor(shape: {2, 2}, strides: {2, 1}, total_size: 4, data: [
           [3, 4]
           [5, 6]
       ])
       */
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

// Tensor Creation Routines
mlib::core::Tensor<int> zeros_matrix = mlib::core::zeros<int>({2, 2});
// Tensor(shape: {2, 2}, strides: {2, 1}, total_size: 4, data: [
//     [0, 0]
//     [0, 0]
// ])

mlib::core::Tensor<float> identity_matrix = mlib::core::eye<float>(3);
// Tensor(shape: {3, 3}, strides: {3, 1}, total_size: 9, data: [
//     [1, 0, 0]
//     [0, 1, 0]
//     [0, 0, 1]
// ])

mlib::core::Tensor<int> range_tensor = mlib::core::arange<int>(0, 5, 1); // 0, 1, 2, 3, 4
// Tensor(shape: {5}, strides: {1}, total_size: 5, data: [0, 1, 2, 3, 4])

mlib::core::Tensor<float> linspace_tensor = mlib::core::linspace<float>(0.0f, 1.0f, 5); // 0.0, 0.25, 0.5, 0.75, 1.0
// Tensor(shape: {5}, strides: {1}, total_size: 5, data: [0, 0.25, 0.5, 0.75, 1])

// Tensor Shape Transformations
mlib::core::Tensor<float> unsqueezed_tensor = mlib::core::unsqueeze(A, 0); // Add a dim at axis 0
// Tensor(shape: {1, 2, 3}, strides: {6, 3, 1}, total_size: 6, data: [
//   [
//     [1, 2, 3]
//     [4, 5, 6]
//   ]
// ])

mlib::core::Tensor<float> squeezed_tensor = mlib::core::squeeze(unsqueezed_tensor, 0); // Remove the dim at axis 0=
// Tensor(shape: {2, 3}, strides: {3, 1}, total_size: 6, data: [
//     [1, 2, 3]
//     [4, 5, 6]
// ])

// Element-wise addition
mlib::core::Tensor<float> C = A + B;
// Tensor(shape: {2, 3}, strides: {3, 1}, total_size: 6, data: [
//     [8, 10, 12]
//     [14, 16, 18]
// ])

// Element-wise logical NOT
mlib::core::Tensor<bool> bool_tensor_a = mlib::core::Tensor<bool>({2,2}, {true, false, true, false});
mlib::core::Tensor<bool> not_a = !bool_tensor_a;
// Tensor(shape: {2, 2}, strides: {2, 1}, total_size: 4, data: [
//   [false, true]
//   [false, true]
// ])

// Tensor-scalar multiplication
mlib::core::Tensor<float> D = A * 2.0f;
// Tensor(shape: {2, 3}, strides: {3, 1}, total_size: 6, data: [
//     [2, 4, 6]
//     [8, 10, 12]
// ])

// Comprehensive Matrix multiplication (2D @ 2D, 1D @ 2D, 2D @ 1D, 1D @ 1D)
mlib::core::Tensor<float> M1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}); // [[1,2],[3,4]]
mlib::core::Tensor<float> M2({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}); // [[5,6],[7,8]]
mlib::core::Tensor<float> P_2D_2D = mlib::core::matmul(M1, M2);
// Tensor(shape: {2, 2}, strides: {2, 1}, total_size: 4, data: [
//     [19, 22]
//     [43, 50]
// ])

mlib::core::Tensor<float> V_1D({3}, {1.0f, 2.0f, 3.0f});
mlib::core::Tensor<float> M_2D_extended({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
mlib::core::Tensor<float> P_1D_2D = mlib::core::matmul(V_1D, M_2D_extended);
// Tensor(shape: {2}, strides: {1}, total_size: 2, data: [22, 28])

mlib::core::Tensor<float> V_2D({3}, {4.0f, 5.0f, 6.0f});
mlib::core::Tensor<float> P_1D_1D = mlib::core::matmul(V_1D, V_2D);
// Tensor(shape: {}, strides: {}, total_size: 1, data: [32])

// Sum reduction along an axis
mlib::core::Tensor<float> axis_sum = mlib::core::sum(A, 1); // Sums along columns for [[1,2,3],[4,5,6]] -> [6.0, 15.0]
// Tensor(shape: {2}, strides: {1}, total_size: 2, data: [6, 15])

// Element-wise comparison
mlib::core::Tensor<bool> mask = A > 3.5f;
// Tensor(shape: {2, 3}, strides: {3, 1}, total_size: 6, data: [
//     [false, false, false]
//     [true, true, true]
// ])
```

## Future Development

- Advanced indexing and slicing for tensors (creating views).
- More complex linear algebra (e.g., QR decomposition, SVD) is not yet available.
- Automatic differentiation engine (autograd) implementation.
- Basic neural network layers and activation functions.
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

