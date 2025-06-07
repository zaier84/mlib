#include "mlib/core/tensor.hpp" // Adjust path as necessary
#include "gtest/gtest.h"
#include <vector>

using namespace mlib::core;

// Test fixture for Tensor tests if needed, but simple tests can go without
class TensorTest : public ::testing::Test {
protected:
  // You can put setup code here if needed by multiple tests
};

TEST_F(TensorTest, DefaultConstructor) {
  Tensor<float> t;
  ASSERT_TRUE(t.is_empty());
  ASSERT_EQ(t.ndim(), 0);
  ASSERT_EQ(t.get_total_size(), 0);
  ASSERT_TRUE(t.get_shape().empty());
  ASSERT_TRUE(t.get_strides().empty());
}

TEST_F(TensorTest, ShapeConstructor) {
  Tensor<int> t({2, 3});
  ASSERT_FALSE(t.is_empty());
  ASSERT_EQ(t.ndim(), 2);
  ASSERT_EQ(t.get_total_size(), 6);
  ASSERT_EQ(t.get_shape().size(), 2);
  ASSERT_EQ(t.get_shape()[0], 2);
  ASSERT_EQ(t.get_shape()[1], 3);
  // Check strides for row-major: {3, 1} for shape {2,3}
  ASSERT_EQ(t.get_strides().size(), 2);
  ASSERT_EQ(t.get_strides()[0], 3);
  ASSERT_EQ(t.get_strides()[1], 1);
  // Check data is zero-initialized
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      ASSERT_EQ(t.at({i, j}), 0);
    }
  }
}

TEST_F(TensorTest, ShapeAndDataConstructor) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t({2, 3}, data);
  ASSERT_EQ(t.get_total_size(), 6);
  ASSERT_EQ(t.at({0, 0}), 1.0f);
  ASSERT_EQ(t.at({0, 1}), 2.0f);
  ASSERT_EQ(t.at({1, 2}), 6.0f);
  ASSERT_THROW(Tensor<float>({2, 2}, data), std::invalid_argument); // Mismatch
}

TEST_F(TensorTest, ShapeAndFillConstructor) {
  Tensor<double> t({2, 2, 2}, 7.5);
  ASSERT_EQ(t.get_total_size(), 8);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        ASSERT_DOUBLE_EQ(t.at({i, j, k}), 7.5);
      }
    }
  }
}

TEST_F(TensorTest, ElementAccessAt) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor<int> t({2, 2, 2}, data);
  t.at({0, 1, 0}) = 99;
  ASSERT_EQ(t.at({0, 1, 0}), 99);
  const Tensor<int> &ct = t; // Test const access
  ASSERT_EQ(ct.at({0, 1, 0}), 99);
  ASSERT_THROW(t.at({2, 0, 0}), std::out_of_range);
  ASSERT_THROW(t.at({0, 0}), std::out_of_range); // Wrong number of indices
}

TEST_F(TensorTest, ElementAccessOperatorParentheses) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor<int> t({2, 2, 2}, data);
  t(0, 1, 0) = 101;
  ASSERT_EQ(t(0, 1, 0), 101);
  const Tensor<int> &ct = t;
  ASSERT_EQ(ct(0, 1, 0), 101);
  ASSERT_THROW(t(2, 0, 0), std::out_of_range);
  ASSERT_THROW(t(0, 0), std::out_of_range); // Wrong number of indices
}

TEST_F(TensorTest, CopyConstructor) {
  Tensor<float> t1({2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor<float> t2 = t1; // Copy constructor

  ASSERT_EQ(t2.get_shape(), t1.get_shape());
  ASSERT_EQ(t2.get_total_size(), t1.get_total_size());
  ASSERT_NE(t2.data(), t1.data()); // Should be a deep copy for std::vector
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      ASSERT_EQ(t2.at({i, j}), t1.at({i, j}));
    }
  }
  t2.at({0, 0}) = 99.f;
  ASSERT_NE(t1.at({0, 0}), t2.at({0, 0})); // Ensure they are independent
}

TEST_F(TensorTest, CopyAssignment) {
  Tensor<float> t1({2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor<float> t2;
  t2 = t1; // Copy assignment

  ASSERT_EQ(t2.get_shape(), t1.get_shape());
  ASSERT_NE(t2.data(), t1.data());
  t2.at({0, 0}) = 99.f;
  ASSERT_NE(t1.at({0, 0}), t2.at({0, 0}));
}

TEST_F(TensorTest, MoveConstructor) {
  Tensor<float> t1({2, 2}, {1.f, 2.f, 3.f, 4.f});
  std::vector<size_t> original_shape = t1.get_shape();
  size_t original_size = t1.get_total_size();
  // If _data is std::vector, its .data() pointer might change after move or
  // stay same if SSO not involved For std::vector, the internal buffer is
  // moved.

  Tensor<float> t2 = std::move(t1);

  ASSERT_EQ(t2.get_shape(), original_shape);
  ASSERT_EQ(t2.get_total_size(), original_size);
  ASSERT_TRUE(t1.is_empty() ||
              t1.get_total_size() ==
                  0); // t1 should be in a valid but empty/moved-from state
  ASSERT_EQ(t2.at({0, 0}), 1.f);
}

TEST_F(TensorTest, MoveAssignment) {
  Tensor<float> t1({2, 2}, {1.f, 2.f, 3.f, 4.f});
  std::vector<size_t> original_shape = t1.get_shape();
  size_t original_size = t1.get_total_size();

  Tensor<float> t2;
  t2 = std::move(t1);

  ASSERT_EQ(t2.get_shape(), original_shape);
  ASSERT_EQ(t2.get_total_size(), original_size);
  ASSERT_TRUE(t1.is_empty() || t1.get_total_size() == 0);
  ASSERT_EQ(t2.at({1, 1}), 4.f);
}

TEST_F(TensorTest, Fill) {
  Tensor<int> t({2, 3});
  t.fill(5);
  for (size_t i = 0; i < t.get_total_size(); ++i) {
    ASSERT_EQ(t.data()[i], 5);
  }
}

TEST_F(TensorTest, Reshape) {
  Tensor<float> t({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  t.reshape({3, 2});
  ASSERT_EQ(t.ndim(), 2);
  ASSERT_EQ(t.get_shape()[0], 3);
  ASSERT_EQ(t.get_shape()[1], 2);
  ASSERT_EQ(t.get_strides()[0], 2); // Strides for {3,2} are {2,1}
  ASSERT_EQ(t.get_strides()[1], 1);
  ASSERT_EQ(t.at({0, 0}), 1.f); // Data should remain in flat order
  ASSERT_EQ(t.at({1, 0}),
            3.f); // This is (0*2 + 2)th element = 3rd element in flat data if
                  // original {2,3} had strides {3,1} After reshape to {3,2}
                  // with strides {2,1}, index (1,0) refers to flat_index = 1*2
                  // + 0*1 = 2, so data[2] which is 3.0f.
  ASSERT_EQ(t.at({2, 1}), 6.f);

  ASSERT_THROW(t.reshape({4, 2}), std::invalid_argument); // Mismatched size
}

TEST_F(TensorTest, ScalarTensor) {
  // Test default constructor (empty, not scalar - still produces 0-size, 0-dim)
  Tensor<float> t_default;
  ASSERT_TRUE(t_default.is_empty());
  ASSERT_FALSE(t_default.is_scalar());
  ASSERT_EQ(t_default.ndim(), 0);
  ASSERT_EQ(t_default.get_total_size(), 0); // Correctly 0
  ASSERT_TRUE(t_default.get_strides().empty());


  // Test Tensor({}) (empty shape constructor) -> NOW PROPER SCALAR
  Tensor<float> scalar_empty_shape(std::vector<size_t>({})); // Constructor `Tensor(const shape_type&)`
  ASSERT_FALSE(scalar_empty_shape.is_empty()); // Is not empty now
  ASSERT_TRUE(scalar_empty_shape.is_scalar()); // Is a scalar now
  ASSERT_EQ(scalar_empty_shape.ndim(), 0);
  ASSERT_EQ(scalar_empty_shape.get_total_size(), 1); // <--- Corrected expectation (now 1)
  ASSERT_TRUE(scalar_empty_shape.get_strides().empty());
  // Should be default-initialized to 0 for numeric types.
  ASSERT_EQ(scalar_empty_shape(), 0.0f); // Access it. (Assumes it's scalar value, and ndim=0)

  // ... (The rest of your TensorTest.ScalarTensor should largely be okay,
  //      as they test other constructors or reshape. Just make sure no other old
  //      `get_total_size() == 0` for empty shape expectations.)
  // The 'Tensor({}, 5.0f)' already tests scalar_fill:
  Tensor<float> scalar_fill({}, 5.0f);
  ASSERT_TRUE(scalar_fill.is_scalar());
  ASSERT_EQ(scalar_fill.get_total_size(), 1);
  ASSERT_FLOAT_EQ(scalar_fill(), 5.0f);

  // The 'Tensor({}, std::vector<float>{42.0f})' already tests scalar_data:
  Tensor<float> scalar_data({}, std::vector<float>{42.0f});
  ASSERT_TRUE(scalar_data.is_scalar());
  ASSERT_EQ(scalar_data.get_total_size(), 1);
  ASSERT_FLOAT_EQ(scalar_data(), 42.0f);

  // When testing reshape to scalar from (1,)
  Tensor<float> t_one_el({1}, {42.0f});
  t_one_el.reshape({}); // Reshape to scalar (empty shape)
  ASSERT_TRUE(t_one_el.is_scalar());
  ASSERT_EQ(t_one_el.get_total_size(), 1);
  ASSERT_EQ(t_one_el.ndim(), 0);
  ASSERT_FLOAT_EQ(t_one_el(), 42.0f); // Access it as scalar
}
// Test `is_contiguous`
TEST_F(TensorTest, IsContiguous) {
    Tensor<int> t_empty;
    ASSERT_TRUE(t_empty.is_contiguous()); // Empty is contiguous

    Tensor<int> t_scalar({}, 5);
    ASSERT_TRUE(t_scalar.is_contiguous()); // Scalar is contiguous

    Tensor<int> t_1d({5}); // {5}, strides {1}
    ASSERT_TRUE(t_1d.is_contiguous());

    Tensor<int> t_2d({2,3}); // shape {2,3}, strides {3,1}
    ASSERT_TRUE(t_2d.is_contiguous());
    ASSERT_EQ(t_2d.get_strides()[0], 3);
    ASSERT_EQ(t_2d.get_strides()[1], 1);

    Tensor<int> t_3d({2,3,4}); // shape {2,3,4}, strides {12,4,1}
    ASSERT_TRUE(t_3d.is_contiguous());
    ASSERT_EQ(t_3d.get_strides()[0], 12);
    ASSERT_EQ(t_3d.get_strides()[1], 4);
    ASSERT_EQ(t_3d.get_strides()[2], 1);

    // Non-contiguous case (e.g., after a transpose or certain slicing - not implemented yet)
    // For now, all our tensors are contiguous by default construction or reshape.
    // If you implement operations that can create non-contiguous views, add tests here.
}

// Test 0-sized dimensions
TEST_F(TensorTest, ZeroSizedDimensions) {
    Tensor<int> t_zero_dim({2, 0, 3});
    ASSERT_TRUE(t_zero_dim.is_empty());
    ASSERT_EQ(t_zero_dim.get_total_size(), 0);
    ASSERT_EQ(t_zero_dim.ndim(), 3);
    ASSERT_EQ(t_zero_dim.get_shape()[1], 0);
    // Strides for {2,0,3}: if stride of 0-dim is 0*3=0, then {0,0,1}
    // If stride of 0-dim is X (next_stride * next_dim_size) -> 1*3=3 -> {3,3,1} (NumPy like)
    // Current `calculate_strides`: _strides[2]=1, _strides[1]=_strides[2]*_shape[2] = 1*3=3, _strides[0]=_strides[1]*_shape[1]=3*0=0
    // So strides would be {0,3,1}
    ASSERT_EQ(t_zero_dim.get_strides()[0], 0);
    ASSERT_EQ(t_zero_dim.get_strides()[1], 3);
    ASSERT_EQ(t_zero_dim.get_strides()[2], 1);

    ASSERT_THROW(t_zero_dim.at({0,0,0}), std::out_of_range); // Cannot access elements
    ASSERT_THROW(t_zero_dim(0,0,0), std::out_of_range);

    // Reshape a 0-element tensor
    t_zero_dim.reshape({0, 5});
    ASSERT_EQ(t_zero_dim.get_total_size(), 0);
    ASSERT_EQ(t_zero_dim.get_shape()[0], 0);
    ASSERT_EQ(t_zero_dim.get_shape()[1], 5);

    ASSERT_THROW(t_zero_dim.reshape({2,2}), std::invalid_argument); // 0 elements to 4 elements

    Tensor<int> t_non_empty({2,3});
    ASSERT_THROW(t_non_empty.reshape({2,0,3}), std::invalid_argument); // 6 elements to 0 elements
}

// Add more tests:
// - Strides calculation for various dimensions (1D, 3D, etc.)
// - Edge cases for reshape (e.g., reshaping to/from 1-element tensor)
// - is_contiguous() test
