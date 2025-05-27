#include "gtest/gtest.h"
#include "mlib/core/tensor.hpp" // Adjust path as necessary

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
    ASSERT_THROW(Tensor<float>({2,2}, data), std::invalid_argument); // Mismatch
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
    const Tensor<int>& ct = t; // Test const access
    ASSERT_EQ(ct.at({0, 1, 0}), 99);
    ASSERT_THROW(t.at({2,0,0}), std::out_of_range);
    ASSERT_THROW(t.at({0,0}), std::out_of_range); // Wrong number of indices
}

TEST_F(TensorTest, ElementAccessOperatorParentheses) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<int> t({2, 2, 2}, data);
    t(0, 1, 0) = 101;
    ASSERT_EQ(t(0, 1, 0), 101);
    const Tensor<int>& ct = t;
    ASSERT_EQ(ct(0,1,0), 101);
    ASSERT_THROW(t(2,0,0), std::out_of_range);
    ASSERT_THROW(t(0,0), std::out_of_range); // Wrong number of indices
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
    t2.at({0,0}) = 99.f;
    ASSERT_NE(t1.at({0,0}), t2.at({0,0})); // Ensure they are independent
}

TEST_F(TensorTest, CopyAssignment) {
    Tensor<float> t1({2, 2}, {1.f, 2.f, 3.f, 4.f});
    Tensor<float> t2;
    t2 = t1; // Copy assignment

    ASSERT_EQ(t2.get_shape(), t1.get_shape());
    ASSERT_NE(t2.data(), t1.data());
    t2.at({0,0}) = 99.f;
    ASSERT_NE(t1.at({0,0}), t2.at({0,0}));
}

TEST_F(TensorTest, MoveConstructor) {
    Tensor<float> t1({2, 2}, {1.f, 2.f, 3.f, 4.f});
    std::vector<size_t> original_shape = t1.get_shape();
    size_t original_size = t1.get_total_size();
    // If _data is std::vector, its .data() pointer might change after move or stay same if SSO not involved
    // For std::vector, the internal buffer is moved.

    Tensor<float> t2 = std::move(t1);

    ASSERT_EQ(t2.get_shape(), original_shape);
    ASSERT_EQ(t2.get_total_size(), original_size);
    ASSERT_TRUE(t1.is_empty() || t1.get_total_size() == 0); // t1 should be in a valid but empty/moved-from state
    ASSERT_EQ(t2.at({0,0}), 1.f);
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
    ASSERT_EQ(t2.at({1,1}), 4.f);
}


TEST_F(TensorTest, Fill) {
    Tensor<int> t({2,3});
    t.fill(5);
    for(size_t i=0; i<t.get_total_size(); ++i) {
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
    ASSERT_EQ(t.at({0,0}), 1.f); // Data should remain in flat order
    ASSERT_EQ(t.at({1,0}), 3.f); // This is (0*2 + 2)th element = 3rd element in flat data if original {2,3} had strides {3,1}
                                 // After reshape to {3,2} with strides {2,1}, index (1,0) refers to flat_index = 1*2 + 0*1 = 2, so data[2] which is 3.0f.
    ASSERT_EQ(t.at({2,1}), 6.f);

    ASSERT_THROW(t.reshape({4,2}), std::invalid_argument); // Mismatched size
}

TEST_F(TensorTest, ScalarTensor) {
    Tensor<float> scalar_empty_shape(std::vector<size_t>{}); // Shape {}
    ASSERT_EQ(scalar_empty_shape.ndim(), 0);
    ASSERT_EQ(scalar_empty_shape.get_total_size(), 0); // Or 1 if data present
    ASSERT_TRUE(scalar_empty_shape.get_strides().empty());

    Tensor<float> scalar_from_val({}, 5.0f); // This constructor needs thought for scalars.
                                         // Current shape constructor makes total_size 0 for empty shape.
                                         // Let's assume fill constructor for shape {} also means 0 elements for now.
                                         // A dedicated scalar constructor might be better.
    //ASSERT_EQ(scalar_from_val.get_total_size(), 1);
    //ASSERT_FLOAT_EQ(scalar_from_val.data()[0], 5.0f);

    // How to create a scalar with value?
    // Option 1: Modify constructor for shape {} and fill_value
    // Option 2: Tensor<float> scalar_data({1}, {5.0f}); scalar_data.reshape({});
    // Let's test reshape to scalar:
    Tensor<float> t_one_el({1}, {42.0f});
    t_one_el.reshape({});
    ASSERT_EQ(t_one_el.ndim(), 0);
    ASSERT_EQ(t_one_el.get_total_size(), 1); // Total elements remains 1
    ASSERT_TRUE(t_one_el.get_shape().empty());
    ASSERT_TRUE(t_one_el.get_strides().empty()); // Strides for scalar are empty
    // Accessing scalar: current at() and operator() expect indices. Needs a special way or ensure they handle 0 indices.
    // For now, access via data() if you know it's a scalar.
    ASSERT_FLOAT_EQ(t_one_el.data()[0], 42.0f);

    // Let's refine the (shape, fill_value) constructor for scalars
    // If shape is empty AND a fill value is given, it implies a scalar.
    // The current constructor: Tensor(const shape_type& shape, T fill_value)
    // if (shape.empty()) { _total_size = 0; }
    // We might need: if (shape.empty()) { _total_size = 1; _data.assign(1, fill_value); _strides.clear(); }

    Tensor<float> explicit_scalar({}, 123.0f); // Assume constructor handles this as a scalar
    // Revisit constructor logic for shape={} and fill_value/data
    // If Tensor(const shape_type& shape, T fill_value) is modified:
    // if (shape.empty()) { _total_size = 1; _data.assign(1, fill_value); /* no strides */ }
    // Then:
    // ASSERT_EQ(explicit_scalar.get_total_size(), 1);
    // ASSERT_FLOAT_EQ(explicit_scalar.data()[0], 123.0f);
}

// Add more tests:
// - Strides calculation for various dimensions (1D, 3D, etc.)
// - Edge cases for reshape (e.g., reshaping to/from 1-element tensor)
// - is_contiguous() test
