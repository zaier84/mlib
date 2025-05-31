#include "mlib/core/tensor.hpp"
#include "mlib/core/operations.hpp"
#include "mlib/core/exceptions.hpp"
#include "gtest/gtest.h"

using namespace mlib::core;

class TensorOperationsTest : public ::testing::Test {
protected:
    Tensor<int> t1_2x2 = Tensor<int>({2,2}, {1,2,3,4});
    Tensor<int> t2_2x2 = Tensor<int>({2,2}, {5,6,7,8});
    Tensor<int> t_2x3 = Tensor<int>({2,3}, {1,2,3,4,5,6});

    Tensor<float> tf1_2x2 = Tensor<float>({2,2}, {1.f,2.f,3.f,4.f});
    Tensor<float> tf2_2x2 = Tensor<float>({2,2}, {0.5f,1.5f,2.5f,3.5f});

    Tensor<int> scalar1 = Tensor<int>({}, 10);
    Tensor<int> scalar2 = Tensor<int>({}, 5);

    Tensor<int> empty_203 = Tensor<int>({2,0,3}); // Total size 0
};


// --- ADDITION TESTS ---
TEST_F(TensorOperationsTest, AddBasic) {
    Tensor<int> result = add(t1_2x2, t2_2x2);
    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,2}));
    ASSERT_EQ(result(0,0), 1 + 5);
    ASSERT_EQ(result(0,1), 2 + 6);
    ASSERT_EQ(result(1,0), 3 + 7);
    ASSERT_EQ(result(1,1), 4 + 8);

    Tensor<float> result_f = tf1_2x2 + tf2_2x2; // Test operator+
    ASSERT_EQ(result_f.get_shape(), Tensor<float>::shape_type({2,2}));
    ASSERT_FLOAT_EQ(result_f(0,0), 1.f + 0.5f);
    ASSERT_FLOAT_EQ(result_f(0,1), 2.f + 1.5f);
    ASSERT_FLOAT_EQ(result_f(1,0), 3.f + 2.5f);
    ASSERT_FLOAT_EQ(result_f(1,1), 4.f + 3.5f);
}

TEST_F(TensorOperationsTest, AddShapeMismatch) {
    ASSERT_THROW(add(t1_2x2, t_2x3), ShapeMismatchError);
    ASSERT_THROW(t1_2x2 + t_2x3, ShapeMismatchError); // Test operator+
}

TEST_F(TensorOperationsTest, AddScalars) {
    Tensor<int> result = add(scalar1, scalar2);
    ASSERT_TRUE(result.is_scalar());
    ASSERT_EQ(result.get_total_size(), 1);
    ASSERT_EQ(result(), 10 + 5);

    Tensor<int> s_result = scalar1 + scalar2; // Test operator+
    ASSERT_TRUE(s_result.is_scalar());
    ASSERT_EQ(s_result(), 15);
}

TEST_F(TensorOperationsTest, AddZeroSizedTensors) {
    Tensor<int> empty1_203({2,0,3});
    Tensor<int> empty2_203({2,0,3});
    Tensor<int> result = add(empty1_203, empty2_203);

    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,0,3}));
    ASSERT_EQ(result.get_total_size(), 0);
    ASSERT_TRUE(result.is_empty());

    Tensor<int> t_non_empty({2,2},1);
    ASSERT_THROW(add(empty1_203, t_non_empty), ShapeMismatchError);
}

// --- UNARY PLUS (IDENTITY) TESTS ---
TEST_F(TensorOperationsTest, UnaryPlusBasic) {
    // t1_2x2 = {1,2,3,4}
    Tensor<int> result = +t1_2x2;
    ASSERT_EQ(result.get_shape(), t1_2x2.get_shape());
    ASSERT_EQ(result.get_total_size(), t1_2x2.get_total_size());
    ASSERT_EQ(result.ndim(), t1_2x2.ndim());
    // Ensure it's a copy, not the same object in memory (data pointers should differ)
    ASSERT_NE(result.data(), t1_2x2.data());

    // Check values
    ASSERT_EQ(result(0,0), 1);
    ASSERT_EQ(result(0,1), 2);
    ASSERT_EQ(result(1,0), 3);
    ASSERT_EQ(result(1,1), 4);

    // Check that original is unchanged
    ASSERT_EQ(t1_2x2(0,0), 1);
}

TEST_F(TensorOperationsTest, UnaryPlusScalar) {
    // scalar1 = Tensor<int>({}, 10);
    Tensor<int> result = +scalar1;
    ASSERT_TRUE(result.is_scalar());
    ASSERT_EQ(result.get_shape(), scalar1.get_shape());
    ASSERT_EQ(result.get_total_size(), 1);
    ASSERT_NE(result.data(), scalar1.data()); // Should be a copy
    ASSERT_EQ(result(), 10);

    // Check original
    ASSERT_EQ(scalar1(), 10);
}

TEST_F(TensorOperationsTest, UnaryPlusEmptyTensor) {
    // empty_203 = Tensor<int>({2,0,3}); // Total size 0
    Tensor<int> result = +empty_203;
    ASSERT_EQ(result.get_shape(), empty_203.get_shape());
    ASSERT_EQ(result.get_total_size(), 0);
    ASSERT_TRUE(result.is_empty());
    // For empty tensors, data() might return nullptr or a valid pointer to no data.
    // If both data() can be non-null (e.g. std::vector::data() on empty vector),
    // they might be different or same depending on small buffer optimization or allocator.
    // The key is that the state (shape, size) is copied.
    // If total_size is 0, _data is likely empty, so .data() comparison is tricky.
    // If default tensor (0 dim, 0 size):
    Tensor<int> default_empty;
    Tensor<int> result_default = +default_empty;
    ASSERT_TRUE(result_default.is_empty());
    ASSERT_FALSE(result_default.is_scalar());
    ASSERT_EQ(result_default.ndim(), 0);
    ASSERT_EQ(result_default.get_total_size(), 0);

}

// --- SUBTRACTION TESTS ---
TEST_F(TensorOperationsTest, SubtractBasic) {
    // t1_2x2 = {1,2,3,4}, t2_2x2 = {5,6,7,8}
    Tensor<int> result = subtract(t1_2x2, t2_2x2);
    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,2}));
    ASSERT_EQ(result(0,0), 1 - 5); // -4
    ASSERT_EQ(result(0,1), 2 - 6); // -4
    ASSERT_EQ(result(1,0), 3 - 7); // -4
    ASSERT_EQ(result(1,1), 4 - 8); // -4

    Tensor<float> result_f = tf1_2x2 - tf2_2x2; // Test operator-
    // tf1_2x2 = {1.f,2.f,3.f,4.f}, tf2_2x2 = {0.5f,1.5f,2.5f,3.5f}
    ASSERT_EQ(result_f.get_shape(), Tensor<float>::shape_type({2,2}));
    ASSERT_FLOAT_EQ(result_f(0,0), 1.f - 0.5f); // 0.5f
    ASSERT_FLOAT_EQ(result_f(0,1), 2.f - 1.5f); // 0.5f
    ASSERT_FLOAT_EQ(result_f(1,0), 3.f - 2.5f); // 0.5f
    ASSERT_FLOAT_EQ(result_f(1,1), 4.f - 3.5f); // 0.5f
}

TEST_F(TensorOperationsTest, SubtractShapeMismatch) {
    ASSERT_THROW(subtract(t1_2x2, t_2x3), ShapeMismatchError);
    ASSERT_THROW(t1_2x2 - t_2x3, ShapeMismatchError);
}

TEST_F(TensorOperationsTest, SubtractScalars) {
    // scalar1 = 10, scalar2 = 5
    Tensor<int> result = subtract(scalar1, scalar2);
    ASSERT_TRUE(result.is_scalar());
    ASSERT_EQ(result.get_total_size(), 1);
    ASSERT_EQ(result(), 10 - 5); // 5

    Tensor<int> s_result = scalar1 - scalar2;
    ASSERT_TRUE(s_result.is_scalar());
    ASSERT_EQ(s_result(), 5);
}

TEST_F(TensorOperationsTest, SubtractZeroSizedTensors) {
    Tensor<int> empty1_203({2,0,3});
    Tensor<int> empty2_203({2,0,3});
    Tensor<int> result = subtract(empty1_203, empty2_203);

    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,0,3}));
    ASSERT_EQ(result.get_total_size(), 0);
    ASSERT_TRUE(result.is_empty());
}


// --- UNARY NEGATION TESTS ---
TEST_F(TensorOperationsTest, NegateBasic) {
    // t1_2x2 = {1,2,3,4}
    Tensor<int> result = negate(t1_2x2);
    ASSERT_EQ(result.get_shape(), t1_2x2.get_shape());
    ASSERT_EQ(result(0,0), -1);
    ASSERT_EQ(result(0,1), -2);
    ASSERT_EQ(result(1,0), -3);
    ASSERT_EQ(result(1,1), -4);

    Tensor<float> result_f = -tf1_2x2; // Test operator- (unary)
    // tf1_2x2 = {1.f,2.f,3.f,4.f}
    ASSERT_EQ(result_f.get_shape(), tf1_2x2.get_shape());
    ASSERT_FLOAT_EQ(result_f(0,0), -1.f);
    ASSERT_FLOAT_EQ(result_f(0,1), -2.f);
    ASSERT_FLOAT_EQ(result_f(1,0), -3.f);
    ASSERT_FLOAT_EQ(result_f(1,1), -4.f);
}

TEST_F(TensorOperationsTest, NegateScalar) {
    // scalar1 = 10
    Tensor<int> result = negate(scalar1);
    ASSERT_TRUE(result.is_scalar());
    ASSERT_EQ(result(), -10);

    Tensor<int> op_result = -scalar1;
    ASSERT_TRUE(op_result.is_scalar());
    ASSERT_EQ(op_result(), -10);
}

TEST_F(TensorOperationsTest, NegateZeroSizedTensor) {
    Tensor<int> empty_t({0, 5});
    Tensor<int> result = negate(empty_t);
    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({0,5}));
    ASSERT_EQ(result.get_total_size(), 0);
    ASSERT_TRUE(result.is_empty());
}

// Optional: Test negate with unsigned type to observe behavior
TEST_F(TensorOperationsTest, NegateUnsigned) {
    Tensor<unsigned int> t_unsigned({2}, {1U, 2U});
    
    // The static_assert in negate(a) related to signed/floating point types
    // is primarily a developer warning. It doesn't stop compilation for unsigned
    // if std::negate<unsigned int> itself is well-defined (which it is).
    // So the test proceeds to check the runtime behavior.
    
    Tensor<unsigned int> result = negate(t_unsigned); 
                                     
    ASSERT_EQ(result.get_shape(), t_unsigned.get_shape());

    // Check against the original value - should be different
    ASSERT_NE(result(0), 1U);
    ASSERT_NE(result(1), 2U);

    // Check the defined behavior of 0U - x for unsigned types
    // 0U - 1U results in UINT_MAX
    // 0U - 2U results in UINT_MAX - 1
    ASSERT_EQ(result(0), 0U - 1U); 
    ASSERT_EQ(result(1), 0U - 2U);

    // This shows that `(unsigned int)-1` often results in UINT_MAX
    // which is the same as `0U - 1U`. So the original ASSERT_NE was failing as expected.
    // We can keep it if we want to explicitly state this equivalence by making it an ASSERT_EQ
    ASSERT_EQ(result(0), static_cast<unsigned int>(-1)); // This will now pass and confirms the equivalence.

    // Operator overload test
    Tensor<unsigned int> op_result = -t_unsigned;
    ASSERT_EQ(op_result(0), 0U - 1U); 
    ASSERT_EQ(op_result(1), 0U - 2U);}

// Placeholder for future scalar broadcast tests
// TEST_F(TensorOperationsTest, AddScalarBroadcast) {
//     Tensor<int> mat = Tensor<int>({2,2}, {1,2,3,4});
//     int scalar_val = 10;
//     Tensor<int> result = add(mat, scalar_val); // This needs add(Tensor, scalar) overload
//     ASSERT_EQ(result(0,0), 1 + 10);
//     // ...
// }

// Add more tests as you implement more operations (subtract, multiply, divide etc.)
