#include "mlib/core/tensor.hpp"
#include "mlib/core/operations.hpp"
#include "mlib/core/exceptions.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>
#include <vector>

using namespace mlib::core;

// --- COMPARISON OPERATION TESTS ---

// Helper function to assert that two Tensor<bool> are element-wise identical
void AssertTensorBoolEqual(const Tensor<bool>& actual, const Tensor<bool>& expected, const std::string& context = "") {
    ASSERT_EQ(actual.get_shape(), expected.get_shape()) << "Shape mismatch in " << context;
    if (actual.get_total_size() == 0) { // Covers scalar default-constructed and empty-shaped
        ASSERT_EQ(expected.get_total_size(), 0) << "Total size mismatch for empty tensors in " << context;
        return;
    }

    // For simplicity in this helper, we'll handle up to 2D.
    // For higher D, you might flatten or write a recursive helper.
    // Or iterate using a flat index if Tensor provides such an accessor or iterator.
    if (actual.ndim() == 0) { // Scalar bool tensor
        ASSERT_EQ(actual(), expected()) << "Scalar value mismatch in " << context;
    } else if (actual.ndim() == 1) {
        for (size_t i = 0; i < actual.get_shape()[0]; ++i) {
            ASSERT_EQ(actual(i), expected(i)) << "Mismatch at (" << i << ") in " << context;
        }
    } else if (actual.ndim() == 2) {
        for (size_t i = 0; i < actual.get_shape()[0]; ++i) {
            for (size_t j = 0; j < actual.get_shape()[1]; ++j) {
                ASSERT_EQ(actual(i,j), expected(i,j)) << "Mismatch at (" << i << "," << j << ") in " << context;
            }
        }
    } else { // Fallback for >2D or general case: iterate flat if possible or mark as not fully tested by this helper.
        // This simplistic flat iteration assumes contiguous underlying bools and a way to access them.
        // If Tensor<bool> doesn't guarantee flat access, this part would need care.
        // For now, let's assume it can be constructed from a flat std::vector<bool> and
        // then elements can be checked via its multi-dim accessors.
        // The loops above are safer.
        ADD_FAILURE() << "AssertTensorBoolEqual currently only implemented thoroughly for 0D, 1D, 2D in " << context;
    }
}

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
    ASSERT_EQ(op_result(1), 0U - 2U);
}


// --- MULTIPLICATION (ELEMENT-WISE) TESTS ---
TEST_F(TensorOperationsTest, MultiplyBasic) {
    // t1_2x2 = {1,2,3,4}, t2_2x2 = {5,6,7,8}
    Tensor<int> result = multiply(t1_2x2, t2_2x2);
    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,2}));
    ASSERT_EQ(result(0,0), 1 * 5); // 5
    ASSERT_EQ(result(0,1), 2 * 6); // 12
    ASSERT_EQ(result(1,0), 3 * 7); // 21
    ASSERT_EQ(result(1,1), 4 * 8); // 32

    // tf1_2x2 = {1.f,2.f,3.f,4.f}, tf2_2x2 = {0.5f,1.5f,2.5f,3.5f}
    Tensor<float> result_f = tf1_2x2 * tf2_2x2; // Test operator*
    ASSERT_EQ(result_f.get_shape(), Tensor<float>::shape_type({2,2}));
    ASSERT_FLOAT_EQ(result_f(0,0), 1.f * 0.5f); // 0.5f
    ASSERT_FLOAT_EQ(result_f(0,1), 2.f * 1.5f); // 3.0f
    ASSERT_FLOAT_EQ(result_f(1,0), 3.f * 2.5f); // 7.5f
    ASSERT_FLOAT_EQ(result_f(1,1), 4.f * 3.5f); // 14.0f
}

TEST_F(TensorOperationsTest, MultiplyShapeMismatch) {
    ASSERT_THROW(multiply(t1_2x2, t_2x3), ShapeMismatchError);
    ASSERT_THROW(t1_2x2 * t_2x3, ShapeMismatchError);
}

TEST_F(TensorOperationsTest, MultiplyScalars) {
    // scalar1 = 10, scalar2 = 5
    Tensor<int> result = multiply(scalar1, scalar2);
    ASSERT_TRUE(result.is_scalar());
    ASSERT_EQ(result.get_total_size(), 1);
    ASSERT_EQ(result(), 10 * 5); // 50

    Tensor<int> s_result = scalar1 * scalar2;
    ASSERT_TRUE(s_result.is_scalar());
    ASSERT_EQ(s_result(), 50);
}

TEST_F(TensorOperationsTest, MultiplyWithZero) {
    Tensor<int> t_zeros({2,2}, 0); // Tensor of zeros
    // t1_2x2 = {1,2,3,4}
    Tensor<int> result = t1_2x2 * t_zeros;
    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,2}));
    ASSERT_EQ(result(0,0), 1 * 0);
    ASSERT_EQ(result(0,1), 2 * 0);
    ASSERT_EQ(result(1,0), 3 * 0);
    ASSERT_EQ(result(1,1), 4 * 0);

    result = t_zeros * t1_2x2; // Commutativity
    ASSERT_EQ(result(0,0), 0 * 1);
    ASSERT_EQ(result(1,1), 0 * 4);
}

TEST_F(TensorOperationsTest, MultiplyZeroSizedTensors) {
    Tensor<int> empty1_203({2,0,3});
    Tensor<int> empty2_203({2,0,3});
    Tensor<int> result = multiply(empty1_203, empty2_203);

    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,0,3}));
    ASSERT_EQ(result.get_total_size(), 0);
    ASSERT_TRUE(result.is_empty());
}


// --- DIVISION (ELEMENT-WISE) TESTS ---
TEST_F(TensorOperationsTest, DivideBasicInteger) {
    Tensor<int> a({2,2}, {10, 21, 35, 44});
    Tensor<int> b({2,2}, {2,  7,  5, 11});
    Tensor<int> result = divide(a, b);
    ASSERT_EQ(result.get_shape(), Tensor<int>::shape_type({2,2}));
    ASSERT_EQ(result(0,0), 10 / 2); // 5
    ASSERT_EQ(result(0,1), 21 / 7); // 3
    ASSERT_EQ(result(1,0), 35 / 5); // 7
    ASSERT_EQ(result(1,1), 44 / 11); // 4

    Tensor<int> op_result = a / b; // Test operator/
    ASSERT_EQ(op_result(0,0), 5);
    ASSERT_EQ(op_result(1,1), 4);
}

TEST_F(TensorOperationsTest, DivideBasicFloat) {
    Tensor<float> a({2,2}, {10.f, 20.f, 30.f, 40.f});
    Tensor<float> b({2,2}, {2.f,  0.5f, 4.f,  5.f});
    Tensor<float> result = divide(a, b);
    ASSERT_EQ(result.get_shape(), Tensor<float>::shape_type({2,2}));
    ASSERT_FLOAT_EQ(result(0,0), 10.f / 2.f);   // 5.0f
    ASSERT_FLOAT_EQ(result(0,1), 20.f / 0.5f); // 40.0f
    ASSERT_FLOAT_EQ(result(1,0), 30.f / 4.f);   // 7.5f
    ASSERT_FLOAT_EQ(result(1,1), 40.f / 5.f);   // 8.0f

    Tensor<float> op_result = a / b; // Test operator/
    ASSERT_FLOAT_EQ(op_result(0,0), 5.0f);
    ASSERT_FLOAT_EQ(op_result(1,1), 8.0f);
}


TEST_F(TensorOperationsTest, DivideShapeMismatch) {
    Tensor<int> a({2,2}, 1);
    Tensor<int> b_wrong_shape({2,3}, 1);
    ASSERT_THROW(divide(a, b_wrong_shape), ShapeMismatchError);
    ASSERT_THROW(a / b_wrong_shape, ShapeMismatchError);
}

TEST_F(TensorOperationsTest, DivideScalarsInteger) {
    Tensor<int> sa({100}); // Use scalar constructor Tensor<int>({}, 100)
    Tensor<int> sb({5});   // Use scalar constructor Tensor<int>({}, 5)
    Tensor<int> s_a_val({},100);
    Tensor<int> s_b_val({},5);

    Tensor<int> result = divide(s_a_val, s_b_val);
    ASSERT_TRUE(result.is_scalar());
    ASSERT_EQ(result(), 100 / 5); // 20

    ASSERT_THROW(divide(s_a_val, Tensor<int>({},0)), std::overflow_error);
}

TEST_F(TensorOperationsTest, DivideScalarsFloat) {
    Tensor<float> sa({},100.f);
    Tensor<float> sb({},4.f);
    Tensor<float> s_zero({},0.f);

    Tensor<float> result = divide(sa, sb);
    ASSERT_TRUE(result.is_scalar());
    ASSERT_FLOAT_EQ(result(), 100.f / 4.f); // 25.f

    Tensor<float> result_div_zero = divide(sa, s_zero); // 100.f / 0.f
    ASSERT_TRUE(result_div_zero.is_scalar());
    ASSERT_TRUE(std::isinf(result_div_zero())); // Check for infinity
    ASSERT_GT(result_div_zero(), 0); // Positive infinity

    Tensor<float> zero_div_zero = divide(s_zero, s_zero); // 0.f / 0.f
    ASSERT_TRUE(zero_div_zero.is_scalar());
    ASSERT_TRUE(std::isnan(zero_div_zero())); // Check for NaN
}

TEST_F(TensorOperationsTest, DivideByZeroInteger) {
    Tensor<int> a({2,2}, {10, 20, 30, 40});
    Tensor<int> b_has_zero({2,2}, {2, 5, 0, 4}); // Zero at (1,0)
    ASSERT_THROW(divide(a, b_has_zero), std::overflow_error);
    ASSERT_THROW(a / b_has_zero, std::overflow_error);

    Tensor<int> first_element_zero_divisor({2,2}, {0,1,1,1});
    ASSERT_THROW(a / first_element_zero_divisor, std::overflow_error);
}

TEST_F(TensorOperationsTest, DivideByZeroFloat) {
    Tensor<float> a({2,2}, {10.f, 20.f, 30.f, 1.f});
    Tensor<float> b_has_zero({2,2}, {2.f, 0.f, 4.f, 0.f}); // Zeros at (0,1) and (1,1)

    Tensor<float> result = divide(a, b_has_zero);
    ASSERT_EQ(result.get_shape(), Tensor<float>::shape_type({2,2}));
    ASSERT_FLOAT_EQ(result(0,0), 10.f / 2.f); // 5.0f
    ASSERT_TRUE(std::isinf(result(0,1)));     // 20.f / 0.f -> inf
    ASSERT_GT(result(0,1), 0);                // Positive infinity
    ASSERT_FLOAT_EQ(result(1,0), 30.f / 4.f); // 7.5f
    ASSERT_TRUE(std::isinf(result(1,1)));     // 1.f / 0.f -> inf
    ASSERT_GT(result(1,1), 0);                // Positive infinity

    Tensor<float> zero_val({}, 0.f);
    Tensor<float> one_val({}, 1.f);
    Tensor<float> neg_one_val({}, -1.f);

    // 0/0 -> NaN
    Tensor<float> nan_res = zero_val / zero_val;
    ASSERT_TRUE(std::isnan(nan_res()));

    // -1/0 -> -inf
    Tensor<float> neg_inf_res = neg_one_val / zero_val;
    ASSERT_TRUE(std::isinf(neg_inf_res()));
    ASSERT_LT(neg_inf_res(), 0); // Negative infinity
}


TEST_F(TensorOperationsTest, DivideZeroSizedTensors) {
    Tensor<int> empty1_203({2,0,3});
    Tensor<int> empty2_203({2,0,3}); // Must also be empty and same shape
    Tensor<int> result_int = divide(empty1_203, empty2_203);

    ASSERT_EQ(result_int.get_shape(), Tensor<int>::shape_type({2,0,3}));
    ASSERT_EQ(result_int.get_total_size(), 0);
    ASSERT_TRUE(result_int.is_empty());

    Tensor<float> empty_f1({2,0,3});
    Tensor<float> empty_f2({2,0,3});
    Tensor<float> result_float = divide(empty_f1, empty_f2);
    ASSERT_EQ(result_float.get_total_size(), 0);
}


TEST_F(TensorOperationsTest, AddTensorScalarTypePromotion) {
    Tensor<float> tf({1,2}, {1.5f, 2.5f});
    int scalar_int = 3;

    // float_tensor + int_scalar -> result should be float tensor
    Tensor<float> result_f_plus_i = tf + scalar_int;
    ASSERT_EQ(result_f_plus_i.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result_f_plus_i(0,0), 1.5f + 3.0f); // 3 converted to 3.0f
    ASSERT_FLOAT_EQ(result_f_plus_i(0,1), 2.5f + 3.0f);

    // int_scalar + float_tensor
    result_f_plus_i = scalar_int + tf;
    ASSERT_EQ(result_f_plus_i.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result_f_plus_i(0,0), 3.0f + 1.5f);
    ASSERT_FLOAT_EQ(result_f_plus_i(0,1), 3.0f + 2.5f);

    Tensor<int> ti({1,2}, {10, 20});
    double scalar_double = 2.5;

    // int_tensor + double_scalar -> result should be int tensor (double truncated)
    Tensor<int> result_i_plus_d = ti + scalar_double;
    ASSERT_EQ(result_i_plus_d.get_shape(), ti.get_shape());
    ASSERT_EQ(result_i_plus_d(0,0), 10 + static_cast<int>(2.5)); // 10 + 2 = 12
    ASSERT_EQ(result_i_plus_d(0,1), 20 + static_cast<int>(2.5)); // 20 + 2 = 22

    // double_scalar + int_tensor
    result_i_plus_d = scalar_double + ti;
    ASSERT_EQ(result_i_plus_d.get_shape(), ti.get_shape());
    ASSERT_EQ(result_i_plus_d(0,0), static_cast<int>(2.5) + 10); // 2 + 10 = 12
    ASSERT_EQ(result_i_plus_d(0,1), static_cast<int>(2.5) + 20); // 2 + 20 = 22
}





// --- SUBTRACTION WITH SCALAR TESTS ---
TEST_F(TensorOperationsTest, SubtractTensorScalar) {
    Tensor<int> local_t1_2x2({2,2}, {10,20,30,40}); // Minuend
    int scalar_subtrahend = 7;

    Tensor<int> result = local_t1_2x2 - scalar_subtrahend;
    ASSERT_EQ(result.get_shape(), local_t1_2x2.get_shape());
    ASSERT_EQ(result(0,0), 10 - scalar_subtrahend); // 3
    ASSERT_EQ(result(0,1), 20 - scalar_subtrahend); // 13
    ASSERT_EQ(result(1,0), 30 - scalar_subtrahend); // 23
    ASSERT_EQ(result(1,1), 40 - scalar_subtrahend); // 33
}

TEST_F(TensorOperationsTest, SubtractScalarTensor) {
    int scalar_minuend = 50;
    Tensor<int> local_t1_2x2_subtrahend({2,2}, {10,20,30,40});

    Tensor<int> result = scalar_minuend - local_t1_2x2_subtrahend;
    ASSERT_EQ(result.get_shape(), local_t1_2x2_subtrahend.get_shape());
    ASSERT_EQ(result(0,0), scalar_minuend - 10); // 40
    ASSERT_EQ(result(0,1), scalar_minuend - 20); // 30
    ASSERT_EQ(result(1,0), scalar_minuend - 30); // 20
    ASSERT_EQ(result(1,1), scalar_minuend - 40); // 10
}

TEST_F(TensorOperationsTest, SubtractScalarTensorToScalarTensor) {
    // scalar1 = Tensor<int>({}, 10); (from fixture, assuming it's still 10)
    Tensor<int> scalar_tensor_minuend({}, 25);
    int scalar_val_subtrahend = 5;

    Tensor<int> result_ts = scalar_tensor_minuend - scalar_val_subtrahend; // Tensor - scalar
    ASSERT_TRUE(result_ts.is_scalar());
    ASSERT_EQ(result_ts(), 25 - 5); // 20

    Tensor<int> scalar_tensor_subtrahend({}, 3);
    int scalar_val_minuend = 10;
    Tensor<int> result_st = scalar_val_minuend - scalar_tensor_subtrahend; // scalar - Tensor
    ASSERT_TRUE(result_st.is_scalar());
    ASSERT_EQ(result_st(), 10 - 3); // 7
}

TEST_F(TensorOperationsTest, SubtractScalarFromEmptyTensor) {
    // empty_203 from fixture
    int scalar_val = 100;
    Tensor<int> result = empty_203 - scalar_val;
    ASSERT_EQ(result.get_shape(), empty_203.get_shape());
    ASSERT_TRUE(result.is_empty());

    result = scalar_val - empty_203;
    ASSERT_EQ(result.get_shape(), empty_203.get_shape());
    ASSERT_TRUE(result.is_empty());
}

TEST_F(TensorOperationsTest, SubtractFloatScalarTypeConversion) {
    Tensor<float> tf({1,2}, {10.5f, 20.5f}); // Minuend
    int scalar_int_subtrahend = 3;           // Subtrahend

    Tensor<float> result_f_minus_i = tf - scalar_int_subtrahend; // float T - int S
    ASSERT_EQ(result_f_minus_i.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result_f_minus_i(0,0), 10.5f - 3.0f); // 7.5f (3 promotes to 3.0f)
    ASSERT_FLOAT_EQ(result_f_minus_i(0,1), 20.5f - 3.0f); // 17.5f

    double scalar_double_minuend = 50.25;
    // tf is {10.5f, 20.5f}           // Subtrahend
    Tensor<float> result_d_minus_f = scalar_double_minuend - tf; // double S - float T. Result is float tensor
    ASSERT_EQ(result_d_minus_f.get_shape(), tf.get_shape());
    // scalar_double_minuend (50.25) is cast to float (50.25f) for result element
    ASSERT_FLOAT_EQ(result_d_minus_f(0,0), static_cast<float>(scalar_double_minuend) - 10.5f); // 50.25f - 10.5f = 39.75f
    ASSERT_FLOAT_EQ(result_d_minus_f(0,1), static_cast<float>(scalar_double_minuend) - 20.5f); // 50.25f - 20.5f = 29.75f
}

TEST_F(TensorOperationsTest, SubtractIntScalarTypeConversion) {
    Tensor<int> ti({1,2}, {10, 20}); // Minuend
    float scalar_float_subtrahend = 3.7f; // Subtrahend

    Tensor<int> result_i_minus_f = ti - scalar_float_subtrahend; // int T - float S
    ASSERT_EQ(result_i_minus_f.get_shape(), ti.get_shape());
    ASSERT_EQ(result_i_minus_f(0,0), 10 - static_cast<int>(3.7f)); // 10 - 3 = 7
    ASSERT_EQ(result_i_minus_f(0,1), 20 - static_cast<int>(3.7f)); // 20 - 3 = 17

    int scalar_int_minuend = 50;
    Tensor<float> tf_subtrahend({1,2}, {2.2f, 8.9f});

    // Here, T (Tensor element type) of result will be float, determined by tf_subtrahend.
    // But we are testing scalar(int) - Tensor(float).
    // The current `subtract(S, Tensor<T>)` assumes the result tensor is `Tensor<T>`.
    // If we wanted S - Tensor<T> to yield Tensor<S> (if S is "wider"), that's a different design.
    // For now, the scalar is cast to T.
    Tensor<float> result_i_minus_tf = static_cast<float>(scalar_int_minuend) - tf_subtrahend; // operator-(S, Tensor<T>) called
    ASSERT_EQ(result_i_minus_tf.get_shape(), tf_subtrahend.get_shape());
    // The scalar_int_minuend (50) is cast to float (50.0f) before subtracting the tensor elements.
    ASSERT_FLOAT_EQ(result_i_minus_tf(0,0), 50.0f - 2.2f); // 47.8f
    ASSERT_FLOAT_EQ(result_i_minus_tf(0,1), 50.0f - 8.9f); // 41.1f
}


// --- MULTIPLICATION WITH SCALAR TESTS ---
TEST_F(TensorOperationsTest, MultiplyTensorScalar) {
    Tensor<int> local_t1_2x2({2,2}, {1,2,3,4});
    int scalar_multiplier = 3;

    Tensor<int> result = local_t1_2x2 * scalar_multiplier;
    ASSERT_EQ(result.get_shape(), local_t1_2x2.get_shape());
    ASSERT_EQ(result(0,0), 1 * scalar_multiplier); // 3
    ASSERT_EQ(result(0,1), 2 * scalar_multiplier); // 6
    ASSERT_EQ(result(1,0), 3 * scalar_multiplier); // 9
    ASSERT_EQ(result(1,1), 4 * scalar_multiplier); // 12

    // Test commutativity with operator S * T
    Tensor<int> result_commutative = scalar_multiplier * local_t1_2x2;
    ASSERT_EQ(result_commutative.get_shape(), local_t1_2x2.get_shape());
    ASSERT_EQ(result_commutative(0,0), scalar_multiplier * 1); // 3
    ASSERT_EQ(result_commutative(1,1), scalar_multiplier * 4); // 12
}

TEST_F(TensorOperationsTest, MultiplyScalarTensorToScalarTensor) {
    Tensor<int> scalar_tensor_factor1({}, 7);
    int scalar_val_factor2 = 6;

    Tensor<int> result_ts = scalar_tensor_factor1 * scalar_val_factor2; // Tensor * scalar
    ASSERT_TRUE(result_ts.is_scalar());
    ASSERT_EQ(result_ts(), 7 * 6); // 42

    result_ts = scalar_val_factor2 * scalar_tensor_factor1; // scalar * Tensor
    ASSERT_TRUE(result_ts.is_scalar());
    ASSERT_EQ(result_ts(), 6 * 7); // 42
}

TEST_F(TensorOperationsTest, MultiplyScalarWithEmptyTensor) {
    // empty_203 from fixture
    int scalar_val = 10;
    Tensor<int> result = empty_203 * scalar_val;
    ASSERT_EQ(result.get_shape(), empty_203.get_shape());
    ASSERT_TRUE(result.is_empty());

    result = scalar_val * empty_203;
    ASSERT_EQ(result.get_shape(), empty_203.get_shape());
    ASSERT_TRUE(result.is_empty());
}

TEST_F(TensorOperationsTest, MultiplyFloatScalarTypeConversion) {
    Tensor<float> tf({1,2}, {1.5f, 2.5f});
    int scalar_int_multiplier = 4;

    Tensor<float> result_f_times_i = tf * scalar_int_multiplier; // float T * int S
    ASSERT_EQ(result_f_times_i.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result_f_times_i(0,0), 1.5f * 4.0f); // 6.0f (4 promotes to 4.0f)
    ASSERT_FLOAT_EQ(result_f_times_i(0,1), 2.5f * 4.0f); // 10.0f

    double scalar_double_multiplier = 0.5;
    Tensor<int> ti({1,2}, {10, 20});

    Tensor<int> result_i_times_d = ti * scalar_double_multiplier; // int T * double S
    ASSERT_EQ(result_i_times_d.get_shape(), ti.get_shape());
    // scalar_double_multiplier (0.5) cast to int (0)
    ASSERT_EQ(result_i_times_d(0,0), 10 * static_cast<int>(0.5)); // 10 * 0 = 0
    ASSERT_EQ(result_i_times_d(0,1), 20 * static_cast<int>(0.5)); // 20 * 0 = 0
}

TEST_F(TensorOperationsTest, MultiplyByZeroScalar) {
    Tensor<int> local_t1_2x2({2,2}, {1,2,3,4});
    int scalar_zero = 0;
    Tensor<int> result = local_t1_2x2 * scalar_zero;
    ASSERT_EQ(result(0,0), 0);
    ASSERT_EQ(result(0,1), 0);
    ASSERT_EQ(result(1,0), 0);
    ASSERT_EQ(result(1,1), 0);

    result = scalar_zero * local_t1_2x2;
    ASSERT_EQ(result(0,0), 0);
    ASSERT_EQ(result(1,1), 0);
}


// --- DIVISION WITH SCALAR TESTS ---
TEST_F(TensorOperationsTest, DivideTensorByScalar) {
    Tensor<int> ti({2,2}, {10, 21, 30, 44});
    int scalar_int_divisor = 2;
    Tensor<int> result_i = ti / scalar_int_divisor;
    ASSERT_EQ(result_i(0,0), 10 / 2); // 5
    ASSERT_EQ(result_i(0,1), 21 / 2); // 10 (integer division)
    ASSERT_EQ(result_i(1,0), 30 / 2); // 15
    ASSERT_EQ(result_i(1,1), 44 / 2); // 22

    Tensor<float> tf({2,2}, {10.f, 21.f, 30.f, 0.f});
    float scalar_float_divisor = 2.0f;
    Tensor<float> result_f = tf / scalar_float_divisor;
    ASSERT_FLOAT_EQ(result_f(0,0), 10.f / 2.f); // 5.0f
    ASSERT_FLOAT_EQ(result_f(0,1), 21.f / 2.f); // 10.5f
    ASSERT_FLOAT_EQ(result_f(1,1), 0.f / 2.f);  // 0.0f

    // Test Tensor<int> / float_scalar
    Tensor<int> ti2({1}, {7});
    float f_div = 2.0f; // When cast to int for division, becomes 2
    Tensor<int> res_i_f = ti2 / f_div; // 7 / static_cast<int>(2.0f) = 7 / 2 = 3
    ASSERT_EQ(res_i_f(0), 3);

    // Test Tensor<float> / int_scalar
    Tensor<float> tf2({1}, {7.0f});
    int i_div = 2; // Promotes to 2.0f
    Tensor<float> res_f_i = tf2 / i_div; // 7.0f / 2.0f = 3.5f
    ASSERT_FLOAT_EQ(res_f_i(0), 3.5f);
}

TEST_F(TensorOperationsTest, DivideScalarByTensor) {
    int scalar_int_dividend = 60;
    Tensor<int> ti_divisor({2,2}, {3, 5, 6, 10});
    Tensor<int> result_i = scalar_int_dividend / ti_divisor;
    ASSERT_EQ(result_i(0,0), 60 / 3);  // 20
    ASSERT_EQ(result_i(0,1), 60 / 5);  // 12
    ASSERT_EQ(result_i(1,0), 60 / 6);  // 10
    ASSERT_EQ(result_i(1,1), 60 / 10); // 6

    float scalar_float_dividend = 10.0f;
    Tensor<float> tf_divisor({2,2}, {2.f, 4.f, 0.5f, 8.f});
    Tensor<float> result_f = scalar_float_dividend / tf_divisor;
    ASSERT_FLOAT_EQ(result_f(0,0), 10.f / 2.f);   // 5.0f
    ASSERT_FLOAT_EQ(result_f(0,1), 10.f / 4.f);   // 2.5f
    ASSERT_FLOAT_EQ(result_f(1,0), 10.f / 0.5f); // 20.0f
    ASSERT_FLOAT_EQ(result_f(1,1), 10.f / 8.f);   // 1.25f
}

TEST_F(TensorOperationsTest, DivideTensorByScalar_ScalarIsZero) {
    Tensor<int> ti({1}, {10});
    int scalar_int_zero = 0;
    ASSERT_THROW(ti / scalar_int_zero, std::overflow_error);

    Tensor<float> tf({1}, {10.f});
    float scalar_float_zero = 0.0f;
    Tensor<float> result_f_div_zero = tf / scalar_float_zero; // 10.0f / 0.0f
    ASSERT_TRUE(std::isinf(result_f_div_zero(0)));
    ASSERT_GT(result_f_div_zero(0), 0); // Positive infinity

    // Test case where T is int but S (scalar) is float 0.0
    ASSERT_THROW(ti / scalar_float_zero, std::overflow_error); // scalar_float_zero cast to int is 0
}

TEST_F(TensorOperationsTest, DivideScalarByTensor_TensorHasZero) {
    int scalar_int_dividend = 10;
    Tensor<int> ti_divisor_has_zero({1,2}, {2, 0});
    ASSERT_THROW(scalar_int_dividend / ti_divisor_has_zero, std::overflow_error);

    float scalar_float_dividend = 10.f;
    Tensor<float> tf_divisor_has_zero({1,3}, {2.f, 0.f, -0.f}); // Test positive and negative zero for floats
    Tensor<float> result_f = scalar_float_dividend / tf_divisor_has_zero;
    ASSERT_FLOAT_EQ(result_f(0,0), 10.f / 2.f); // 5.0f
    ASSERT_TRUE(std::isinf(result_f(0,1)));     // 10.0f / 0.0f -> +inf
    ASSERT_GT(result_f(0,1), 0);
    ASSERT_TRUE(std::isinf(result_f(0,2)));     // 10.0f / -0.0f -> -inf (behavior for -0.0f can vary slightly in representation but usually consistent)
    ASSERT_LT(result_f(0,2), 0);
}


TEST_F(TensorOperationsTest, DivideScalarTensorToScalarTensor) {
    Tensor<int> scalar_tensor_dividend({}, 20);
    int scalar_val_divisor = 4;
    Tensor<int> result_ts = scalar_tensor_dividend / scalar_val_divisor; // Tensor / scalar
    ASSERT_TRUE(result_ts.is_scalar());
    ASSERT_EQ(result_ts(), 20 / 4); // 5

    Tensor<int> scalar_tensor_divisor({}, 3);
    int scalar_val_dividend = 21;
    Tensor<int> result_st = scalar_val_dividend / scalar_tensor_divisor; // scalar / Tensor
    ASSERT_TRUE(result_st.is_scalar());
    ASSERT_EQ(result_st(), 21 / 3); // 7

    // Scalar tensor division by zero
    ASSERT_THROW(scalar_tensor_dividend / 0, std::overflow_error);
    ASSERT_THROW(scalar_val_dividend / Tensor<int>({}, 0), std::overflow_error);

    Tensor<float> scalar_tf_dividend({}, 20.f);
    float scalar_f_zero = 0.f;
    ASSERT_TRUE(std::isinf((scalar_tf_dividend / scalar_f_zero)()));
    ASSERT_TRUE(std::isinf((1.0f / Tensor<float>({},0.f))()));

}

TEST_F(TensorOperationsTest, DivideScalarWithEmptyTensor) {
    int scalar_val = 100;
    Tensor<int> result_ts = empty_203 / scalar_val;
    ASSERT_EQ(result_ts.get_shape(), empty_203.get_shape());
    ASSERT_TRUE(result_ts.is_empty());

    Tensor<int> result_st = scalar_val / empty_203;
    ASSERT_EQ(result_st.get_shape(), empty_203.get_shape());
    ASSERT_TRUE(result_st.is_empty());
}

TEST_F(TensorOperationsTest, Exp) {
    Tensor<float> tf({1,3}, {0.0f, 1.0f, -1.0f});
    Tensor<float> result = mlib::core::exp(tf); // Explicit namespace for clarity

    ASSERT_EQ(result.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result(0,0), std::exp(0.0f)); // 1.0f
    ASSERT_FLOAT_EQ(result(0,1), std::exp(1.0f)); // approx 2.71828
    ASSERT_FLOAT_EQ(result(0,2), std::exp(-1.0f)); // approx 0.36787

    Tensor<double> td_scalar({}, 2.0);
    Tensor<double> result_d_scalar = mlib::core::exp(td_scalar);
    ASSERT_TRUE(result_d_scalar.is_scalar());
    ASSERT_DOUBLE_EQ(result_d_scalar(), std::exp(2.0));

    // Test with empty (0-element but shaped) tensor
    Tensor<float> empty_shaped_tf({2,0,3});
    Tensor<float> result_empty_exp = mlib::core::exp(empty_shaped_tf);
    ASSERT_EQ(result_empty_exp.get_shape(), empty_shaped_tf.get_shape());
    ASSERT_TRUE(result_empty_exp.is_empty());
}

TEST_F(TensorOperationsTest, Log) {
    Tensor<float> tf({1,3}, {1.0f, std::exp(2.0f), 0.1f});
    Tensor<float> result = mlib::core::log(tf);

    ASSERT_EQ(result.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result(0,0), std::log(1.0f));   // 0.0f
    ASSERT_FLOAT_EQ(result(0,1), std::log(std::exp(2.0f))); // 2.0f
    ASSERT_FLOAT_EQ(result(0,2), std::log(0.1f));   // approx -2.30258

    // Test log domain (log(0) and log(-ve))
    Tensor<float> tf_domain({1,2}, {0.0f, -1.0f});
    Tensor<float> result_domain = mlib::core::log(tf_domain);
    ASSERT_TRUE(std::isinf(result_domain(0,0))); // log(0) -> -inf
    ASSERT_LT(result_domain(0,0), 0);           // specifically negative infinity
    ASSERT_TRUE(std::isnan(result_domain(0,1))); // log(-1) -> NaN
}

TEST_F(TensorOperationsTest, Sqrt) {
    Tensor<float> tf({1,3}, {0.0f, 4.0f, 9.0f});
    Tensor<float> result = mlib::core::sqrt(tf);

    ASSERT_EQ(result.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result(0,0), std::sqrt(0.0f)); // 0.0f
    ASSERT_FLOAT_EQ(result(0,1), std::sqrt(4.0f)); // 2.0f
    ASSERT_FLOAT_EQ(result(0,2), std::sqrt(9.0f)); // 3.0f

    // Test sqrt domain (sqrt(-ve))
    Tensor<float> tf_domain({1,1}, {-4.0f});
    Tensor<float> result_domain = mlib::core::sqrt(tf_domain);
    ASSERT_TRUE(std::isnan(result_domain(0,0))); // sqrt(-4.0) -> NaN
}

TEST_F(TensorOperationsTest, Abs) {
    Tensor<float> tf({1,3}, {0.0f, -2.5f, 3.0f});
    Tensor<float> result_f = mlib::core::abs(tf);
    ASSERT_EQ(result_f.get_shape(), tf.get_shape());
    ASSERT_FLOAT_EQ(result_f(0,0), std::fabs(0.0f));   // 0.0f
    ASSERT_FLOAT_EQ(result_f(0,1), std::fabs(-2.5f)); // 2.5f
    ASSERT_FLOAT_EQ(result_f(0,2), std::fabs(3.0f));   // 3.0f

    Tensor<int> ti({1,3}, {0, -5, 7});
    Tensor<int> result_i = mlib::core::abs(ti);
    ASSERT_EQ(result_i.get_shape(), ti.get_shape());
    ASSERT_EQ(result_i(0,0), std::abs(0));  // 0
    ASSERT_EQ(result_i(0,1), std::abs(-5)); // 5
    ASSERT_EQ(result_i(0,2), std::abs(7));  // 7

    Tensor<int> scalar_i_abs({}, -100);
    Tensor<int> result_scalar_i_abs = mlib::core::abs(scalar_i_abs);
    ASSERT_TRUE(result_scalar_i_abs.is_scalar());
    ASSERT_EQ(result_scalar_i_abs(), std::abs(-100));
}



// --- EQUAL (==) TESTS ---
// (You should have these already from my previous response or your implementation)
TEST_F(TensorOperationsTest, EqualTensorTensor) {
    Tensor<int> a({2,2}, {1,2,3,4});
    Tensor<int> b({2,2}, {1,2,3,4});
    Tensor<int> c({2,2}, {1,0,3,5});

    Tensor<bool> expected_ab({2,2}, {true,true,true,true});
    AssertTensorBoolEqual(a == b, expected_ab, "a == b");

    Tensor<bool> expected_ac({2,2}, {true,false,true,false});
    AssertTensorBoolEqual(a == c, expected_ac, "a == c");

    Tensor<int> d_wrong_shape({2,3}, 1);
    ASSERT_THROW(a == d_wrong_shape, ShapeMismatchError);

    Tensor<int> sa({}, 10), sb({}, 10), sc({}, 11);
    AssertTensorBoolEqual(sa == sb, Tensor<bool>({}, true), "sa == sb (scalar)");
    AssertTensorBoolEqual(sa == sc, Tensor<bool>({}, false), "sa == sc (scalar)");

    Tensor<bool> result_empty = (empty_203 == empty_203); // Comparing two empty tensors of same shape
    ASSERT_EQ(result_empty.get_shape(), Tensor<bool>::shape_type({2,0,3}));
    ASSERT_TRUE(result_empty.is_empty());
}

TEST_F(TensorOperationsTest, EqualTensorScalar) {
    Tensor<int> a({2,2}, {5,10,5,15});
    Tensor<bool> expected_a_eq_5({2,2}, {true,false,true,false});
    AssertTensorBoolEqual(a == 5, expected_a_eq_5, "a == 5");
    AssertTensorBoolEqual(5 == a, expected_a_eq_5, "5 == a");

    Tensor<float> tf({1}, {10.0f});
    AssertTensorBoolEqual(tf == 10, Tensor<bool>({1},{true}), "tf == 10 (int)"); // tf == 10.0f (casted int)
    AssertTensorBoolEqual(10 == tf, Tensor<bool>({1},{true}), "10 (int) == tf");

    Tensor<int> ti({1}, {10});
    AssertTensorBoolEqual(ti == 10.0f, Tensor<bool>({1},{true}), "ti == 10.0f (float)"); // 10 == (int)10.0f
    AssertTensorBoolEqual(ti == 10.1f, Tensor<bool>({1},{true}), "ti == 10.1f (float)");// 10 == (int)10.1f
    AssertTensorBoolEqual(ti == 10.9f, Tensor<bool>({1},{true}), "ti == 10.9f (float)");// 10 == (int)10.9f
    AssertTensorBoolEqual(ti != 10.1f, Tensor<bool>({1},{false}), "ti != 10.1f (float)");
}


// --- NOT EQUAL (!=) TESTS ---
TEST_F(TensorOperationsTest, NotEqualTensorTensor) {
    Tensor<int> a({2,2}, {1,2,3,4});
    Tensor<int> b({2,2}, {1,0,3,5});
    Tensor<bool> expected_ab_ne({2,2}, {false,true,false,true});
    AssertTensorBoolEqual(a != b, expected_ab_ne, "a != b");
}

TEST_F(TensorOperationsTest, NotEqualTensorScalar) {
    Tensor<int> a({2,2}, {5,10,5,15});
    Tensor<bool> expected_a_ne_5({2,2}, {false,true,false,true});
    AssertTensorBoolEqual(a != 5, expected_a_ne_5, "a != 5");
    AssertTensorBoolEqual(5 != a, expected_a_ne_5, "5 != a");
}

// --- GREATER (>) TESTS ---
TEST_F(TensorOperationsTest, GreaterTensorTensor) {
    Tensor<int> a({2,2}, {5,2,8,3});
    Tensor<int> b({2,2}, {1,6,3,3}); // a > b will be {T,F,T,F}

    Tensor<bool> expected_a_gt_b({2,2}, {true,false,true,false});
    AssertTensorBoolEqual(a > b, expected_a_gt_b, "a > b");
}

TEST_F(TensorOperationsTest, GreaterTensorScalar) {
    Tensor<int> a({2,2}, {10,2,15,7});
    int scalar_val = 8; // a > scalar will be {T,F,T,F}

    Tensor<bool> expected_a_gt_s({2,2}, {true,false,true,false});
    AssertTensorBoolEqual(a > scalar_val, expected_a_gt_s, "a > scalar_val");
}

TEST_F(TensorOperationsTest, GreaterScalarTensor) {
    int scalar_val = 8;
    Tensor<int> a({2,2}, {10,2,5,7}); // scalar > a will be {F,T,T,T}

    Tensor<bool> expected_s_gt_a({2,2}, {false,true,true,true});
    AssertTensorBoolEqual(scalar_val > a, expected_s_gt_a, "scalar_val > a");
}


// --- LESS (<) TESTS ---
TEST_F(TensorOperationsTest, LessTensorTensor) {
    Tensor<int> a({2,2}, {1,6,3,3});
    Tensor<int> b({2,2}, {5,2,8,3}); // a < b will be {T,F,T,F}

    Tensor<bool> expected_a_lt_b({2,2}, {true,false,true,false});
    AssertTensorBoolEqual(a < b, expected_a_lt_b, "a < b");
}

TEST_F(TensorOperationsTest, LessTensorScalar) {
    Tensor<int> a({2,2}, {1,9,5,12});
    int scalar_val = 8; // a < scalar will be {T,F,T,F}

    Tensor<bool> expected_a_lt_s({2,2}, {true,false,true,false});
    AssertTensorBoolEqual(a < scalar_val, expected_a_lt_s, "a < scalar_val");
}

TEST_F(TensorOperationsTest, LessScalarTensor) {
    int scalar_val = 8;
    Tensor<int> a({2,2}, {1,9,10,5}); // scalar < a will be {F,T,T,F}

    Tensor<bool> expected_s_lt_a({2,2}, {false,true,true,false});
    AssertTensorBoolEqual(scalar_val < a, expected_s_lt_a, "scalar_val < a");
}


// --- GREATER EQUAL (>=) TESTS ---
TEST_F(TensorOperationsTest, GreaterEqualTensorTensor) {
    Tensor<int> a({2,2}, {5,2,8,3});
    Tensor<int> b({2,2}, {1,2,3,3}); // a >= b will be {T,T,T,T}

    Tensor<bool> expected_a_ge_b({2,2}, {true,true,true,true});
    AssertTensorBoolEqual(a >= b, expected_a_ge_b, "a >= b");
}

TEST_F(TensorOperationsTest, GreaterEqualTensorScalar) {
    Tensor<int> a({2,2}, {10,8,15,7});
    int scalar_val = 8; // a >= scalar will be {T,T,T,F}

    Tensor<bool> expected_a_ge_s({2,2}, {true,true,true,false});
    AssertTensorBoolEqual(a >= scalar_val, expected_a_ge_s, "a >= scalar_val");
}

TEST_F(TensorOperationsTest, GreaterEqualScalarTensor) {
    int scalar_val = 8;
    Tensor<int> a({2,2}, {10,8,5,9}); // scalar >= a will be {F,T,T,F}

    Tensor<bool> expected_s_ge_a({2,2}, {false,true,true,false});
    AssertTensorBoolEqual(scalar_val >= a, expected_s_ge_a, "scalar_val >= a");
}


// --- LESS EQUAL (<=) TESTS ---
TEST_F(TensorOperationsTest, LessEqualTensorTensor) {
    Tensor<int> a({2,2}, {1,2,3,3});
    Tensor<int> b({2,2}, {5,2,8,3}); // a <= b will be {T,T,T,T}

    Tensor<bool> expected_a_le_b({2,2}, {true,true,true,true});
    AssertTensorBoolEqual(a <= b, expected_a_le_b, "a <= b");
}

TEST_F(TensorOperationsTest, LessEqualTensorScalar) {
    Tensor<int> a({2,2}, {1,8,5,12});
    int scalar_val = 8; // a <= scalar will be {T,T,T,F}

    Tensor<bool> expected_a_le_s({2,2}, {true,true,true,false});
    AssertTensorBoolEqual(a <= scalar_val, expected_a_le_s, "a <= scalar_val");
}

TEST_F(TensorOperationsTest, LessEqualScalarTensor) {
    int scalar_val = 8;
    Tensor<int> a({2,2}, {1,8,10,5}); // scalar <= a will be {F,T,T,F}

    Tensor<bool> expected_s_le_a({2,2}, {false,true,true,false});
    AssertTensorBoolEqual(scalar_val <= a, expected_s_le_a, "scalar_val <= a");
}


// --- Common Edge Cases for Comparisons (Apply these patterns to all comparison types) ---
TEST_F(TensorOperationsTest, ComparisonsWithScalarTensors) {
    Tensor<int> s_a({}, 5);
    Tensor<int> s_b({}, 5);
    Tensor<int> s_c({}, 3);
    Tensor<int> s_d({}, 7);

    // ==
    AssertTensorBoolEqual(s_a == s_b, Tensor<bool>({}, true), "s_a == s_b");
    AssertTensorBoolEqual(s_a == s_c, Tensor<bool>({}, false), "s_a == s_c");
    // !=
    AssertTensorBoolEqual(s_a != s_c, Tensor<bool>({}, true), "s_a != s_c");
    // >
    AssertTensorBoolEqual(s_a > s_c, Tensor<bool>({}, true), "s_a > s_c");
    AssertTensorBoolEqual(s_c > s_a, Tensor<bool>({}, false), "s_c > s_a");
    // <
    AssertTensorBoolEqual(s_c < s_a, Tensor<bool>({}, true), "s_c < s_a");
    // >=
    AssertTensorBoolEqual(s_a >= s_b, Tensor<bool>({}, true), "s_a >= s_b");
    AssertTensorBoolEqual(s_a >= s_c, Tensor<bool>({}, true), "s_a >= s_c");
    // <=
    AssertTensorBoolEqual(s_a <= s_b, Tensor<bool>({}, true), "s_a <= s_b");
    AssertTensorBoolEqual(s_d <= s_a, Tensor<bool>({}, false), "s_d <= s_a");
}

TEST_F(TensorOperationsTest, ComparisonsWithEmptyShapedTensors) {
    Tensor<int> empty_s1({0});
    Tensor<int> empty_s2({0});
    Tensor<float> empty_s_f({0});
    int scalar_val = 5;

    // Tensor<bool> == Tensor<bool>
    Tensor<bool> res_eq_empty = (empty_s1 == empty_s2);
    ASSERT_EQ(res_eq_empty.get_shape(), Tensor<bool>::shape_type({0}));
    ASSERT_TRUE(res_eq_empty.is_empty());

    // Tensor<bool> == scalar
    Tensor<bool> res_eq_empty_scalar = (empty_s1 == scalar_val);
    ASSERT_EQ(res_eq_empty_scalar.get_shape(), Tensor<bool>::shape_type({0}));
    ASSERT_TRUE(res_eq_empty_scalar.is_empty());

    // Apply similar empty tests for >, <, >=, <=, !=
     Tensor<bool> res_gt_empty = (empty_s1 > empty_s2);
    ASSERT_EQ(res_gt_empty.get_shape(), Tensor<bool>::shape_type({0}));
    ASSERT_TRUE(res_gt_empty.is_empty());
}

TEST_F(TensorOperationsTest, ComparisonsShapeMismatch) {
    Tensor<int> a({2,2}, 1);
    Tensor<int> b({2,3}, 1);
    int scalar_val = 1;

    ASSERT_THROW(a == b, ShapeMismatchError);
    ASSERT_THROW(a != b, ShapeMismatchError);
    ASSERT_THROW(a > b, ShapeMismatchError);
    ASSERT_THROW(a < b, ShapeMismatchError);
    ASSERT_THROW(a >= b, ShapeMismatchError);
    ASSERT_THROW(a <= b, ShapeMismatchError);
    // Tensor-scalar operations do not throw shape mismatch for the scalar part.
}


// Placeholder for future scalar broadcast tests
// TEST_F(TensorOperationsTest, AddScalarBroadcast) {
//     Tensor<int> mat = Tensor<int>({2,2}, {1,2,3,4});
//     int scalar_val = 10;
//     Tensor<int> result = add(mat, scalar_val); // This needs add(Tensor, scalar) overload
//     ASSERT_EQ(result(0,0), 1 + 10);
//     // ...
// }

// Add more tests as you implement more operations (subtract, multiply, divide etc.)
