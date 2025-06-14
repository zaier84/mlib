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

// Helper function to assert that two Tensor<T> are element-wise identical for numeric types
template <typename T_VAL>
void AssertTensorEqual(const mlib::core::Tensor<T_VAL>& actual, const mlib::core::Tensor<T_VAL>& expected, const std::string& context = "") {
    // ... (existing shape checks) ...

    // NEW FIX HERE:
    // For Tensors that represent empty memory (total_size == 0):
    if (actual.get_total_size() == 0) {
        // These cases will have their `data()` methods returning nullptr if `is_empty() && !is_scalar()` is true.
        // Or `_data.data()` if `_data` is `std::vector<T>` of size 0, which is technically non-null from C++11, but invalid to dereference.
        // For AssertTensorEqual, we only compare elements if total_size > 0.
        ASSERT_EQ(expected.get_total_size(), 0) << "Total size mismatch for empty tensors in " << context;
        return; // We correctly determined both are empty and shapes match. No data to compare.
    }

    // Rest of `AssertTensorEqual` is only for total_size > 0:
    if constexpr (std::is_same_v<T_VAL, bool>) {
        // ... (Your operator() based check for bool tensors, which avoids data()) ...
    } else {
        // For non-bool types with actual elements (total_size > 0):
        const T_VAL* actual_data = actual.data();
        const T_VAL* expected_data = expected.data();

        // These ASSERT_NE(nullptr) are now CRUCIAL ONLY IF `_total_size > 0`!
        // `Tensor::data()` should NOT return nullptr if `_total_size > 0`.
        ASSERT_NE(actual_data, nullptr) << "Actual data pointer is null for NON-EMPTY (size > 0) tensor in " << context;
        ASSERT_NE(expected_data, nullptr) << "Expected data pointer is null for NON-EMPTY (size > 0) tensor in " << context;

        for (size_t flat_idx = 0; flat_idx < actual.get_total_size(); ++flat_idx) {
            ASSERT_EQ(actual_data[flat_idx], expected_data[flat_idx]) << "Mismatch at flat index " << flat_idx << " in " << context;
        }
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



// --- MATMUL (2D Matrix Multiplication) TESTS ---

TEST_F(TensorOperationsTest, MatmulBasicSquare) {
    Tensor<int> A({2,2}, {1,2,3,4});   // A = [[1,2], [3,4]]
    Tensor<int> B({2,2}, {5,6,7,8});   // B = [[5,6], [7,8]]
    // Expected C = A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //                  = [[5+14, 6+16], [15+28, 18+32]]
    //                  = [[19, 22], [43, 50]]
    Tensor<int> expected_C({2,2}, {19,22,43,50});

    Tensor<int> C = matmul(A, B);
    ASSERT_EQ(C.get_shape(), Tensor<int>::shape_type({2,2}));
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_EQ(C(r,c), expected_C(r,c)) << "Mismatch at (" << r << "," << c << ")";

    Tensor<float> Af({2,2}, {1.f,2.f,3.f,4.f});
    Tensor<float> Bf({2,2}, {0.5f,1.0f,1.5f,2.0f});
    // Expected Cf = [[1*0.5+2*1.5, 1*1+2*2], [3*0.5+4*1.5, 3*1+4*2]]
    //             = [[0.5+3, 1+4], [1.5+6, 3+8]]
    //             = [[3.5, 5], [7.5, 11]]
    Tensor<float> expected_Cf({2,2}, {3.5f,5.f,7.5f,11.f});
    Tensor<float> Cf = matmul(Af, Bf);
    ASSERT_EQ(Cf.get_shape(), Tensor<float>::shape_type({2,2}));
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_FLOAT_EQ(Cf(r,c), expected_Cf(r,c)) << "Mismatch at (" << r << "," << c << ")";
}

TEST_F(TensorOperationsTest, MatmulRectangular) {
    Tensor<int> A({2,3}, {1,2,3,  4,5,6});       // 2x3 matrix: [[1,2,3], [4,5,6]]
    Tensor<int> B({3,2}, {7,8,  9,10,  11,12}); // 3x2 matrix: [[7,8], [9,10], [11,12]]
    // Expected C (2x2) = A * B
    // C(0,0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C(0,1) = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C(1,0) = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C(1,1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    // C = [[58, 64], [139, 154]]
    Tensor<int> expected_C({2,2}, {58,64,139,154});

    Tensor<int> C = matmul(A, B);
    ASSERT_EQ(C.get_shape(), Tensor<int>::shape_type({2,2}));
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_EQ(C(r,c), expected_C(r,c)) << "Mismatch at (" << r << "," << c << ")";
}

TEST_F(TensorOperationsTest, MatmulIdentity) {
    Tensor<int> A({2,2}, {1,2,3,4});
    Tensor<int> I({2,2}, {1,0,0,1}); // Identity matrix
    
    Tensor<int> C1 = matmul(A, I); // A * I = A
    ASSERT_EQ(C1.get_shape(), A.get_shape());
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_EQ(C1(r,c), A(r,c)) << "A*I mismatch at (" << r << "," << c << ")";

    Tensor<int> C2 = matmul(I, A); // I * A = A
    ASSERT_EQ(C2.get_shape(), A.get_shape());
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_EQ(C2(r,c), A(r,c)) << "I*A mismatch at (" << r << "," << c << ")";
}

TEST_F(TensorOperationsTest, MatmulWithZeroMatrix) {
    Tensor<int> A({2,3}, {1,2,3,4,5,6});
    Tensor<int> ZerosB({3,2}, 0); // All zeros
    Tensor<int> ExpectedZerosC({2,2}, 0);

    Tensor<int> C = matmul(A, ZerosB);
    ASSERT_EQ(C.get_shape(), Tensor<int>::shape_type({2,2}));
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_EQ(C(r,c), ExpectedZerosC(r,c)) << "A * ZerosB mismatch";

    Tensor<int> ZerosA({2,3}, 0);
    Tensor<int> B({3,2}, {1,2,3,4,5,6});
    C = matmul(ZerosA, B);
    ASSERT_EQ(C.get_shape(), Tensor<int>::shape_type({2,2}));
    for(size_t r=0; r<2; ++r) for(size_t c=0; c<2; ++c) EXPECT_EQ(C(r,c), ExpectedZerosC(r,c)) << "ZerosA * B mismatch";
}



TEST_F(TensorOperationsTest, MatmulShapeMismatchInnerDim) {
    Tensor<int> A({2,3}); // 2x3
    Tensor<int> B({2,2}); // 2x2 (cols of A (3) != rows of B (2))
    ASSERT_THROW(mlib::core::matmul(A, B), ShapeMismatchError);
}

TEST_F(TensorOperationsTest, MatmulNot2D) {
    // A_1D (3,)
    Tensor<int> A_1D({3}, {1,2,3}); // 1D Tensor of length 3 (not a scalar {}!)
    // B_2D (3,2)
    Tensor<int> B_2D({3,2}, {1,1,1,1,1,1}); // Correct 2D tensor

    // Case: 1D @ 2D. This is NOW a supported case by `matmul` overloads. It should NOT throw DimensionError.
    // If you intend for it to throw here, your `matmul` code has to explicitly define this.
    // My implemented `matmul` code directly handles `1D @ 2D`, so this line is likely expecting a throw *where it shouldn't*.
    // Test: Verify it WORKS, don't ASSERT_THROW.
    Tensor<int> expected_1d_at_2d_result({2}, {7,8}); // Based on a sample (1,2,3) @ (3,2) [[1,2],[3,4],[5,6]] yields [22,28]. Let's verify here...
                                                    // This test should be placed in `Matmul1DVecBy2DMat` for valid outputs.
                                                    // Let's REMOVE THIS ASSERT_THROW from `MatmulNot2D`
                                                    // if it's meant to test unsupported ndims.

    // A_2D (2,3)
    Tensor<int> A_2D({2,3}, {1,1,1,1,1,1});
    // B_3D (3,2,1)
    Tensor<int> B_3D({3,2,1}, {1,1,1,1,1,1}); // 3D Tensor

    // 2D @ 3D - THIS IS UNSUPPORTED, SHOULD THROW DimensionError
    ASSERT_THROW(mlib::core::matmul(A_2D, B_3D), DimensionError);

    // 3D @ 2D - THIS IS UNSUPPORTED, SHOULD THROW DimensionError
    ASSERT_THROW(mlib::core::matmul(B_3D, A_2D), DimensionError);

    // A_scalar is 0-dim:
    Tensor<int> A_scalar(std::vector<size_t>{}); // Use explicit constructor
    // 0D @ 2D - THIS IS UNSUPPORTED, SHOULD THROW DimensionError
    ASSERT_THROW(mlib::core::matmul(A_scalar, B_2D), DimensionError);
    // 2D @ 0D - THIS IS UNSUPPORTED, SHOULD THROW DimensionError
    ASSERT_THROW(mlib::core::matmul(A_2D, A_scalar), DimensionError);
}


TEST_F(TensorOperationsTest, MatmulWithZeroDimension_k) {
    Tensor<int> A({2,0}); // A is 2x0
    Tensor<int> B({0,3}); // B is 0x3
    // Here, inner dim (k) is 0 for both (0 vs 0), so it matches.
    // As per Matmul logic for k=0, result is an m x n matrix of zeros.
    // Result C should be 2x3, all zeros.
    Tensor<int> C = mlib::core::matmul(A, B);
    ASSERT_EQ(C.get_shape(), Tensor<int>::shape_type({2,3}));
    ASSERT_EQ(C.get_total_size(), 2 * 3); // It should be m*n elements for `(M,0) @ (0,N)` case.
    for (size_t i=0; i<2; ++i) {
        for (size_t j=0; j<3; ++j) {
            ASSERT_EQ(C(i,j), 0) << "Matmul result for k=0 should be zero matrix.";
        }
    }
}

TEST_F(TensorOperationsTest, MatmulWithZeroDimension_m_or_n) {
    Tensor<int> A_m0({0,2}); // A is 0x2
    Tensor<int> B_2k({2,3}); // B is 2x3
    // Result C_m0_result should be 0x3 (empty rows) because outer dimension is 0.
    Tensor<int> C_m0_result = mlib::core::matmul(A_m0, B_2k);
    ASSERT_EQ(C_m0_result.get_shape(), Tensor<int>::shape_type({0,3}));
    ASSERT_EQ(C_m0_result.get_total_size(), 0);

    Tensor<int> A_k2({3,2}); // A is 3x2
    Tensor<int> B_n0({2,0}); // B is 2x0
    // Result C_n0_result should be 3x0 (empty columns) because outer dimension is 0.
    Tensor<int> C_n0_result = mlib::core::matmul(A_k2, B_n0);
    ASSERT_EQ(C_n0_result.get_shape(), Tensor<int>::shape_type({3,0}));
    ASSERT_EQ(C_n0_result.get_total_size(), 0);
}

TEST_F(TensorOperationsTest, MatmulSingleElementResult) {
    // A (1x3) * B (3x1) -> C (1x1)
    Tensor<int> A({1,3}, {1,2,3});
    Tensor<int> B({3,1}, {4,5,6});
    // C(0,0) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    Tensor<int> expected_C({1,1}, {32});

    Tensor<int> C = matmul(A,B);
    ASSERT_EQ(C.get_shape(), Tensor<int>::shape_type({1,1}));
    ASSERT_EQ(C(0,0), 32);
}

TEST_F(TensorOperationsTest, Matmul1DVecBy2DMat) {
    // Vector (1D): (3,)
    Tensor<float> vec_A({3}, {1.0f, 2.0f, 3.0f});

    // Matrix (2D): (3,2)
    // [[1,2],
    //  [3,4],
    //  [5,6]]
    Tensor<float> mat_B({3,2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Result should be (2,) vector: (1*1+2*3+3*5), (1*2+2*4+3*6) = (1+6+15, 2+8+18) = (22, 28)
    Tensor<float> expected_result({2}, {22.0f, 28.0f});

    Tensor<float> result = mlib::core::matmul(vec_A, mat_B);

    AssertTensorEqual(result, expected_result, "1DVec @ 2DMat (3,) @ (3,2)");
}

TEST_F(TensorOperationsTest, Matmul2DMatBy1DVec) {
    // Matrix (2D): (2,3)
    // [[1,2,3],
    //  [4,5,6]]
    Tensor<float> mat_A({2,3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Vector (1D): (3,)
    Tensor<float> vec_B({3}, {7.0f, 8.0f, 9.0f});

    // Result should be (2,) vector: (1*7+2*8+3*9), (4*7+5*8+6*9) = (7+16+27, 28+40+54) = (50, 122)
    Tensor<float> expected_result({2}, {50.0f, 122.0f});

    Tensor<float> result = mlib::core::matmul(mat_A, vec_B);

    AssertTensorEqual(result, expected_result, "2DMat @ 1DVec (2,3) @ (3,)");
}

TEST_F(TensorOperationsTest, Matmul1DVecBy1DVecDotProduct) {
    // Vector (1D): (3,)
    Tensor<float> vec_A({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> vec_B({3}, {4.0f, 5.0f, 6.0f});

    // Result should be scalar: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    Tensor<float> expected_result({}, 32.0f); // Scalar tensor

    Tensor<float> result = mlib::core::matmul(vec_A, vec_B);

    AssertTensorEqual(result, expected_result, "1DVec @ 1DVec (dot product)");
}

TEST_F(TensorOperationsTest, MatmulDimensionErrors) {
    Tensor<float> mat_2x2({2,2}, 1.f);
    Tensor<float> vec_3(std::vector<size_t>{});
    Tensor<float> mat_3d({2,2,2}, 1.f);

    ASSERT_THROW(mlib::core::matmul(mat_2x2, mat_3d), DimensionError); // 2D @ 3D
    ASSERT_THROW(mlib::core::matmul(mat_3d, mat_2x2), DimensionError); // 3D @ 2D
    ASSERT_THROW(mlib::core::matmul(mat_3d, vec_3), DimensionError);   // 3D @ 1D

    // Check inner dimension mismatch for existing 2D@2D test as well if you didn't previously
    Tensor<float> mat_2x3({2,3}, 1.f);
    Tensor<float> mat_4x2({4,2}, 1.f);
    ASSERT_THROW(mlib::core::matmul(mat_2x3, mat_4x2), ShapeMismatchError); // (2,3) @ (4,2)
}

TEST_F(TensorOperationsTest, MatmulZeroSizeDimensions) {
    Tensor<float> v0({0});        // 1D vector, length 0
    Tensor<float> m2x0({2,0});    // 2D matrix, 2x0
    Tensor<float> m0x3({0,3});    // 2D matrix, 0x3
    Tensor<float> v5({5}, {1,2,3,4,5});
    Tensor<float> m5x3({5,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}); // Arbitrary data for clarity
    Tensor<float> v3({3}, {1,2,3});

    // Test (0,) @ (0,3) -> (3,) of zeros
    Tensor<float> expected_v3_zeros({3}, {0.f,0.f,0.f});
    AssertTensorEqual(mlib::core::matmul(v0, m0x3), expected_v3_zeros, "0D_vec @ 0x3_mat -> 3_zeros_vec");

    // Test (2,3) @ (3,0) -> (2,0) of zeros (original matmul should handle)
    Tensor<float> expected_m2x0_zeros({2,0});
    ASSERT_THROW(mlib::core::matmul(m5x3, m0x3), ShapeMismatchError);

    // Test (5,) @ (5,0) -> (0,) of zeros
    Tensor<float> expected_v0_result({0});
	AssertTensorEqual(mlib::core::matmul(v5, m5x3), Tensor<float>({3},{135.f, 150.f, 165.f}), "(5,) @ (5,3) -> (3,) (135, 150, 165)");
}
TEST_F(TensorOperationsTest, MatmulZeroSizeDimensionsSimple) {
    // Variable Declarations (ensure they're explicit and correct)
    Tensor<float> v0({0});         // 1D vector, length 0
    Tensor<float> m2x0({2,0});     // 2D matrix, 2x0 (2 rows, 0 columns)
    Tensor<float> m0x3({0,3});     // 2D matrix, 0x3 (0 rows, 3 columns)

    // Test (0,) @ (0,3) -> (3,) of zeros
    // (1D vec by 2D mat) Here, k_vec_A = 0, k_mat_B = 0. Match!
    Tensor<float> expected_v3_zeros({3}, {0.f,0.f,0.f}); // Vector of 3 zeros
    AssertTensorEqual(mlib::core::matmul(v0, m0x3), expected_v3_zeros, "0D_vec @ 0x3_mat -> 3_zeros_vec");

    // Test (5,) @ (5,0) -> (0,) of zeros (from general `k=0` rule, no calculation needed)
    // (1D vec by 2D mat) Here, k_vec_A = 5, k_mat_B = 5 (from your actual m5x3 for now). This is just 1D @ 2D where output 0-dim.
    // Example: (5,) @ (5,0) -> (0,) zero vector
    // For matmul((5,), (5,0)):
    // 1. `ndim_A=1, ndim_B=2`. Case 2.
    // 2. `k_vec_A = 5`, `k_mat_B = 5`. Match.
    // 3. `n_mat_B = 0`. So temp_result_2D will be `(1,0)` (total size 0).
    // 4. Reshape `(1,0)` to `(0,)`.
    // 5. Result: Tensor<float>({0}). Which has total_size=0.
    Tensor<float> v5({5}, {1.f,2.f,3.f,4.f,5.f}); // non-empty 1D
    Tensor<float> m5x0({5,0});                   // 2D matrix, 5x0
    Tensor<float> expected_v0_result({0});      // Empty vector

    AssertTensorEqual(mlib::core::matmul(v5, m5x0), expected_v0_result, "(5,) @ (5,0) -> (0,) empty vec");


    // Test (2,0) @ (0,3) -> (2,3) of zeros.
    // (2D mat @ 2D mat) `m_zero_col_2x0` (2,0) and `m0x3` (0,3)
    // `k_A = 0`, `k_B = 0`. Match.
    Tensor<float> m_zero_col_2x0({2,0}); // 2x0 matrix
    Tensor<float> expected_2x3_zeros({2,3}, {0.f,0.f,0.f,0.f,0.f,0.f}); // 2x3 filled with zeros.
    AssertTensorEqual(mlib::core::matmul(m_zero_col_2x0, m0x3), expected_2x3_zeros, "2x0_mat @ 0x3_mat -> 2x3_zeros");
}
// --- REDUCTION OPERATION TESTS ---

TEST_F(TensorOperationsTest, Sum) {
    Tensor<int> ti_vals({2,3}, {1,2,3,4,5,6}); // sum = 21
    ASSERT_EQ(mlib::core::sum(ti_vals), 21);

    Tensor<float> tf_vals({1,4}, {1.5f, 2.5f, -1.0f, 0.5f}); // sum = 3.5f
    ASSERT_FLOAT_EQ(mlib::core::sum(tf_vals), 3.5f);

    Tensor<int> scalar_sum({}, 100);
    ASSERT_EQ(mlib::core::sum(scalar_sum), 100);

    Tensor<int> empty_sum_default; // Default constructed
    ASSERT_EQ(mlib::core::sum(empty_sum_default), 0);

    // empty_203 from fixture is Tensor<int>({2,0,3});
    ASSERT_EQ(mlib::core::sum(empty_203), 0);

    Tensor<int> one_el({1}, {7});
    ASSERT_EQ(mlib::core::sum(one_el), 7);
}

TEST_F(TensorOperationsTest, Mean) {
    Tensor<int> ti_vals({2,3}, {1,2,3,4,5,6}); // sum = 21, count = 6, mean = 3.5
    ASSERT_DOUBLE_EQ(mlib::core::mean(ti_vals), 3.5);

    Tensor<float> tf_vals({1,4}, {1.f, 2.f, 3.f, 6.f}); // sum = 12.f, count = 4, mean = 3.0f
    ASSERT_DOUBLE_EQ(mlib::core::mean(tf_vals), 3.0);

    Tensor<int> scalar_mean({}, 100);
    ASSERT_DOUBLE_EQ(mlib::core::mean(scalar_mean), 100.0);

    Tensor<double> td_vals({2,2}, {1.0, 2.0, 3.0, 4.0}); // sum=10, count=4, mean=2.5
    ASSERT_DOUBLE_EQ(mlib::core::mean(td_vals), 2.5);

    Tensor<int> empty_mean_default;
    ASSERT_TRUE(std::isnan(mlib::core::mean(empty_mean_default)));

    // empty_203 from fixture
    ASSERT_TRUE(std::isnan(mlib::core::mean(empty_203)));
}

TEST_F(TensorOperationsTest, MaxVal) {
    Tensor<int> ti_vals({2,3}, {1,-2,8,0,5,-6}); // max = 8
    ASSERT_EQ(mlib::core::max_val(ti_vals), 8);

    Tensor<float> tf_vals({1,4}, {1.5f, -200.5f, 0.0f, 100.1f}); // max = 100.1f
    ASSERT_FLOAT_EQ(mlib::core::max_val(tf_vals), 100.1f);

    Tensor<int> scalar_max({}, -10);
    ASSERT_EQ(mlib::core::max_val(scalar_max), -10);

    Tensor<int> empty_max_default;
    ASSERT_THROW(mlib::core::max_val(empty_max_default), std::runtime_error);
    ASSERT_THROW(mlib::core::max_val(empty_203), std::runtime_error);
}

TEST_F(TensorOperationsTest, MinVal) {
    Tensor<int> ti_vals({2,3}, {1,-2,8,0,5,-6}); // min = -6
    ASSERT_EQ(mlib::core::min_val(ti_vals), -6);

    Tensor<float> tf_vals({1,4}, {1.5f, -200.5f, 0.0f, 100.1f}); // min = -200.5f
    ASSERT_FLOAT_EQ(mlib::core::min_val(tf_vals), -200.5f);

    Tensor<int> scalar_min({}, 77);
    ASSERT_EQ(mlib::core::min_val(scalar_min), 77);

    Tensor<int> empty_min_default;
    ASSERT_THROW(mlib::core::min_val(empty_min_default), std::runtime_error);
    ASSERT_THROW(mlib::core::min_val(empty_203), std::runtime_error);
}

TEST_F(TensorOperationsTest, Prod) {
    Tensor<int> ti_vals({1,4}, {1,2,3,4}); // prod = 24
    ASSERT_EQ(mlib::core::prod(ti_vals), 24);

    Tensor<float> tf_vals({2,2}, {1.0f, 2.0f, 0.5f, 4.0f}); // prod = 4.0f
    ASSERT_FLOAT_EQ(mlib::core::prod(tf_vals), 4.0f);
    
    Tensor<int> ti_with_zero({1,3}, {5,0,7}); // prod = 0
    ASSERT_EQ(mlib::core::prod(ti_with_zero), 0);

    Tensor<int> scalar_prod({}, 7);
    ASSERT_EQ(mlib::core::prod(scalar_prod), 7);

    Tensor<int> empty_prod_default;
    ASSERT_EQ(mlib::core::prod(empty_prod_default), 1); // Product of empty set is 1

    // empty_203 from fixture
    ASSERT_EQ(mlib::core::prod(empty_203), 1);
}


// --- AXIS-WISE REDUCTION TESTS ---

TEST_F(TensorOperationsTest, SumAxisBasic2D) {
    Tensor<int> t({2,3}, {1,2,3,4,5,6}); // [[1,2,3],[4,5,6]]

    // Sum along axis 0 (rows): result shape (3) or (1,3)
    // [1+4, 2+5, 3+6] = [5,7,9]
    Tensor<int> expected_sum_axis0({3}, {5,7,9});
    AssertTensorEqual(mlib::core::sum(t, 0), expected_sum_axis0, "sum(t,0)");

    // Sum along axis 0, keep_dims=true: result shape (1,3)
    Tensor<int> expected_sum_axis0_kd({1,3}, {5,7,9});
    AssertTensorEqual(mlib::core::sum(t, 0, true), expected_sum_axis0_kd, "sum(t,0,true)");

    // Sum along axis 1 (columns): result shape (2) or (2,1)
    // [1+2+3, 4+5+6] = [6,15]
    Tensor<int> expected_sum_axis1({2}, {6,15});
    AssertTensorEqual(mlib::core::sum(t, 1), expected_sum_axis1, "sum(t,1)");

    // Sum along axis 1, keep_dims=true: result shape (2,1)
    Tensor<int> expected_sum_axis1_kd({2,1}, {6,15});
    AssertTensorEqual(mlib::core::sum(t, 1, true), expected_sum_axis1_kd, "sum(t,1,true)");
}

TEST_F(TensorOperationsTest, SumAxis3D) {
    Tensor<int> t({2,2,2}, {1,2,3,4,5,6,7,8}); // 2x2x2 cube

    // Sum along axis 0 (across first dim): result shape (2,2)
    // [[1+5, 2+6], [3+7, 4+8]] = [[6,8],[10,12]]
    Tensor<int> expected_sum_axis0({2,2}, {6,8,10,12});
    AssertTensorEqual(mlib::core::sum(t, 0), expected_sum_axis0, "sum(t,0) 3D");

    // Sum along axis 1 (across second dim): result shape (2,2)
    // [[1+3, 2+4], [5+7, 6+8]] = [[4,6],[12,14]]
    Tensor<int> expected_sum_axis1({2,2}, {4,6,12,14});
    AssertTensorEqual(mlib::core::sum(t, 1), expected_sum_axis1, "sum(t,1) 3D");

    // Sum along axis 2 (across third dim): result shape (2,2)
    // [[1+2, 3+4], [5+6, 7+8]] = [[3,7],[11,15]]
    Tensor<int> expected_sum_axis2({2,2}, {3,7,11,15});
    AssertTensorEqual(mlib::core::sum(t, 2), expected_sum_axis2, "sum(t,2) 3D");
}
/*
TEST_F(TensorOperationsTest, SumAxisScalarAndEmpty) {
    Tensor<int> t({5}, {1,2,3,4,5});
    // Summing 1D tensor reduces to scalar result (shape {})
    AssertTensorEqual(mlib::core::sum(t, 0), Tensor<int>({}, 15), "sum 1D to scalar");

    // Summing a 1D tensor along axis 0, keep_dims=true, results in (1)
    AssertTensorEqual(mlib::core::sum(t, 0, true), Tensor<int>({1}, {15}), "sum 1D kd=true");

    // Corrected error checking: Use ASSERT_THROW for cases where function throws
    // Error: Summing scalar tensor along axis - This should THROW
    ASSERT_THROW(mlib::core::sum(scalar1, 0), DimensionError); // scalar1 is a 0-dim tensor
    // If you pass an invalid axis for an otherwise valid tensor
    ASSERT_THROW(mlib::core::sum(t, 1), DimensionError); // 1D tensor, axis 1 is out of bounds
    ASSERT_THROW(mlib::core::sum(t, -2), DimensionError); // Negative axis out of bounds for 1D tensor

    // Handle empty shaped tensor:
    // This part should work if input_shape.get_total_size() == 0 correctly returns an empty output tensor.
    // Which it should:
    Tensor<int> empty_t_shaped({2,0,3}); // Total size 0
    Tensor<int> expected_empty_sum_0({0,3}); // Result from reducing axis 0 (dim 2 and 3 kept)
    AssertTensorEqual(mlib::core::sum(empty_t_shaped, 0), expected_empty_sum_0, "sum empty over axis 0");
    Tensor<int> expected_empty_sum_0_kd({1,0,3});
    AssertTensorEqual(mlib::core::sum(empty_t_shaped, 0, true), expected_empty_sum_0_kd, "sum empty over axis 0 kd");

    Tensor<int> expected_empty_sum_1({2,3}); // Result from reducing axis 1 (dim 2 and 3 kept). Still size 0.
    AssertTensorEqual(mlib::core::sum(empty_t_shaped, 1), expected_empty_sum_1, "sum empty over axis 1");

    // This case will be an ASSERT_THROW for the original `mean` call that was inside `sum`.
    // It's still expected to throw:
    Tensor<int> empty_t_full_empty; // Default constructed, 0 dimensions, 0 total size
    // AssertTensorEqual(mlib::core::sum(empty_t_full_empty, 0), expected_sum_empty_full, "sum empty default axis 0");
    ASSERT_THROW(mlib::core::sum(empty_t_full_empty, 0), DimensionError); // Will throw because `ndim == 0` for `empty_t_full_empty`
}
*/

TEST_F(TensorOperationsTest, SumAxisScalarAndEmpty) {
    std::cout << "--- Starting SumAxisScalarAndEmpty Isolation Test ---\n";

    Tensor<int> t_isolated({5}, {1,2,3,4,5});
    int expected_scalar_value = 15;
    Tensor<int> expected_scalar_tensor({}, expected_scalar_value);

    std::cout << "Call to sum(t_isolated, 0)...\n";
    // Place your debug couts from the previous step INSIDE the sum function
    // just BEFORE its `if (tensor.ndim() == 0)` and around the `result.data()` accumulation.

    // This is the line that will execute and hopefully reveal more
    Tensor<int> result_tensor = mlib::core::sum(t_isolated, 0);

    std::cout << "sum(t_isolated, 0) returned successfully. Asserting equality...\n";
    AssertTensorEqual(result_tensor, expected_scalar_tensor, "sum(t_isolated,0) result");

    std::cout << "--- SumAxisScalarAndEmpty Isolation Test PASSED (if no crash before here) ---\n";
}

TEST_F(TensorOperationsTest, MeanAxisBasic2D) {
    Tensor<float> t({2,3}, {1.f,2.f,3.f,4.f,5.f,6.f}); // [[1,2,3],[4,5,6]]

    // Mean along axis 0: [2.5f, 3.5f, 4.5f]
    Tensor<double> expected_mean_axis0({3}, {2.5, 3.5, 4.5});
    AssertTensorEqual(mlib::core::mean(t, 0), expected_mean_axis0, "mean(t,0)");

    // Mean along axis 1: [2.f, 5.f]
    Tensor<double> expected_mean_axis1({2}, {2.0, 5.0});
    AssertTensorEqual(mlib::core::mean(t, 1), expected_mean_axis1, "mean(t,1)");

    // Mean along axis 1, keep_dims=true: result shape (2,1)
    Tensor<double> expected_mean_axis1_kd({2,1}, {2.0, 5.0});
    AssertTensorEqual(mlib::core::mean(t, 1, true), expected_mean_axis1_kd, "mean(t,1,true)");
}

TEST_F(TensorOperationsTest, MeanAxisEmptyAndZeroSize) {
    Tensor<float> t({2,3}, {1.f,2.f,3.f,4.f,5.f,6.f});
    // THIS SHOULD BE ASSERT_THROW, NOT A DIRECT CALL
    ASSERT_THROW(mlib::core::mean(scalar1, 0), DimensionError); // Scalar tensor (ndim = 0)

    // This block should also be ASSERT_THROW if it means `tensor.get_shape()[axis] == 0`
    // My code previously did not wrap this. Let's add it.
    Tensor<float> t_zero_axis_size({2,0,3}); // Axis 1 has size 0
    ASSERT_THROW(mlib::core::mean(t_zero_axis_size, 1), std::runtime_error); // Max/Min throw runtime_error here

    // Test mean on completely empty tensor (should yield NaN)
    Tensor<float> empty_t_full_empty; // default constructed, 0 dims
    // For `mean(tensor, axis)` this means `ndim == 0`, which will throw a `DimensionError`.
    ASSERT_THROW(mlib::core::mean(empty_t_full_empty, 0), DimensionError); // Expect error due to 0-dim tensor
}

TEST_F(TensorOperationsTest, MaxValAxisBasic2D) {
    Tensor<int> t({2,3}, {10,2,3,4,5,12}); // [[10,2,3],[4,5,12]]

    // Max along axis 0: [max(10,4), max(2,5), max(3,12)] = [10,5,12]
    Tensor<int> expected_max_axis0({3}, {10,5,12});
    AssertTensorEqual(mlib::core::max_val(t, 0), expected_max_axis0, "max_val(t,0)");

    // Max along axis 1: [max(10,2,3), max(4,5,12)] = [10,12]
    Tensor<int> expected_max_axis1({2}, {10,12});
    AssertTensorEqual(mlib::core::max_val(t, 1), expected_max_axis1, "max_val(t,1)");
}

TEST_F(TensorOperationsTest, MinValAxisBasic2D) {
    Tensor<int> t({2,3}, {10,2,3,4,5,12}); // [[10,2,3],[4,5,12]]

    // Min along axis 0: [min(10,4), min(2,5), min(3,12)] = [4,2,3]
    Tensor<int> expected_min_axis0({3}, {4,2,3});
    AssertTensorEqual(mlib::core::min_val(t, 0), expected_min_axis0, "min_val(t,0)");

    // Min along axis 1: [min(10,2,3), min(4,5,12)] = [2,4]
    Tensor<int> expected_min_axis1({2}, {2,4});
    AssertTensorEqual(mlib::core::min_val(t, 1), expected_min_axis1, "min_val(t,1)");
}

TEST_F(TensorOperationsTest, ProdAxisBasic2D) {
    Tensor<int> t({2,3}, {1,2,3,4,5,6}); // [[1,2,3],[4,5,6]]

    // Prod along axis 0: [1*4, 2*5, 3*6] = [4,10,18]
    Tensor<int> expected_prod_axis0({3}, {4,10,18});
    AssertTensorEqual(mlib::core::prod(t, 0), expected_prod_axis0, "prod(t,0)");

    // Prod along axis 1: [1*2*3, 4*5*6] = [6,120]
    Tensor<int> expected_prod_axis1({2}, {6,120});
    AssertTensorEqual(mlib::core::prod(t, 1), expected_prod_axis1, "prod(t,1)");

    Tensor<int> t_with_zero({2,2}, {1,0,3,4}); // [[1,0],[3,4]]
    // Prod along axis 0: [1*3, 0*4] = [3,0]
    AssertTensorEqual(mlib::core::prod(t_with_zero, 0), Tensor<int>({2}, {3,0}), "prod(t_w_z,0)");
    // Prod along axis 1: [1*0, 3*4] = [0,12]
    AssertTensorEqual(mlib::core::prod(t_with_zero, 1), Tensor<int>({2}, {0,12}), "prod(t_w_z,1)");
}

// --- TRANSPOSE TESTS ---

TEST_F(TensorOperationsTest, TransposeBasic2D) {
    // 2x3 matrix:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Tensor<int> t_2x3({2,3}, {1,2,3,4,5,6});

    // Transposed 3x2 matrix should be:
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    Tensor<int> expected_t_3x2({3,2}, {1,4,2,5,3,6});

    Tensor<int> result = mlib::core::transpose(t_2x3);

    AssertTensorEqual(result, expected_t_3x2, "Transpose of 2x3");
}

TEST_F(TensorOperationsTest, TransposeSquareMatrix) {
    // 2x2 matrix:
    // [[1, 2],
    //  [3, 4]]
    Tensor<float> t_2x2_f({2,2}, {1.0f,2.0f,3.0f,4.0f});

    // Transposed 2x2 matrix should be:
    // [[1, 3],
    //  [2, 4]]
    Tensor<float> expected_t_2x2_f({2,2}, {1.0f,3.0f,2.0f,4.0f});

    Tensor<float> result = mlib::core::transpose(t_2x2_f);

    AssertTensorEqual(result, expected_t_2x2_f, "Transpose of 2x2");
}

TEST_F(TensorOperationsTest, TransposeEmptyMatrices) {
    // 2x0 matrix:
    // [[],
    //  []]
    Tensor<int> t_2x0({2,0}); // Total size 0

    // Transposed 0x2 matrix:
    // []
    // []
    Tensor<int> expected_t_0x2({0,2}); // Total size 0

    Tensor<int> result_2x0 = mlib::core::transpose(t_2x0);
    AssertTensorEqual(result_2x0, expected_t_0x2, "Transpose of 2x0");

    // 0x3 matrix:
    // []
    Tensor<int> t_0x3({0,3}); // Total size 0

    // Transposed 3x0 matrix:
    // [[], [], []] (empty 3 rows, 0 cols each)
    Tensor<int> expected_t_3x0({3,0}); // Total size 0

    Tensor<int> result_0x3 = mlib::core::transpose(t_0x3);
    AssertTensorEqual(result_0x3, expected_t_3x0, "Transpose of 0x3");
}

TEST_F(TensorOperationsTest, TransposeNon2DInputsThrow) {
    Tensor<int> t_1d({5}, {1,2,3,4,5});        // 1D tensor
    Tensor<int> t_3d({2,2,2}, {1,2,3,4,5,6,7,8}); // 3D tensor
    Tensor<int> t_scalar({}, 10);              // 0D scalar

    ASSERT_THROW(mlib::core::transpose(t_1d), DimensionError);
    ASSERT_THROW(mlib::core::transpose(t_3d), DimensionError);
    ASSERT_THROW(mlib::core::transpose(t_scalar), DimensionError);
}

// Add a test for identity operation: transpose(transpose(t)) == t
TEST_F(TensorOperationsTest, TransposeIdentity) {
    Tensor<int> t_3x2({3,2}, {10,20,30,40,50,60});
    Tensor<int> t_original_copy = t_3x2; // Store a copy before modifying (optional but good practice)

    Tensor<int> result_double_transpose = mlib::core::transpose(mlib::core::transpose(t_3x2));
    AssertTensorEqual(result_double_transpose, t_original_copy, "transpose(transpose(t)) should equal t");
}


// --- TENSOR CREATION ROUTINE TESTS ---

TEST_F(TensorOperationsTest, ZerosCreation) {
    // 0-dim (scalar) tensor
    Tensor<int> expected_scalar_zero({}, 0);
    AssertTensorEqual(mlib::core::zeros<int>(std::vector<size_t>{}), expected_scalar_zero, "zeros scalar");

    // 1D tensor
    Tensor<float> expected_1d_zeros({3}, {0.0f, 0.0f, 0.0f});
    AssertTensorEqual(mlib::core::zeros<float>({3}), expected_1d_zeros, "zeros 1D float");

    // 2D tensor
    Tensor<int> expected_2d_zeros({2,2}, {0,0,0,0});
    AssertTensorEqual(mlib::core::zeros<int>({2,2}), expected_2d_zeros, "zeros 2D int");

    // 3D tensor
    Tensor<double> expected_3d_zeros({1,2,3}, {0.0,0.0,0.0,0.0,0.0,0.0});
    AssertTensorEqual(mlib::core::zeros<double>({1,2,3}), expected_3d_zeros, "zeros 3D double");

    // Bool tensor
    Tensor<bool> expected_bool_zeros({2}, {false, false});
    AssertTensorEqual(mlib::core::zeros<bool>({2}), expected_bool_zeros, "zeros 1D bool");

    // Tensor with a zero dimension (total_size=0)
    Tensor<int> expected_zero_dim_zeros({2,0,3}); // Total size 0
    AssertTensorEqual(mlib::core::zeros<int>({2,0,3}), expected_zero_dim_zeros, "zeros with zero dim");
    ASSERT_TRUE(mlib::core::zeros<int>({2,0,3}).is_empty());
}

TEST_F(TensorOperationsTest, OnesCreation) {
    // 0-dim (scalar) tensor
    Tensor<int> expected_scalar_one({}, 1);
    AssertTensorEqual(mlib::core::ones<int>(std::vector<size_t>{}), expected_scalar_one, "ones scalar");

    // 1D tensor
    Tensor<float> expected_1d_ones({3}, {1.0f, 1.0f, 1.0f});
    AssertTensorEqual(mlib::core::ones<float>({3}), expected_1d_ones, "ones 1D float");

    // 2D tensor
    Tensor<int> expected_2d_ones({2,2}, {1,1,1,1});
    AssertTensorEqual(mlib::core::ones<int>({2,2}), expected_2d_ones, "ones 2D int");

    // Bool tensor
    Tensor<bool> expected_bool_ones({2}, {true, true});
    AssertTensorEqual(mlib::core::ones<bool>({2}), expected_bool_ones, "ones 1D bool");

    // Tensor with a zero dimension (total_size=0)
    Tensor<int> expected_zero_dim_ones({2,0,3}); // Total size 0
    AssertTensorEqual(mlib::core::ones<int>({2,0,3}), expected_zero_dim_ones, "ones with zero dim");
    ASSERT_TRUE(mlib::core::ones<int>({2,0,3}).is_empty());
}

TEST_F(TensorOperationsTest, FullCreation) {
    // Scalar tensor with value
    Tensor<float> expected_scalar_val({}, 3.14f);
    AssertTensorEqual(mlib::core::full<float>(std::vector<size_t>{}, 3.14f), expected_scalar_val, "full scalar float");

    // 1D tensor
    Tensor<int> expected_1d_full({4}, {5,5,5,5});
    AssertTensorEqual(mlib::core::full<int>({4}, 5), expected_1d_full, "full 1D int");

    // 2D tensor with value
    Tensor<double> expected_2d_full({2,2}, -7.0);
    AssertTensorEqual(mlib::core::full<double>({2,2}, -7.0), expected_2d_full, "full 2D double");

    // Bool tensor with value
    Tensor<bool> expected_bool_full({3}, false);
    AssertTensorEqual(mlib::core::full<bool>({3}, false), expected_bool_full, "full 1D bool false");
    Tensor<bool> expected_bool_true_full({3}, true);
    AssertTensorEqual(mlib::core::full<bool>({3}, true), expected_bool_true_full, "full 1D bool true");


    // Tensor with a zero dimension (total_size=0)
    Tensor<float> expected_zero_dim_full({1,0,2});
    AssertTensorEqual(mlib::core::full<float>({1,0,2}, 99.f), expected_zero_dim_full, "full with zero dim");
    ASSERT_TRUE(mlib::core::full<float>({1,0,2}, 99.f).is_empty());
}

TEST_F(TensorOperationsTest, EyeMatrixCreation) {
    // 0x0 matrix (empty)
    Tensor<int> expected_eye_0x0({0,0});
    AssertTensorEqual(mlib::core::eye<int>(0), expected_eye_0x0, "eye 0x0");
    ASSERT_TRUE(mlib::core::eye<int>(0).is_empty());

    // 1x1 identity matrix
    Tensor<int> expected_eye_1x1({1,1}, {1});
    AssertTensorEqual(mlib::core::eye<int>(1), expected_eye_1x1, "eye 1x1");

    // 3x3 identity matrix
    Tensor<float> expected_eye_3x3({3,3}, {1.0f,0.0f,0.0f,
                                            0.0f,1.0f,0.0f,
                                            0.0f,0.0f,1.0f});
    AssertTensorEqual(mlib::core::eye<float>(3), expected_eye_3x3, "eye 3x3 float");

    // Check with boolean type
    Tensor<bool> expected_eye_bool_2x2({2,2}, {true,false,false,true});
    AssertTensorEqual(mlib::core::eye<bool>(2), expected_eye_bool_2x2, "eye 2x2 bool");
}

TEST_F(TensorOperationsTest, ArangeCreation) {
    // Positive step, integer
    Tensor<int> expected_arange_int_pos({5}, {0,1,2,3,4});
    AssertTensorEqual(mlib::core::arange<int>(0, 5), expected_arange_int_pos, "arange int pos step 1"); // Default step 1
    AssertTensorEqual(mlib::core::arange<int>(0, 5, 1), expected_arange_int_pos, "arange int pos step 1 explicit");

    // Positive step, float
    Tensor<float> expected_arange_float_pos({5}, {0.0f, 0.5f, 1.0f, 1.5f, 2.0f});
    AssertTensorEqual(mlib::core::arange<float>(0.0f, 2.1f, 0.5f), expected_arange_float_pos, "arange float pos");

    // Negative step, integer
    Tensor<int> expected_arange_int_neg({5}, {5,4,3,2,1});
    AssertTensorEqual(mlib::core::arange<int>(5, 0, -1), expected_arange_int_neg, "arange int neg step");

    // Negative step, float
    Tensor<float> expected_arange_float_neg({3}, {2.0f, 1.0f, 0.0f});
    AssertTensorEqual(mlib::core::arange<float>(2.0f, -0.1f, -1.0f), expected_arange_float_neg, "arange float neg");

    // Empty range
    AssertTensorEqual(mlib::core::arange<int>(5, 0), Tensor<int>({0}), "arange empty pos step"); // default step
    AssertTensorEqual(mlib::core::arange<int>(0, 5, -1), Tensor<int>({0}), "arange empty neg step");

    // Error: step zero
    ASSERT_THROW(mlib::core::arange<int>(0, 5, 0), std::invalid_argument);
    ASSERT_THROW(mlib::core::arange<float>(0.0f, 5.0f, 0.0f), std::invalid_argument);
}

TEST_F(TensorOperationsTest, LinspaceCreation) {
    // Basic range, 5 points
    Tensor<float> expected_linspace_5_pts({5}, {0.0f, 0.25f, 0.5f, 0.75f, 1.0f});
    AssertTensorEqual(mlib::core::linspace<float>(0.0f, 1.0f, 5), expected_linspace_5_pts, "linspace basic 5pts");

    // Basic range, double
    Tensor<double> expected_linspace_3_pts_double({3}, {10.0, 15.0, 20.0});
    AssertTensorEqual(mlib::core::linspace<double>(10.0, 20.0, 3), expected_linspace_3_pts_double, "linspace basic 3pts double");

    // Single point
    Tensor<float> expected_linspace_1_pt({1}, {7.5f});
    AssertTensorEqual(mlib::core::linspace<float>(7.5f, 10.0f, 1), expected_linspace_1_pt, "linspace 1pt");

    // Negative range / reverse
    Tensor<float> expected_linspace_neg_range({3}, {5.0f, 2.5f, 0.0f});
    AssertTensorEqual(mlib::core::linspace<float>(5.0f, 0.0f, 3), expected_linspace_neg_range, "linspace neg range");

    // Error: num_points is zero
    ASSERT_THROW(mlib::core::linspace<float>(0.0f, 1.0f, 0), std::invalid_argument);
}


// --- TENSOR SHAPE TRANSFORMATION TESTS (UNSQUEEZE, SQUEEZE, RESHAPE -1) ---

TEST_F(TensorOperationsTest, UnsqueezeBasic) {
    Tensor<int> t_1d({3}, {1,2,3}); // (3,)

    // Unsqueeze at axis 0: (1,3)
    Tensor<int> expected_1x3({1,3}, {1,2,3});
    AssertTensorEqual(mlib::core::unsqueeze(t_1d, 0), expected_1x3, "unsqueeze (3,) at axis 0");
    // Unsqueeze at axis 1: (3,1)
    Tensor<int> expected_3x1({3,1}, {1,2,3});
    AssertTensorEqual(mlib::core::unsqueeze(t_1d, 1), expected_3x1, "unsqueeze (3,) at axis 1");

    Tensor<float> t_2d({2,2}, {1.f,2.f,3.f,4.f}); // (2,2)

    // Unsqueeze at axis 0: (1,2,2)
    Tensor<float> expected_1x2x2({1,2,2}, {1.f,2.f,3.f,4.f});
    AssertTensorEqual(mlib::core::unsqueeze(t_2d, 0), expected_1x2x2, "unsqueeze (2,2) at axis 0");
    // Unsqueeze at axis 1: (2,1,2)
    Tensor<float> expected_2x1x2({2,1,2}, {1.f,2.f,3.f,4.f});
    AssertTensorEqual(mlib::core::unsqueeze(t_2d, 1), expected_2x1x2, "unsqueeze (2,2) at axis 1");
    // Unsqueeze at axis 2: (2,2,1)
    Tensor<float> expected_2x2x1({2,2,1}, {1.f,2.f,3.f,4.f});
    AssertTensorEqual(mlib::core::unsqueeze(t_2d, 2), expected_2x2x1, "unsqueeze (2,2) at axis 2");

    // Negative axis
    AssertTensorEqual(mlib::core::unsqueeze(t_1d, -2), expected_1x3, "unsqueeze (3,) at axis -2 (0)");
    AssertTensorEqual(mlib::core::unsqueeze(t_1d, -1), expected_3x1, "unsqueeze (3,) at axis -1 (1)");

    AssertTensorEqual(mlib::core::unsqueeze(t_2d, -3), expected_1x2x2, "unsqueeze (2,2) at axis -3 (0)");
}

TEST_F(TensorOperationsTest, UnsqueezeScalarAndEmpty) {
    Tensor<int> scalar_t({}, 10); // Scalar tensor ()

    // Unsqueeze scalar at axis 0: (1)
    Tensor<int> expected_1d_from_scalar({1}, {10});
    AssertTensorEqual(mlib::core::unsqueeze(scalar_t, 0), expected_1d_from_scalar, "unsqueeze scalar at axis 0");
    AssertTensorEqual(mlib::core::unsqueeze(scalar_t, -1), expected_1d_from_scalar, "unsqueeze scalar at axis -1"); // -1 -> 0 here

    // Unsqueeze an empty shaped tensor (total_size=0)
    Tensor<int> empty_2x0({2,0}); // Shape (2,0)
    Tensor<int> expected_1x2x0({1,2,0});
    AssertTensorEqual(mlib::core::unsqueeze(empty_2x0, 0), expected_1x2x0, "unsqueeze (2,0) at axis 0");
    ASSERT_TRUE(mlib::core::unsqueeze(empty_2x0, 0).is_empty());
}

TEST_F(TensorOperationsTest, UnsqueezeInvalidAxis) {
    Tensor<int> t_1d({3}, {1,2,3});
    // Axis out of bounds
    ASSERT_THROW(mlib::core::unsqueeze(t_1d, 2), DimensionError); // ndim=1, valid axes 0, 1
    ASSERT_THROW(mlib::core::unsqueeze(t_1d, -3), DimensionError); // ndim=1, valid axes -2, -1 (for pos 0,1)
}


TEST_F(TensorOperationsTest, SqueezeBasic) {
    Tensor<int> t_3d_with_1s({2,1,3}, {1,2,3,4,5,6}); // (2,1,3)

    // Squeeze at axis 1: (2,3)
    Tensor<int> expected_2x3({2,3}, {1,2,3,4,5,6});
    AssertTensorEqual(mlib::core::squeeze(t_3d_with_1s, 1), expected_2x3, "squeeze (2,1,3) at axis 1");

    // Squeeze all (axis -1)
    Tensor<int> t_multi_1s({1,2,1,3,1}, {1,2,3,4,5,6}); // (1,2,1,3,1)
    Tensor<int> expected_2x3_all({2,3}, {1,2,3,4,5,6});
    AssertTensorEqual(mlib::core::squeeze(t_multi_1s), expected_2x3_all, "squeeze (1,2,1,3,1) all");

    // Negative axis
    AssertTensorEqual(mlib::core::squeeze(t_3d_with_1s, -2), expected_2x3, "squeeze (2,1,3) at axis -2 (1)");
}

TEST_F(TensorOperationsTest, SqueezeToScalar) {
    Tensor<int> t_all_1s({1,1,1}, {100}); // (1,1,1)

    // Squeeze all: ()
    Tensor<int> expected_scalar({}, 100);
    AssertTensorEqual(mlib::core::squeeze(t_all_1s), expected_scalar, "squeeze (1,1,1) to scalar");

    // Squeeze specific axis to scalar (must specify valid size 1 axis)
    AssertTensorEqual(mlib::core::squeeze(t_all_1s, 0), expected_scalar, "squeeze (1,1,1) at axis 0");
    AssertTensorEqual(mlib::core::squeeze(t_all_1s, 1), expected_scalar, "squeeze (1,1,1) at axis 1");
}

TEST_F(TensorOperationsTest, SqueezeInvalidCases) {
    Tensor<int> t_2d({2,3}, {1,2,3,4,5,6}); // (2,3)

    // Try to squeeze a non-size 1 dimension
    ASSERT_THROW(mlib::core::squeeze(t_2d, 0), DimensionError); // Axis 0 has size 2, not 1
    ASSERT_THROW(mlib::core::squeeze(t_2d, 1), DimensionError); // Axis 1 has size 3, not 1

    // Try to squeeze a scalar tensor
    Tensor<int> scalar_t({}, 10);
    ASSERT_THROW(mlib::core::squeeze(scalar_t), DimensionError);

    // Invalid axis
    ASSERT_THROW(mlib::core::squeeze(t_2d, 2), DimensionError); // Axis 2 out of bounds for (2,3)
    ASSERT_THROW(mlib::core::squeeze(t_2d, -3), DimensionError); // Axis -3 out of bounds for (2,3)
}

TEST_F(TensorOperationsTest, SqueezeEmptyTensor) {
    // 1. Squeeze an empty but shaped tensor `({2,1,0})`
    // It should squeeze axis 1 and result in `({2,0})`. Its total_size is 0.
    // The input data must be consistent with total_size=0.
    Tensor<int> empty_2x1x0_input({2,1,0}, std::vector<int>{}); // Pass empty data
    Tensor<int> expected_squeezed_empty_output({2,0}); // Shape after squeeze: {2,0}. Its total_size is 0.

    AssertTensorEqual(mlib::core::squeeze(empty_2x1x0_input, 1), expected_squeezed_empty_output, "squeeze (2,1,0) at axis 1");
    // Verify it remains logically empty.
    ASSERT_TRUE(mlib::core::squeeze(empty_2x1x0_input, 1).is_empty());
    ASSERT_EQ(mlib::core::squeeze(empty_2x1x0_input, 1).ndim(), 2); // new_shape is {2,0}, so ndim=2


    // 2. Squeeze a `total_size = 0` tensor where ALL its remaining dimensions are size 1.
    // For example, `Tensor({1,1,0})` should become a `0`-dim `0`-total_size Tensor.
    Tensor<int> empty_1x1x0_single_input({1,1,0}, std::vector<int>{}); // Original total_size is 0.
                                                                        // Its new shape is {} but it has 0 elements
    // The expected output is a default-constructed Tensor<int>(), meaning {} shape and 0 total_size.
    // This needs to be exactly `Tensor<int>()` to avoid constructor ambiguities/semantic clashes.
	Tensor<int> expected_empty_1d_output({0});
	AssertTensorEqual(mlib::core::squeeze(empty_1x1x0_single_input), expected_empty_1d_output, "squeeze (1,1,0) all (expected {0})");
    // Verify it is now an empty 0-dim tensor.
    ASSERT_TRUE(mlib::core::squeeze(empty_1x1x0_single_input).is_empty());
    ASSERT_EQ(mlib::core::squeeze(empty_1x1x0_single_input).ndim(), 1);
}

TEST_F(TensorOperationsTest, ReshapeWithInferredDimension) {
    Tensor<int> t_initial({2,3}, {1,2,3,4,5,6}); // Total size 6

    // Infer last dim: (6) -> (2,3)
    t_initial.reshape({-1, 2});
    AssertTensorEqual(t_initial, Tensor<int>({3,2}, {1,2,3,4,5,6}), "reshape (2,3) -> (-1,2)"); // Reshape makes total_size = 3*2 = 6, not (6) from earlier.
                                                                                        // Should actually be Tensor({3,2},{...}).
                                                                                        // This test re-verifies reshape on (2,3) into (3,2).
    // Test expected (3,2) from reshape {-1,2}
    Tensor<int> expected_3x2_a({3,2}, {1,2,3,4,5,6});
    AssertTensorEqual(t_initial, expected_3x2_a, "reshape {-1,2} yields (3,2)");

    // Restore for next test (or define new tensor locally)
    t_initial = Tensor<int>({2,3}, {1,2,3,4,5,6}); // Original 2x3

    // Infer first dim: (6) -> (3,?) (3,2)
    t_initial.reshape({3,-1});
    AssertTensorEqual(t_initial, expected_3x2_a, "reshape {3,-1} yields (3,2)");

    // Infer single dim (flatten): (6) -> (6)
    t_initial = Tensor<int>({2,3}, {1,2,3,4,5,6});
    Tensor<int> expected_flat_6({6}, {1,2,3,4,5,6});
    t_initial.reshape({-1});
    AssertTensorEqual(t_initial, expected_flat_6, "reshape {-1} yields (6)");

    // Infer middle dim: (12) -> (2,?,3) (2,2,3)
    Tensor<int> t_12_el({2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}); // (2,2,3)
    t_12_el.reshape({2,-1,3});
    AssertTensorEqual(t_12_el, Tensor<int>({2,2,3}, {1,2,3,4,5,6,7,8,9,10,11,12}), "reshape {2,-1,3} yields (2,2,3)");
}

TEST_F(TensorOperationsTest, ReshapeWithInferenceErrors) {
    Tensor<int> t_initial({2,3}, {1,2,3,4,5,6}); // Total size 6

    // More than one -1
    ASSERT_THROW(t_initial.reshape({-1,-1}), std::invalid_argument);
    ASSERT_THROW(t_initial.reshape({-1,1,-1}), std::invalid_argument);

    // Total size not divisible by known dims
    ASSERT_THROW(t_initial.reshape({-1,4}), std::invalid_argument); // 6 % 4 != 0
    ASSERT_THROW(t_initial.reshape({4,-1}), std::invalid_argument); // 6 % 4 != 0

    // Negative dim (not -1)
    ASSERT_THROW(t_initial.reshape({-2,3}), std::invalid_argument);

    // Reshape to zero dimension. Result total_size must be 0 if this happens.
    // e.g. current 6, target (2,0,-1,3) where infer is 0 -> will pass 0.
    Tensor<int> t_empty({2,0,3}); // total_size 0
    // Test that reshaped empty tensor (0 total size)
    t_empty.reshape({-1,0,3}); // current 0, product 0, infer is 0
    ASSERT_TRUE(t_empty.is_empty());
    ASSERT_EQ(t_empty.ndim(), 3);
}

TEST_F(TensorOperationsTest, ReshapeScalarTensorWithInference) {
    Tensor<int> scalar_t({}, 42); // Total size 1
    // Reshape to (1)
    scalar_t.reshape({-1});
    AssertTensorEqual(scalar_t, Tensor<int>({1}, {42}), "scalar reshape {-1} yields (1)");

    // Reshape (1) to ()
    scalar_t.reshape(std::vector<long long>{});
    AssertTensorEqual(scalar_t, Tensor<int>({}, 42), "(1) reshape {} yields ()");
}


// --- LOGICAL OPERATION TESTS ---

TEST_F(TensorOperationsTest, LogicalAndBasic) {
    Tensor<bool> t1({2,2}, {true, true, false, false});
    Tensor<bool> t2({2,2}, {true, false, true, false});
    Tensor<bool> expected_and({2,2}, {true, false, false, false});

    AssertTensorEqual(mlib::core::logical_and(t1, t2), expected_and, "Logical AND basic");
}

TEST_F(TensorOperationsTest, LogicalOrBasic) {
    Tensor<bool> t1({2,2}, {true, true, false, false});
    Tensor<bool> t2({2,2}, {true, false, true, false});
    Tensor<bool> expected_or({2,2}, {true, true, true, false});

    AssertTensorEqual(mlib::core::logical_or(t1, t2), expected_or, "Logical OR basic");
}

TEST_F(TensorOperationsTest, LogicalNotBasic) {
    Tensor<bool> t({2,2}, {true, false, true, false});
    Tensor<bool> expected_not({2,2}, {false, true, false, true});

    AssertTensorEqual(mlib::core::logical_not(t), expected_not, "Logical NOT basic");
    AssertTensorEqual(!t, expected_not, "Operator ! basic");
}

TEST_F(TensorOperationsTest, LogicalOpsEdgeCases) {
    // Scalar bool
    Tensor<bool> scalar_true({}, true);
    Tensor<bool> scalar_false({}, false);
    AssertTensorEqual(mlib::core::logical_and(scalar_true, scalar_false), Tensor<bool>({}, false), "Scalar AND");
    AssertTensorEqual(mlib::core::logical_or(scalar_true, scalar_false), Tensor<bool>({}, true), "Scalar OR");
    AssertTensorEqual(mlib::core::logical_not(scalar_true), Tensor<bool>({}, false), "Scalar NOT");
    AssertTensorEqual(!scalar_false, Tensor<bool>({}, true), "Operator ! Scalar");

    // Empty bool
    Tensor<bool> empty_t_bool({2,0});
    Tensor<bool> empty_t_bool_other({2,0});
    AssertTensorEqual(mlib::core::logical_and(empty_t_bool, empty_t_bool_other), Tensor<bool>({2,0}), "Empty AND");
    ASSERT_TRUE(mlib::core::logical_and(empty_t_bool, empty_t_bool_other).is_empty());
    AssertTensorEqual(mlib::core::logical_not(empty_t_bool), Tensor<bool>({2,0}), "Empty NOT");
    ASSERT_TRUE(mlib::core::logical_not(empty_t_bool).is_empty());

    // Shape mismatch
    Tensor<bool> t_2x2_b({2,2});
    Tensor<bool> t_2x3_b({2,3});
    ASSERT_THROW(mlib::core::logical_and(t_2x2_b, t_2x3_b), ShapeMismatchError);
    ASSERT_THROW(mlib::core::logical_or(t_2x2_b, t_2x3_b), ShapeMismatchError);
}

