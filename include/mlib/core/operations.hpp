#ifndef MLIB_CORE_OPERATIONS_HPP
#define MLIB_CORE_OPERATIONS_HPP

#include "tensor.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mlib
{
namespace core
{

// --- Element-wise Binary Operations ---

/**
 * @brief Performs element-wise addition of two tensors.
 *
 * Both tensors must have the exact same shape.
 *
 * @tparam T The data type of the tensor elements.
 * @param a The first tensor.
 * @param b The second tensor.
 * @return A new tensor containing the element-wise sum of a and b.
 * @throw ShapeMismatchError if the shapes of a and b are not identical.
 * @throw MlibException if either tensor is empty but not a scalar. (Scalars can be handled by broadcasting later)
 */
template <typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b)
{
	if(a.get_shape() != b.get_shape())
	{
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "add");
	}

	// For element-wise operations, an empty tensor (total_size 0) that isn't a scalar is problematic
    // or results in an empty tensor.
    // If shapes are equal, and one is empty (e.g. {2,0,3}), the other is too.
    // The result will also be an empty tensor of that shape.
	if(a.is_empty() && !a.is_scalar()) // True if shape has a 0-dim, or default constructed
	{
		// If shapes match, and a is empty (e.g. {0,2}), then b is also {0,2}
        // and the result is also an empty tensor of that shape.
		return Tensor<T>(a.get_shape()); // Create an empty tensor of the same shape.
	}

	// If both are scalars, result is a scalar
	if(a.is_scalar()) // and b.is_scalar() because shapes must match
	{
		return Tensor<T>({}, a.data()[0] + b.data()[0]);
	}

	// At this point, shapes are identical and non-empty (or scalar)
    // We can rely on the data() pointers being valid and sizes matching.
	Tensor<T> result(a.get_shape()); // Result tensor with the same shape, default-initialized (to zero for numeric T)
	
	// Check for contiguous data for potential optimization.
    // For now, simple loop.
    // We assume both a and b are contiguous for this simple implementation,
    // or we access them element by element using operator().
    // Since they have the same shape and default constructed Tensors are contiguous,
    // a direct data pointer operation is faster IF WE KNOW THEY ARE CONTIGUOUS
    // and have the same strides. For a generic `add`, if we don't make these assumptions,
    // we should iterate logically. But since we only support contiguous, this is fine for now.
	if(a.get_total_size() > 0)
	{
		std::transform(
			a.data(), a.data() + a.get_total_size(), // Source 1
			b.data(), // Source 2
			result.data(), // Destination
			std::plus<T>()); // Operation
	}
	// If total_size is 0 (e.g. shape {2,0,3}), transform won't do anything, which is correct.
    // result is already an empty tensor of that shape.

	return result;
}

/**
 * @brief Adds a scalar value to each element of a tensor.
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar value (must be an arithmetic type).
 * @param tensor The input tensor.
 * @param scalar_value The scalar value to add.
 * @return A new tensor with the scalar added to each element.
 *         The result tensor will have elements of type T.
 */
template <typename T, typename S>
Tensor<T> add(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor addition must be an arithmetic type.");
	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, tensor.data()[0] + static_cast<T>(scalar_value));

	Tensor<T> result(tensor.get_shape());
	T casted_scalar = static_cast<T>(scalar_value);

	auto scalar_plus_op = [casted_scalar](T tensor_element_val){ return tensor_element_val + casted_scalar; };

	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		scalar_plus_op);

	return result;
}

/**
 * @brief Adds each element of a tensor to a scalar value (scalar + tensor_element).
 *        This is commutative with add(tensor, scalar_value).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar value (must be an arithmetic type).
 * @param scalar_value The scalar value.
 * @param tensor The input tensor.
 * @return A new tensor with each tensor element added to the scalar.
 *         The result tensor will have elements of type T.
 */
template <typename T, typename S>
Tensor<T> add(S scalar_value, const Tensor<T>& tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor addition must be an arithmetic type.");
	return add(tensor, scalar_value);
}

/**
 * @brief Overloads the + operator for element-wise tensor addition.
 * @see mlib::core::add(const Tensor<T>&, const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b)
{
	return add(a, b);
}

/**
 * @brief Overloads the + operator for adding a scalar to a tensor.
 * (tensor + scalar)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>> // SFINAE to ensure S is arithmetic
                                                                // and to help with overload resolution if T could also be arithmetic
Tensor<T> operator+(const Tensor<T>& tensor, S scalar_value)
{
	return add(tensor, scalar_value);
}

/**
 * @brief Overloads the + operator for adding a tensor to a scalar.
 * (scalar + tensor) - addition is commutative.
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator+(S scalar_value, const Tensor<T>& tensor)
{
	return add(scalar_value, tensor);
}

/**
 * @brief Returns a copy of the tensor (identity operation for unary +).
 *
 * @tparam T The data type of the tensor elements.
 * @param a The tensor.
 * @return A new tensor that is a copy of a.
 */
template <typename T>
Tensor<T> identity(const Tensor<T>& a) {
    // The copy constructor of Tensor<T> will handle all cases:
    // - regular tensor
    // - scalar tensor
    // - empty tensor (shape with 0-dim)
    // - default-constructed (0-dim, 0-size) tensor
    return Tensor<T>(a); // Simply uses the Tensor's copy constructor
}

/**
 * @brief Overloads the unary + operator. Returns a copy of the tensor.
 * @see mlib::core::identity(const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator+(const Tensor<T>& a) {
    return identity(a);
}


/**
 * @brief Performs element-wise subtraction of two tensors (a - b).
 *
 * Both tensors must have the exact same shape.
 *
 * @tparam T The data type of the tensor elements.
 * @param a The tensor from which to subtract (minuend).
 * @param b The tensor to subtract (subtrahend).
 * @return A new tensor containing the element-wise difference of a and b.
 * @throw ShapeMismatchError if the shapes of a and b are not identical.
 */
template <typename T>
Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "subtract");

	if(a.is_empty() && !a.is_scalar())
	{
		return Tensor<T>(a.get_shape());
	}

	if(a.is_scalar())
	{
		return Tensor<T>({}, a.data()[0] - b.data()[0]);
	}

	Tensor<T> result(a.get_shape());

	if(a.get_total_size() > 0)
	{
		std::transform(
			a.data(), a.data() + a.get_total_size(),
			b.data(),
			result.data(),
			std::minus<T>());
	}

	return result;
}

/**
 * @brief Subtracts a scalar value from each element of a tensor (tensor_element - scalar).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar value (must be an arithmetic type).
 * @param tensor The input tensor (minuend).
 * @param scalar_value The scalar value to subtract (subtrahend).
 * @return A new tensor with the scalar subtracted from each element.
 *         The result tensor will have elements of type T.
 */
template <typename T, typename S>
Tensor<T> subtract(const Tensor<T> tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor subraction must be an arithmetic type.");
	
	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, tensor.data()[0] - scalar_value);

	Tensor<T> result(tensor.get_shape());
	T casted_scalar = static_cast<T>(scalar_value);

	auto scalar_minus_op = [casted_scalar](T tensor_element_val) { return tensor_element_val - casted_scalar; };

	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		scalar_minus_op);

	return result;
}

/**
 * @brief Subtracts each element of a tensor from a scalar value (scalar - tensor_element).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar value (must be an arithmetic type).
 * @param scalar_value The scalar value (minuend).
 * @param tensor The input tensor (subtrahend).
 * @return A new tensor with each tensor element subtracted from the scalar.
 *         The result tensor will have elements of type T.
 */
template <typename T, typename S>
Tensor<T> subtract(S scalar_value, const Tensor<T> tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor subraction must be an arithmetic type.");
	
	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, scalar_value - tensor.data()[0]);

	Tensor<T> result(tensor.get_shape());
	T casted_scalar = static_cast<T>(scalar_value);

	auto scalar_minus_op = [casted_scalar](T tensor_element_val) { return casted_scalar - tensor_element_val; };

	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		scalar_minus_op);

	return result;
}

/**
 * @brief Overloads the - operator for element-wise tensor subtraction.
 * @see mlib::core::subtract(const Tensor<T>&, const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b)
{
	return subtract(a, b);
}

/**
 * @brief Overloads the - operator for subtracting a scalar from a tensor.
 * (tensor - scalar)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator-(const Tensor<T>& tensor, S scalar_value)
{
	return subtract(tensor, scalar_value);
}

/**
 * @brief Overloads the - operator for subtracting a tensor from a scalar.
 * (scalar - tensor)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator-(S scalar_value, const Tensor<T>& tensor)
{
	return subtract(scalar_value, tensor);
}

/**
 * @brief Performs element-wise negation of a tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param a The tensor to negate.
 * @return A new tensor containing the element-wise negation of a.
 */
template <typename T>
Tensor<T> negate(const Tensor<T>& a)
{
	if (a.is_empty() && !a.is_scalar())
		return Tensor<T>(a.get_shape());

	if(a.is_scalar())
		return Tensor<T>({}, -a.data()[0]);

	Tensor<T> result(a.get_shape());

	if(a.get_total_size() > 0)
	{
		// Check if T is an unsigned type, as std::negate might not be what you want
        // or might cause issues. Standard library std::negate works fine for signed types.
        // For unsigned types, negation typically involves modular arithmetic (e.g., 2's complement),
        // which std::negate will do (0 - x).
        //static_assert(std::is_signed_v<T> || std::is_floating_point_v<T>,
        //              "Negate operation is typically for signed or floating point types. "
        //              "For unsigned types, ensure behavior is intended (e.g., 0-x).");

		std::transform(
			a.data(), a.data() + a.get_total_size(),
			result.data(),
			std::negate<T>());
	}
	
	return result;
}

/**
 * @brief Overloads the unary - operator for element-wise tensor negation.
 * @see mlib::core::negate(const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator-(const Tensor<T> a)
{
	return negate(a);
}


/**
 * @brief Performs element-wise multiplication of two tensors.
 *
 * Both tensors must have the exact same shape.
 *
 * @tparam T The data type of the tensor elements.
 * @param a The first tensor.
 * @param b The second tensor.
 * @return A new tensor containing the element-wise product of a and b.
 * @throw ShapeMismatchError if the shapes of a and b are not identical.
 */
template <typename T>
Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "multiply");

	if(a.is_empty() && !a.is_scalar())
		return Tensor<T>(a.get_shape());

	if(a.is_scalar())
		return Tensor<T>({}, a.data()[0] * b.data()[0]);

	Tensor<T> result(a.get_shape());

	if(a.get_total_size() > 1)
	{
		std::transform(
			a.data(), a.data() + a.get_total_size(), // Source 1
			b.data(), // Source 2
			result.data(), // Destination
			std::multiplies<T>()); // Operation
	}

	return result;
}

/**
 * @brief Multiplies each element of a tensor by a scalar value (tensor_element * scalar).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar value (must be an arithmetic type).
 * @param tensor The input tensor.
 * @param scalar_value The scalar value to multiply by.
 * @return A new tensor with each element multiplied by the scalar.
 *         The result tensor will have elements of type T.
 */
template <typename T, typename S>
Tensor<T> multiply(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor multiplication must be an arithmetic type.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, tensor.data()[0] * static_cast<T>(scalar_value));

	Tensor<T> result(tensor.get_shape());
	T casted_scalar = static_cast<T>(scalar_value);

	auto scalar_multiply_op = [casted_scalar](T tensor_element_val) { return tensor_element_val * casted_scalar; };

	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		scalar_multiply_op);

	return result;
}

/**
 * @brief Multiplies a scalar value by each element of a tensor (scalar * tensor_element).
 *        This is commutative with multiply(tensor, scalar_value).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar value (must be an arithmetic type).
 * @param scalar_value The scalar value.
 * @param tensor The input tensor.
 * @return A new tensor with the scalar multiplied by each tensor element.
 *         The result tensor will have elements of type T.
 */
template <typename T, typename S>
Tensor<T> multiply(S scalar_value, const Tensor<T>& tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor multiplication must be an arithmetic type.");
	return multiply(tensor, scalar_value);
}

/**
 * @brief Overloads the * operator for element-wise tensor multiplication.
 * @see mlib::core::multiply(const Tensor<T>&, const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b)
{
	// IMPORTANT: This operator* is for element-wise multiplication.
    // For matrix multiplication (dot product), a separate function like matmul()
    // or a different operator/method convention is typically used.
	return multiply(a, b);
}

/**
 * @brief Overloads the * operator for multiplying a tensor by a scalar.
 * (tensor * scalar)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator*(const Tensor<T>& tensor, S scalar_value)
{
	return multiply(tensor, scalar_value);
}

/**
 * @brief Overloads the * operator for multiplying a scalar by a tensor.
 * (scalar * tensor)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator*(S scalar_value, const Tensor<T>& tensor)
{
	return multiply(scalar_value, tensor);
}

/**
 * @brief Performs element-wise division of two tensors (a / b).
 *
 * Both tensors must have the exact same shape.
 * For integral types, this is integer division.
 * For floating-point types, division by zero will result in +/- infinity or NaN
 * as per IEEE 754 standards, unless checks are added to throw an exception.
 *
 * @tparam T The data type of the tensor elements.
 * @param a The dividend tensor.
 * @param b The divisor tensor.
 * @return A new tensor containing the element-wise quotient of a and b.
 * @throw ShapeMismatchError if the shapes of a and b are not identical.
 * @throw std::overflow_error (or similar) if division by zero is attempted with integral types.
 *        (Behavior for floating point types depends on IEEE 754 and compiler).
 */
template <typename T>
Tensor<T> divide(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "divide");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<T>(a.get_shape());

	if (a.is_scalar())
	{
		if constexpr (std::is_integral_v<T>)
		{
			if (b.data()[0] == static_cast<T>(0))
			{
				throw std::overflow_error("Division by zero in scalar tensor division.");
			}
		}
		// For floating point, b.data()[0] == 0.0 will result in inf/NaN by default
		return Tensor<T>({}, a.data()[0] / b.data()[0]);
	}

	Tensor<T> result(a.get_shape());

	if(a.get_total_size() > 0)
	{
		// For integral types, direct std::divides can cause runtime error (or UB) on division by zero.
        // For floating point types, std::divides handles division by zero according to IEEE 754 (yields inf/NaN).
		if constexpr (std::is_integral_v<T>)
		{
			// Manual loop to check for division by zero for integral types
			const T* ptr_a = a.data();
			const T* ptr_b = b.data();
			T* ptr_res = result.data();
			for (size_t i = 0; i < a.get_total_size(); i++)
			{
				if (ptr_b[i] == static_cast<T>(0))
				{
					throw std::overflow_error("Element-wise division by zero encountered for integral type at index " + std::to_string(i));
				}

				ptr_res[i] = ptr_a[i] / ptr_b[i];
			}
		}
		else
		{
			// For floating point types, std::divides is fine and follows IEEE 754.
			std::transform(
				a.data(), a.data() + a.get_total_size(), // Source 1
				b.data(), // Source 2
				result.data(), // Destination
				std::divides<T>()); // Operation
		}
	}

	return result;
}

/**
 * @brief Divides each element of a tensor by a scalar value (tensor_element / scalar_divisor).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar divisor (must be an arithmetic type).
 * @param tensor The dividend tensor.
 * @param scalar_divisor The scalar value to divide by.
 * @return A new tensor with each element divided by the scalar.
 *         The result tensor will have elements of type T.
 * @throw std::overflow_error if scalar_divisor is zero and T is an integral type.
 *        For floating-point T, division by zero yields inf/NaN.
 */
template <typename T, typename S>
Tensor<T> divide(const Tensor<T>& tensor, S scalar_divisor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for tensor division must be an arithmetic type.");

	if constexpr (std::is_integral_v<T> || std::is_integral_v<S>)
	{
		if (scalar_divisor == static_cast<S>(0))
			throw std::overflow_error("Division by zero: scalar_divisor is zero for Tensor / Scalar operation.");
	}
	// For floating point T and S, S == 0.0 will result in inf/NaN by default.

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());
	
	if (tensor.is_scalar())
	{
		// The check for scalar_divisor == 0 already happened for integral types.
        // For float, direct division handles inf/NaN.
		return Tensor<T>({}, tensor.data()[0] / scalar_divisor);
	}

	Tensor<T> result(tensor.get_shape());
	T casted_divisor = static_cast<T>(scalar_divisor);

	// If T is integral, casted_divisor being 0 was already caught.
    // If T is float, casted_divisor being 0.0 is handled by IEEE 754.
	auto scalar_divide_op = [casted_divisor](T tensor_element_val) { return tensor_element_val / casted_divisor; };

	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		scalar_divide_op);

	return result;
}

/**
 * @brief Divides a scalar value by each element of a tensor (scalar_dividend / tensor_element).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar dividend (must be an arithmetic type).
 * @param scalar_dividend The scalar value (dividend).
 * @param tensor_divisor The tensor whose elements act as divisors.
 * @return A new tensor where each element is scalar_dividend / tensor_element.
 *         The result tensor will have elements of type T.
 * @throw std::overflow_error if any element in tensor_divisor is zero and T is an integral type.
 *        For floating-point T, division by zero yields inf/NaN.
 */
template <typename T, typename S>
Tensor<T> divide(S scalar_dividend, const Tensor<T>& tensor_divisor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar dividend for tensor division msut be an arithmetic type.");

	if (tensor_divisor.is_empty() && !tensor_divisor.is_scalar())
		return Tensor<T>(tensor_divisor.get_shape());

	if (tensor_divisor.is_scalar())
	{
		T divisor_element = tensor_divisor.data()[0];
		if constexpr (std::is_integral_v<S>)
		{
			if (divisor_element == static_cast<T>(0))
				throw std::overflow_error("Division by zero: scalar tensor_divisor element is zero for Scalar / Tensor operation.");
		}

		return Tensor<T>({}, static_cast<T>(scalar_dividend) / divisor_element);
	}

	Tensor<T> result(tensor_divisor.get_shape());
	T casted_dividend = static_cast<T>(scalar_dividend);

	if (tensor_divisor.get_total_size() > 0)
	{
		if constexpr (std::is_integral_v<T>)
		{
			const T* ptr_divisor = tensor_divisor.data();
			T* ptr_res = result.data();
			for (size_t i = 0; i < tensor_divisor.get_total_size(); i++)
			{
				if (ptr_divisor[i] == static_cast<T>(0))
					throw std::overflow_error("Element-wise division by zero in tensor_divisor for Scalar / Tensor (integral type) at index " + std::to_string(i));

				ptr_res[i] = casted_dividend / ptr_divisor[i];
			}
		}
		else
		{
			// For floating point types, std::divides with a lambda is fine
			auto scalar_divide_op = [casted_dividend](T tensor_element_val) { return casted_dividend / tensor_element_val; };

			std::transform(
				tensor_divisor.data(), tensor_divisor.data() + tensor_divisor.get_total_size(),
				result.data(),
				scalar_divide_op);

		}

	}

	return result;
}

/**
 * @brief Overloads the / operator for element-wise tensor division.
 * @see mlib::core::divide(const Tensor<T>&, const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b)
{
	return divide(a, b);
}

/**
 * @brief Overloads the / operator for dividing a tensor by a scalar.
 * (tensor / scalar)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator/(const Tensor<T>& tensor, S scalar_divisor)
{
	return divide(tensor, scalar_divisor);
}

/**
 * @brief Overloads the / operator for dividing a scalar by a tensor.
 * (scalar / tensor)
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be an arithmetic type).
 */
template <typename T, typename S,
		 typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<T> operator/(S scalar_dividend, const Tensor<T>& tensor_divisor)
{
	return divide(scalar_dividend, tensor_divisor);
}

/**
 * @brief Computes the element-wise exponential (e^x) of a tensor.
 *
 * @tparam T Data type of the tensor elements (typically float or double).
 * @param tensor The input tensor.
 * @return A new tensor containing e^x for each element of the input.
 */
template <typename T>
Tensor<T> exp(const Tensor<T>& tensor)
{
	static_assert(std::is_floating_point_v<T>, "exp operation requires floating-point tensor elements.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, std::exp(tensor.data()[0]));

	Tensor<T> result(tensor.get_shape());
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		[](T val) { return std::exp(val); });

	return result;
}

/**
 * @brief Computes the element-wise natural logarithm (ln(x)) of a tensor.
 *        Input elements must be positive.
 *
 * @tparam T Data type of the tensor elements (typically float or double).
 * @param tensor The input tensor.
 * @return A new tensor containing ln(x) for each element.
 * @throw std::domain_error if any input element is not positive for log.
 *        (Alternatively, can let it produce -inf/NaN as per IEEE 754 for log(0) or log(-ve))
 */
template <typename T>
Tensor<T> log(const Tensor<T>& tensor)
{
	static_assert(std::is_floating_point_v<T>, "log operation requires floating-point tensor elements.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, std::log(tensor.data()[0])); // std::log handles 0 and -ve according to IEEE 754

	Tensor<T> result(tensor.get_shape());
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		[](T val) { return std::log(val); }); // std::log handles 0 (-inf) and -ve (NaN)

	return result;
}

/**
 * @brief Computes the element-wise square root of a tensor.
 *        Input elements must be non-negative.
 *
 * @tparam T Data type of the tensor elements (typically float or double).
 * @param tensor The input tensor.
 * @return A new tensor containing sqrt(x) for each element.
 * @throw std::domain_error if any input element is negative. (Or let it produce NaN)
 */
template <typename T>
Tensor<T> sqrt(const Tensor<T>& tensor)
{
	static_assert(std::is_floating_point_v<T>, "sqrt operation requires floating-point tensor elements.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, std::sqrt(tensor.data()[0])); // std::sqrt handles -ve (NaN)

	Tensor<T> result(tensor.get_shape());
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		[](T val) { return std::sqrt(val); }); // std::sqrt for negative input yields NaN

	return result;
}

/**
 * @brief Computes the element-wise absolute value of a tensor.
 *
 * @tparam T Data type of the tensor elements (integral or floating-point).
 * @param tensor The input tensor.
 * @return A new tensor containing |x| for each element.
 */
template <typename T>
Tensor<T> abs(const Tensor<T>& tensor)
{
	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<T>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<T>({}, std::abs(tensor.data()[0]));

	Tensor<T> result(tensor.get_shape());
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		result.data(),
		[](T val) { return std::abs(val); });

	return result;
}

// --- Element-wise Comparison Operations ---

// --- equal (==) ---

/**
 * @brief Performs element-wise equality comparison of two tensors.
 *
 * @tparam T Data type of the elements in the input tensors.
 * @param a The first tensor.
 * @param b The second tensor.
 * @return A Tensor<bool> where each element is true if a_i == b_i, false otherwise.
 * @throw ShapeMismatchError if shapes of a and b are not identical.
 */
template <typename T>
Tensor<bool> equal(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "equal (tensor-tensor)");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<bool>(a.get_shape());

	if (a.is_scalar())
		return Tensor<bool>({}, a.data()[0] == b.data()[0]);

	std::vector<bool> result;
	std::transform(
		a.data(), a.data() + a.get_total_size(),
		b.data(),
		std::back_inserter(result),
		std::equal_to<T>()); // Or [](T val_a, T val_b) { return val_a == val_b; });

	return Tensor<bool>(a.get_shape(), result);
}

/**
 * @brief Performs element-wise equality comparison between a tensor and a scalar.
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be arithmetic).
 * @param tensor The tensor.
 * @param scalar_value The scalar value.
 * @return A Tensor<bool> where each element is true if tensor_i == scalar_value.
 */
template <typename T, typename S>
Tensor<bool> equal(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for 'equal' comparison must be an arithmetic type.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, tensor.data()[0] == static_cast<T>(scalar_value));

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return val == casted_scalar; });

	return Tensor<bool>(tensor.get_shape(), result);
}

/**
 * @brief Performs element-wise equality comparison between a scalar and a tensor.
 * (Symmetric to tensor == scalar).
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be arithmetic).
 * @param scalar_value The scalar value.
 * @param tensor The tensor.
 * @return A Tensor<bool> where each element is true if scalar_value == tensor_i.
 */
template <typename T, typename S>
Tensor<bool> equal(S scalar_value, const Tensor<T>& tensor)
{
	return equal(tensor, scalar_value);
}

// Operator overloads for ==
template <typename T>
Tensor<bool> operator==(const Tensor<T>& a, const Tensor<T>& b)
{
	return equal(a, b);
}

template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator==(const Tensor<T>& a, S scalar_value)
{
	return equal(a, scalar_value);
}

template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator==(S scalar_value, const Tensor<T>& a)
{
	return equal(scalar_value, a);
}

// --- not_equal (!=) ---

/**
 * @brief Performs element-wise inequality comparison of two tensors.
 *
 * @tparam T Data type of the elements in the input tensors.
 * @param a The first tensor.
 * @param b The second tensor.
 * @return A Tensor<bool> where each element is true if a_i != b_i, false otherwise.
 * @throw ShapeMismatchError if shapes of a and b are not identical.
 */
template <typename T>
Tensor<bool> not_equal(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "not_equal (tensor-tensor)");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<bool>(a.get_shape());

	if (a.is_scalar())
		return Tensor<bool>({}, a.data()[0] != b.data()[0]);

	std::vector<bool> result;
	result.reserve(a.get_total_size());
	std::transform(
		a.data(), a.data() + a.get_total_size(),
		b.data(),
		std::back_inserter(result),
		std::not_equal_to<T>());// Or [](T val_a, T val_b) { return val_a != val_b; });

	return Tensor<bool>(a.get_shape(), result);
}

/**
 * @brief Performs element-wise inequality comparison between a tensor and a scalar.
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be arithmetic).
 * @param tensor The tensor.
 * @param scalar_value The scalar value.
 * @return A Tensor<bool> where each element is true if tensor_i != scalar_value.
 */
template <typename T, typename S>
Tensor<bool> not_equal(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar value for 'not_equal' comparison must be an arithmetic type.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, tensor.data()[0] != static_cast<T>(scalar_value));

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return val != casted_scalar; });

	return Tensor<bool>(tensor.get_shape(), result);
}

/**
 * @brief Performs element-wise inequality comparison between a scalar and a tensor.
 * (Symmetric to tensor != scalar).
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be arithmetic).
 * @param scalar_value The scalar value.
 * @param tensor The tensor.
 * @return A Tensor<bool> where each element is true if scalar_value != tensor_i.
 */
template <typename T, typename S>
Tensor<bool> not_equal(S scalar_value, const Tensor<T>& tensor)
{
	return not_equal(tensor, scalar_value);
}

// Operator overloads for !=
template <typename T>
Tensor<bool> operator!=(const Tensor<T>& a, const Tensor<T>& b)
{
	return not_equal(a, b);
}

template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator!=(const Tensor<T>& a, S scalar_value)
{
	return not_equal(a, scalar_value);
}

template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator!=(S scalar_value, const Tensor<T>& a)
{
	return not_equal(a, scalar_value);
}

// --- greater (>) ---

/**
 * @brief Performs element-wise "greater than" comparison of two tensors (a > b).
 *
 * @tparam T Data type of the elements in the input tensors.
 * @param a The first tensor (left-hand side).
 * @param b The second tensor (right-hand side).
 * @return A Tensor<bool> where each element is true if a_i > b_i, false otherwise.
 * @throw ShapeMismatchError if shapes of a and b are not identical.
 */
template <typename T>
Tensor<bool> greater(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "greater (tensor-tensor)");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<bool>(a.get_shape());

	if (a.is_scalar())
		return Tensor<bool>({}, a.data()[0] > b.data()[0]);

	std::vector<bool> result;
	result.reserve(a.get_total_size());
	std::transform(
		a.data(), a.data() + a.get_total_size(),
		b.data(),
		std::back_inserter(result),
		std::greater<T>());

	return Tensor<bool>(a.get_shape(), result);
}

/**
 * @brief Performs element-wise "greater than" comparison between a tensor and a scalar (tensor > scalar).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be arithmetic).
 * @param tensor The tensor (left-hand side).
 * @param scalar_value The scalar value (right-hand side).
 * @return A Tensor<bool> where each element is true if tensor_i > scalar_value.
 */
template <typename T, typename S>
Tensor<bool> greater(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'greater' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, tensor.data()[0] > static_cast<T>(scalar_value));

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_value = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_value](T val) { return val > casted_value; });

	return Tensor<bool>(tensor.get_shape(), result);
}

/**
 * @brief Performs element-wise "greater than" comparison between a scalar and a tensor (scalar > tensor).
 *
 * @tparam T Data type of the tensor elements.
 * @tparam S Data type of the scalar (must be arithmetic).
 * @param scalar_value The scalar value (left-hand side).
 * @param tensor The tensor (right-hand side).
 * @return A Tensor<bool> where each element is true if scalar_value > tensor_i.
 */
template <typename T, typename S>
Tensor<bool> greater(S scalar_value, const Tensor<T>& tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'greater' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, static_cast<T>(scalar_value) > tensor.data()[0]);

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_value = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_value](T val) { return casted_value > val; });

	return Tensor<bool>(tensor.get_shape(), result);
}

// Operator >
template <typename T>
Tensor<bool> operator>(const Tensor<T>& a, const Tensor<T>& b)
{
	return greater(a, b);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator>(const Tensor<T>& tensor, S scalar_value)
{
	return greater(tensor, scalar_value);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator>(S scalar_value, const Tensor<T>& tensor)
{
	return greater(scalar_value, tensor);
}

// --- greater_equal (>=) ---

/**
 * @brief Performs element-wise "greater than or equal to" comparison of two tensors (a >= b).
 * @copydetails mlib::core::greater(const Tensor<T>&, const Tensor<T>&)
 * @return A Tensor<bool> where each element is true if a_i >= b_i, false otherwise.
 */
template <typename T>
Tensor<bool> greater_equal(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "greater_equal (tensor-tensor)");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<bool>(a.get_shape());

	if (a.is_scalar())
		return Tensor<bool>({}, a.data()[0] >= b.data()[0]);

	std::vector<bool> result;
	result.reserve(a.get_total_size());
	std::transform(
		a.data(), a.data() + a.get_total_size(),
		b.data(),
		std::back_inserter(result),
		[](T val_a, T val_b) { return val_a >= val_b; });

	return Tensor<bool>(a.get_shape(), result);
}

/**
 * @brief Performs element-wise "greater than or equal to" comparison between a tensor and a scalar (tensor >= scalar).
 * @copydetails mlib::core::greater(const Tensor<T>&, S)
 * @return A Tensor<bool> where each element is true if tensor_i >= scalar_value.
 */
template <typename T, typename S>
Tensor<bool> greater_equal(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'greater_equal' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, tensor.data()[0] >= static_cast<T>(scalar_value));

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return val >= casted_scalar; });

	return Tensor<bool>(tensor.get_shape(), result);
}

/**
 * @brief Performs element-wise "greater than or equal to" comparison between a scalar and a tensor (scalar >= tensor).
 * @copydetails mlib::core::greater(S, const Tensor<T>&)
 * @return A Tensor<bool> where each element is true if scalar_value >= tensor_i.
 */
template <typename T, typename S>
Tensor<bool> greater_equal(S scalar_value, const Tensor<T>& tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'greater_equal' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, static_cast<T>(scalar_value) >= tensor.data()[0]);

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return casted_scalar >= val; });

	return Tensor<bool>(tensor.get_shape(), result);
}

// Operator >=
template <typename T>
Tensor<bool> operator>=(const Tensor<T>& a, const Tensor<T>& b)
{
	return greater_equal(a, b);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator>=(const Tensor<T>& tensor,S scalar_value)
{
	return greater_equal(tensor, scalar_value);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator>=(S scalar_value,const Tensor<T>& tensor)
{
	return greater_equal(scalar_value, tensor);
}

// --- less (<)

/**
 * @brief Performs element-wise "less than" comparison of two tensors (a < b).
 * @copydetails mlib::core::greater(const Tensor<T>&, const Tensor<T>&)
 * @return A Tensor<bool> where each element is true if a_i < b_i, false otherwise.
 */
template <typename T>
Tensor<bool> less(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "less (tensor-tensor)");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<bool>(a.get_shape());

	if (a.is_scalar())
		return Tensor<bool>({}, a.data()[0] < b.data()[0]);

	std::vector<bool> result;
	result.reserve(a.get_total_size());
	std::transform(
		a.data(), a.data() + a.get_total_size(),
		b.data(),
		std::back_inserter(result),
		std::less<T>());

	return Tensor<bool>(a.get_shape(), result);
}

/**
 * @brief Performs element-wise "less than" comparison between a tensor and a scalar (tensor < scalar).
 * @copydetails mlib::core::greater(const Tensor<T>&, S)
 * @return A Tensor<bool> where each element is true if tensor_i < scalar_value.
 */
template <typename T, typename S>
Tensor<bool> less(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'less' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, tensor.data()[0] < static_cast<T>(scalar_value));

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return val < casted_scalar; });

	return Tensor<bool>(tensor.get_shape(), result);
}

/**
 * @brief Performs element-wise "less than" comparison between a scalar and a tensor (scalar < tensor).
 * @copydetails mlib::core::greater(S, const Tensor<T>&)
 * @return A Tensor<bool> where each element is true if scalar_value < tensor_i.
 */
template <typename T, typename S>
Tensor<bool> less(S scalar_value, const Tensor<T>& tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'less' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, static_cast<T>(scalar_value) < tensor.data()[0]);

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return casted_scalar < val; });

	return Tensor<bool>(tensor.get_shape(), result);
}

// Operator <
template <typename T>
Tensor<bool> operator<(const Tensor<T>& a, const Tensor<T>& b)
{
	return less(a, b);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator<(const Tensor<T>& tensor, S scalar_value)
{
	return less(tensor, scalar_value);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator<(S scalar_value, const Tensor<T>& tensor)
{
	return less(scalar_value, tensor);
}

// --- less_equal (<=) ---

/**
 * @brief Performs element-wise "less than or equal to" comparison of two tensors (a <= b).
 * @copydetails mlib::core::greater(const Tensor<T>&, const Tensor<T>&)
 * @return A Tensor<bool> where each element is true if a_i <= b_i, false otherwise.
 */
template <typename T>
Tensor<bool> less_equal(const Tensor<T>& a, const Tensor<T>& b)
{
	if (a.get_shape() != b.get_shape())
		throw ShapeMismatchError(a.get_shape(), b.get_shape(), "less_equal (tensor-tensor)");

	if (a.is_empty() && !a.is_scalar())
		return Tensor<bool>(a.get_shape());

	if (a.is_scalar())
		return Tensor<bool>({}, a.data()[0] <= b.data()[0]);

	std::vector<bool> result;
	result.reserve(a.get_total_size());
	std::transform(
		a.data(), a.data() + a.get_total_size(),
		b.data(),
		std::back_inserter(result),
		std::less_equal<T>());//[](T val_a, T val_b) { return val_a <= val_b; });

	return Tensor<bool>(a.get_shape(), result);
}

/**
 * @brief Performs element-wise "less than or equal to" comparison between a tensor and a scalar (tensor <= scalar).
 * @copydetails mlib::core::greater(const Tensor<T>&, S)
 * @return A Tensor<bool> where each element is true if tensor_i <= scalar_value.
 */
template <typename T, typename S>
Tensor<bool> less_equal(const Tensor<T>& tensor, S scalar_value)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'less_equal' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, tensor.data()[0] <= static_cast<T>(scalar_value));

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return val <= casted_scalar; });

	return Tensor<bool>(tensor.get_shape(), result);
}

/**
 * @brief Performs element-wise "less than or equal to" comparison between a scalar and a tensor (scalar <= tensor).
 * @copydetails mlib::core::greater(S, const Tensor<T>&)
 * @return A Tensor<bool> where each element is true if scalar_value <= tensor_i.
 */
template <typename T, typename S>
Tensor<bool> less_equal(S scalar_value, const Tensor<T>& tensor)
{
	static_assert(std::is_arithmetic_v<S>, "Scalar for 'less_equal' must be arithmetic.");

	if (tensor.is_empty() && !tensor.is_scalar())
		return Tensor<bool>(tensor.get_shape());

	if (tensor.is_scalar())
		return Tensor<bool>({}, static_cast<T>(scalar_value) <= tensor.data()[0]);

	std::vector<bool> result;
	result.reserve(tensor.get_total_size());
	T casted_scalar = static_cast<T>(scalar_value);
	std::transform(
		tensor.data(), tensor.data() + tensor.get_total_size(),
		std::back_inserter(result),
		[casted_scalar](T val) { return casted_scalar <= val; });

	return Tensor<bool>(tensor.get_shape(), result);
}

// Operator <=
template <typename T>
Tensor<bool> operator<=(const Tensor<T>& a, const Tensor<T>& b)
{
	return less_equal(a, b);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator<=(const Tensor<T>& tensor, S scalar_value)
{
	return less_equal(tensor, scalar_value);
}

template <typename T,typename S,typename=std::enable_if_t<std::is_arithmetic_v<S>>>
Tensor<bool> operator<=(S scalar_value, const Tensor<T>& tensor)
{
	return less_equal(scalar_value, tensor);
}

/**
 * @brief Performs matrix multiplication between two tensors, supporting various
 *        combinations of 1D (vectors) and 2D (matrices) inputs.
 *
 * This function dispatches to specific linear algebra routines based on input
 * tensor dimensionalities (ndim):
 * - If A is (m, k) and B is (k, n): Returns (m, n) matrix. (2D @ 2D)
 * - If A is (k,) and B is (k, n): Returns (n,) vector (A is treated as (1, k) row vector). (1D @ 2D)
 * - If A is (m, k) and B is (k,): Returns (m,) vector (B is treated as (k, 1) column vector). (2D @ 1D)
 * - If A is (k,) and B is (k,): Returns scalar () (A as (1,k), B as (k,1) for dot product). (1D @ 1D)
 *
 * All operations involve creating temporary copies of data if needed for 1D->2D conversions.
 *
 * @tparam T The data type of the tensor elements.
 * @param A The first tensor (left-hand side).
 * @param B The second tensor (right-hand side).
 * @return A new tensor containing the result of A @ B.
 * @throw DimensionError if input dimensions are incompatible (e.g., 3D or unsupported 0D inputs).
 * @throw ShapeMismatchError if inner dimensions do not match for multiplication.
 */
template <typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B)
{
	const auto& shape_A = A.get_shape();
	const auto& shape_B = B.get_shape();
	size_t ndim_A = A.ndim();
	size_t ndim_B = B.ndim();

	// --- CASE 1: 2D Matrix @ 2D Matrix: (m, k) @ (k, n) -> (m, n) ---
	if (ndim_A == 2 && ndim_B == 2)
	{
		size_t m = shape_A[0];   // Rows of A
		size_t k_A = shape_A[1]; // Columns of A
		size_t k_B = shape_B[0]; // Columns of B
		size_t n = shape_B[1];   // Column of B


		if (k_A != k_B)
		{
			throw ShapeMismatchError("Inner dimensions for matrix multiplication do not match: A.shape=" +
				ShapeMismatchError::shape_to_string(shape_A) + " vs B.shape=" + // Using the helper from your exception
				ShapeMismatchError::shape_to_string(shape_B) + ".");
		}

		size_t k = k_A;
		
		Tensor<T> C({m, n});

		for (size_t i = 0; i < m; i++) // Iterate rows of A (and C)
		{
			for (size_t j = 0; j < n; j++) // Iterate Columns of B (and C)
			{
				T sum_val = T{};
				if (k > 0)
				{
					for (size_t p = 0; p < k; p++) // Iterate common dimension
						sum_val += A(i, p) * B(p, j);
				}
				C(i, j) = sum_val;
			}
		}

		return C;
	}
	// --- END CASE 1 ---

    // --- CASE 2: 1D Vector @ 2D Matrix: (k,) @ (k, n) -> (n,) ---
	else if (ndim_A == 1 && ndim_B == 2)
	{
		size_t k_A = shape_A[0]; // Length of vector A
		size_t k_B = shape_B[0]; // Rows of matrix B
		size_t n = shape_B[1];   // Columns of matrix B

		if (k_A != k_B)
		{
			throw ShapeMismatchError(
                "Inner dimensions for vector-matrix multiplication (1D @ 2D) do not match: "
                "vector length " + std::to_string(k_A) + " vs matrix rows " + std::to_string(k_B) + "."
            );
		}

		Tensor<T> temp_vec_A_2D({1, k_A}, A.get_data_vector());
		Tensor<T> temp_result_2D = matmul(temp_vec_A_2D, B);
		
		temp_result_2D.reshape({n});
		return temp_result_2D;
	}
	// --- END CASE 2 ---

    // --- CASE 3: 2D Matrix @ 1D Vector: (m, k) @ (k,) -> (m,) ---
	else if (ndim_A == 2 && ndim_B == 1)
	{
		size_t m = shape_A[0];   // Rows of matrix A
		size_t k_A = shape_A[1]; // Columns of matrix A
		size_t k_B = shape_B[0]; // Length of vector B

		if (k_A != k_B)
		{
			throw ShapeMismatchError(
                "Inner dimensions for matrix-vector multiplication (2D @ 1D) do not match: "
                "matrix columns " + std::to_string(k_A) + " vs vector length " + std::to_string(k_B) + "."
            );
		}

		Tensor<T> temp_vec_B_2D({k_B, 1}, B.get_data_vector());
		Tensor<T> temp_result_2D = matmul(A, temp_vec_B_2D);

		temp_result_2D.reshape({m});
		return temp_result_2D;
	}
	// --- END CASE 3 ---

    // --- CASE 4: 1D Vector @ 1D Vector (Dot Product): (k,) @ (k,) -> scalar () ---
	else if (ndim_A == 1 && ndim_B == 1)
	{
		size_t k_A = shape_A[0];
		size_t k_B = shape_B[0];

		if (k_A != k_B)
		{
			throw ShapeMismatchError(
                "Inner dimensions for vector dot product (1D @ 1D) do not match: "
                "vector A length " + std::to_string(k_A) + " vs vector B length " + std::to_string(k_B) + "."
            );
		}

		Tensor<T> temp_vec_A_2D({1, k_A}, A.get_data_vector());
		Tensor<T> temp_vec_B_2D({k_B, 1}, B.get_data_vector());
		Tensor<T> temp_result_2D = matmul(temp_vec_A_2D, temp_vec_B_2D);

		temp_result_2D.reshape({});
		return temp_result_2D;
	}
	// --- END CASE 4 ---

    // --- CASE 5: Unsupported Dimensionality Combination ---
	else
	{
		throw DimensionError(
            "Unsupported dimensions for matmul: Received A.ndim=" + std::to_string(ndim_A) +
            " and B.ndim=" + std::to_string(ndim_B) + ". "
            "Supported are: (1D @ 2D), (2D @ 1D), (1D @ 1D), (2D @ 2D)."
        );
	}
}

// --- Reduction Operations ---

/**
 * @brief Calculates the sum of all elements in the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @return The sum of all elements. Returns T{} (e.g., 0 for numeric types) for an empty tensor.
 */
template <typename T>
T sum(const Tensor<T>& tensor)
{
	if (tensor.is_empty() && !tensor.is_scalar())
		return T{};

	return std::accumulate(tensor.data(), tensor.data() + tensor.get_total_size(), T{});
}

/**
 * @brief Calculates the mean (average) of all elements in the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @return The mean of all elements as a double. Returns NaN for an empty tensor.
 */
template <typename T>
double mean(const Tensor<T>& tensor)
{
	if (tensor.get_total_size() == 0)
		return std::numeric_limits<double>::quiet_NaN();

	if (tensor.is_scalar())
		return static_cast<double>(tensor.data()[0]);

	return static_cast<double>(sum(tensor)) / static_cast<double>(tensor.get_total_size());
}

/**
 * @brief Finds the maximum element in the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @return The maximum element.
 * @throw std::runtime_error if the tensor is empty.
 */
template <typename T>
T max_val(const Tensor<T>& tensor)
{
	if (tensor.get_total_size() == 0)
		throw std::runtime_error("Cannot find maximum value of an empty tensor.");

	const T* max_ptr = std::max_element(tensor.data(), tensor.data() + tensor.get_total_size());

	return *max_ptr;
}

/**
 * @brief Finds the minimum element in the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @return The minimum element.
 * @throw std::runtime_error if the tensor is empty.
 */
template <typename T>
T min_val(const Tensor<T>& tensor)
{
	if (tensor.get_total_size() == 0)
		throw std::runtime_error("Cannot find minimum value of an empty tensor.");

	const T* min_ptr = std::min_element(tensor.data(), tensor.data() + tensor.get_total_size());

	return *min_ptr;
}

/**
 * @brief Calculates the product of all elements in the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @return The product of all elements. Returns T{1} (e.g., 1 for numeric types) for an empty tensor.
 */
template <typename T>
T prod(const Tensor<T>& tensor)
{
	if (tensor.get_total_size() == 0)
		return static_cast<T>(1);  // Product of an empty set is 1 (multiplicative identity)
	
	return std::accumulate(tensor.data(), tensor.data() + tensor.get_total_size(), static_cast<T>(1), std::multiplies<T>());
}

// Helper to convert flat index to multi-dimensional indices
// This is effectively the inverse of calculate_flat_index
template <typename T>
std::vector<size_t> get_multi_dim_indices(size_t flat_idx, const Tensor<T>& tensor)
{
	if (tensor.ndim() == 0)
		return {};

	std::vector<size_t> indices(tensor.ndim());
	size_t remainder = flat_idx;
	for (size_t d = 0; d < tensor.ndim(); d++)
	{
		indices[d] = remainder / tensor.get_strides()[d];
		remainder %= tensor.get_strides()[d];
	}
	return indices;
}

// Helper to convert multi-dimensional indices to flat index (for the result tensor)
template <typename T_Result>
size_t calculate_output_flat_index(const std::vector<size_t>& out_indices, const Tensor<T_Result>& result_tensor)
{
	size_t flat_idx = 0;
	for (size_t d = 0; d < result_tensor.ndim(); d++)
		flat_idx += out_indices[d] * result_tensor.get_strides()[d];

	return flat_idx;
}

// Helper to build output shape for reductions
template <typename T>
typename Tensor<T>::shape_type build_reduced_shape(const typename Tensor<T>::shape_type& input_shape, int axis, bool keep_dims)
{
	typename Tensor<T>::shape_type output_shape;
	for (size_t d = 0; d < input_shape.size(); d++)
	{
		if (static_cast<int>(d) == axis)
		{
			if (keep_dims)
				output_shape.push_back(1);
		}
		else
			output_shape.push_back(input_shape[d]);
	}

	return output_shape;
}

// --- Axis-wise Sum ---

/**
 * @brief Calculates the sum of elements along a specified axis of the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @param axis The dimension along which to sum. Can be negative (-ndim to -1).
 * @param keep_dims If true, the reduced dimension will be kept as size 1 in the output shape.
 * @return A new tensor with the sum along the specified axis.
 * @throw DimensionError if the tensor is scalar, or if axis is out of bounds.
 */
template <typename T>
Tensor<T> sum(const Tensor<T>& tensor, int axis, bool keep_dims = false)
{
	if (tensor.ndim() == 0)
		throw DimensionError("Cannot sum a scalar tensor along an axis. Use mlib::core::sum(Tensor) for full sum.");

	if (axis < 0)
		axis += tensor.ndim();

	if(axis < 0 || static_cast<size_t>(axis) >= tensor.ndim())
		throw DimensionError("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(tensor.ndim()) + " dimensions.");

	typename Tensor<T>::shape_type output_shape = build_reduced_shape<T>(tensor.get_shape(), axis, keep_dims);

	if (tensor.get_total_size() == 0)
		return Tensor<T>(output_shape);

	Tensor<T> result(output_shape);

	for (size_t flat_idx = 0; flat_idx < tensor.get_total_size(); flat_idx++)
	{
		std::vector<size_t> current_in_indices = get_multi_dim_indices(flat_idx, tensor);

		std::vector<size_t> current_out_indices;
		current_out_indices.reserve(result.ndim());

		for (size_t d = 0; d < tensor.ndim(); d++)
		{
			if (static_cast<int>(d) == axis)
			{
				if (keep_dims)
					current_out_indices.push_back(0);
			}
			else
				current_out_indices.push_back(current_in_indices[d]);
		}

		size_t output_flat_idx = calculate_output_flat_index(current_out_indices, result);

		result.data()[output_flat_idx] += tensor.data()[flat_idx];
	}

	return result;
}

// --- Axis-wise Mean ---

/**
 * @brief Calculates the mean (average) of elements along a specified axis of the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @param axis The dimension along which to compute the mean. Can be negative.
 * @param keep_dims If true, the reduced dimension will be kept as size 1 in the output shape.
 * @return A new tensor with the mean along the specified axis.
 * @throw DimensionError if the tensor is scalar, or if axis is out of bounds.
 * @throw std::runtime_error if trying to calculate mean along an axis of size 0.
 */
template <typename T>
Tensor<double> mean(const Tensor<T>& tensor, int axis, bool keep_dims = false)
{
	if (tensor.ndim() == 0)
		throw DimensionError("Cannot compute mean of a scalar tensor along an axis. Use mlib::core::mean(Tensor) for full mean.");

	if (axis < 0)
		axis += tensor.ndim();

	if (axis < 0 || static_cast<size_t>(axis) >= tensor.ndim())
		throw DimensionError("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(tensor.ndim()) + " dimensions.");

	if (tensor.get_shape()[axis] == 0)
	{
		// If the axis itself has size 0 (e.g. mean of (2,0,3) along axis 1),
        // the sum along that axis would be 0, but total_size is 0, leading to 0/0.
        // It's like asking for mean of an empty slice.
        // For float, this can result in NaN. For integral, it would be a divide by zero.
        // Often, libraries return NaN for mean of empty slice.
        // Or, if any dimension is 0 and axis is not that dimension.
		typename Tensor<double>::shape_type output_shape = build_reduced_shape<T>(tensor.get_shape(), axis, keep_dims);
		Tensor<double> result(output_shape);
		if (result.get_total_size() == 0)
			return result;
		else
		{
			// Example: (2,0,3) mean over axis 0. Output (0,3). This is handled by result.get_total_size() == 0.
            // But if (2,1) mean over axis 1 where axis 1 is [1,0]. No, this is not a valid shape.
            // If axis 1 has 0 elements. (2,0). mean over axis 1. Sum is 0, count is 0.
            // We return NaN if the *reduced dimension* has size 0, otherwise it's fine.
			throw std::runtime_error("Cannot compute mean along axis " + std::to_string(axis) + " which has size 0.");
		}
	}
	if (tensor.get_total_size() == 0)
	{
		typename Tensor<double>::shape_type output_shape = build_reduced_shape<T>(tensor.get_shape(), axis, keep_dims);
		return Tensor<double>(output_shape);
	}

	Tensor<T> sum_result_t = sum(tensor, axis, keep_dims);
	typename Tensor<double>::shape_type output_shape_mean = sum_result_t.get_shape();
	Tensor<double> result(output_shape_mean);

	size_t count_per_slice = tensor.get_shape()[axis]; // The number of elements that were summed for each output value

	// Divide sum by count_per_slice
	for (size_t flat_idx = 0; flat_idx < sum_result_t.get_total_size(); flat_idx++)
		result.data()[flat_idx] = static_cast<double>(sum_result_t.data()[flat_idx]) / static_cast<double>(count_per_slice);

	return result;
}


// --- Axis-wise Max Val ---

/**
 * @brief Finds the maximum element along a specified axis of the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @param axis The dimension along which to find the maximum. Can be negative.
 * @param keep_dims If true, the reduced dimension will be kept as size 1 in the output shape.
 * @return A new tensor with the maximum elements along the specified axis.
 * @throw DimensionError if the tensor is scalar, or if axis is out of bounds.
 * @throw std::runtime_error if any slice along the axis is empty.
 */
template <typename T>
Tensor<T> max_val(const Tensor<T>& tensor, int axis, bool keep_dims = false)
{
	if (tensor.ndim() == 0)
		throw DimensionError("Cannot compute max of a scalar tensor along an axis. Use mlib::core::max(Tensor) for full max.");

	if (axis < 0)
		axis += tensor.ndim();

	if (axis < 0 || static_cast<size_t>(axis) >= tensor.ndim())
		throw DimensionError("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(tensor.ndim()) + " dimensions.");

	if (tensor.get_shape()[axis] == 0) // If the axis itself has size 0, no max can be found
		throw std::runtime_error("Cannot compute max along axis " + std::to_string(axis) + " which has size 0.");

	typename Tensor<T>::shape_type output_shape = build_reduced_shape<T>(tensor.get_shape(), axis, keep_dims);

	if (tensor.get_total_size() == 0) // If entire tensor is empty (e.g. from 0-dim elsewhere)
		return Tensor<T>(output_shape);

	Tensor<T> result(output_shape);

	// The length of the axis being reduced (number of elements in each slice)
    size_t reduced_dim_size = tensor.get_shape()[axis];

    for (size_t out_flat_idx = 0; out_flat_idx < result.get_total_size(); ++out_flat_idx)
	{
        std::vector<size_t> current_out_coords = get_multi_dim_indices(out_flat_idx, result); // This needs result.ndim()

        T current_max_val = std::numeric_limits<T>::lowest(); // Initialize for comparison

        std::vector<size_t> temp_in_coords_base = current_out_coords; // Copy output coords to form base input coords

        // Need to insert a placeholder for 'axis' dimension in temp_in_coords_base
        // For example if input (2,3,4) axis 1. Output (2,4).
        // current_out_coords (r,c) maps to original (r, [p], c).
        temp_in_coords_base.insert(temp_in_coords_base.begin() + axis, 0); // Placeholder, will be replaced by 'p'

        // Loop over the elements in the slice defined by 'axis'
        for (size_t p = 0; p < reduced_dim_size; ++p)
		{
            temp_in_coords_base[axis] = p;

            size_t flat_input_idx = 0;
            for (size_t d = 0; d < tensor.ndim(); ++d)
                flat_input_idx += temp_in_coords_base[d] * tensor.get_strides()[d];

            T current_element = tensor.data()[flat_input_idx];
            if (current_element > current_max_val)
                current_max_val = current_element;
        }
        result.data()[out_flat_idx] = current_max_val;
    }
    return result;
}


// --- Axis-wise Min Val ---

/**
 * @brief Finds the minimum element along a specified axis of the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @param axis The dimension along which to find the minimum. Can be negative.
 * @param keep_dims If true, the reduced dimension will be kept as size 1 in the output shape.
 * @return A new tensor with the minimum elements along the specified axis.
 * @throw DimensionError if the tensor is scalar, or if axis is out of bounds.
 * @throw std::runtime_error if any slice along the axis is empty.
 */
template <typename T>
Tensor<T> min_val(const Tensor<T>& tensor, int axis, bool keep_dims = false)
{
	if (tensor.ndim() == 0)
		throw DimensionError("Cannot compute min of a scalar tensor along an axis. Use mlib::core::min(Tensor) for full min.");

	if (axis < 0)
		axis += tensor.ndim();

	if (axis < 0 || static_cast<size_t>(axis) >= tensor.ndim())
		throw DimensionError("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(tensor.ndim()) + " dimensions.");

	if (tensor.get_shape()[axis] == 0) // If the axis itself has size 0, no max can be found
		throw std::runtime_error("Cannot compute min along axis " + std::to_string(axis) + " which has size 0.");

	typename Tensor<T>::shape_type output_shape = build_reduced_shape<T>(tensor.get_shape(), axis, keep_dims);

	if (tensor.get_total_size() == 0) // If entire tensor is empty (e.g. from 0-dim elsewhere)
		return Tensor<T>(output_shape);

	Tensor<T> result(output_shape);

	size_t reduced_dim_size = tensor.get_shape()[axis];

    for (size_t out_flat_idx = 0; out_flat_idx < result.get_total_size(); ++out_flat_idx)
	{
        std::vector<size_t> current_out_coords = get_multi_dim_indices(out_flat_idx, result);

        T current_min_val = std::numeric_limits<T>::max();

        std::vector<size_t> temp_in_coords_base = current_out_coords;
        temp_in_coords_base.insert(temp_in_coords_base.begin() + axis, 0); // Placeholder

        for (size_t p = 0; p < reduced_dim_size; ++p)
		{
            temp_in_coords_base[axis] = p;

            size_t flat_input_idx = 0;
            for (size_t d = 0; d < tensor.ndim(); ++d)
                flat_input_idx += temp_in_coords_base[d] * tensor.get_strides()[d];

            T current_element = tensor.data()[flat_input_idx];
            if (current_element < current_min_val)
                current_min_val = current_element;
        }
        result.data()[out_flat_idx] = current_min_val;
    }
    return result;
}


// --- Axis-wise Prod ---

/**
 * @brief Calculates the product of elements along a specified axis of the tensor.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor.
 * @param axis The dimension along which to compute the product. Can be negative.
 * @param keep_dims If true, the reduced dimension will be kept as size 1 in the output shape.
 * @return A new tensor with the product along the specified axis.
 * @throw DimensionError if the tensor is scalar, or if axis is out of bounds.
 */
template <typename T>
Tensor<T> prod(const Tensor<T>& tensor, int axis, int keep_dims = false)
{
	if (tensor.ndim() == 0)
		throw DimensionError("Cannot compute product of a scalar tensor along an axis.. Use mlib::core::prod(Tensor) for full prod.");

	if (axis < 0)
		axis += tensor.ndim();

	if (axis < 0 || static_cast<size_t>(axis) >= tensor.ndim())
		throw DimensionError("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(tensor.ndim()) + " dimensions.");

    typename Tensor<T>::shape_type output_shape = build_reduced_shape<T>(tensor.get_shape(), axis, keep_dims);

	if (tensor.get_shape()[axis] == 0)
	{
        Tensor<T> result_with_ones(output_shape);
        // Fill result with ones (default is T{}, i.e., 0, so must re-initialize)
        for (size_t i = 0; i < result_with_ones.get_total_size(); ++i)
             result_with_ones.data()[i] = static_cast<T>(1);

        return result_with_ones;
    }
    if (tensor.get_total_size() == 0)
        return Tensor<T>(output_shape); // Correctly returns an empty tensor

	Tensor<T> result(output_shape);

	size_t reduced_dim_size = tensor.get_shape()[axis];

    for (size_t out_flat_idx = 0; out_flat_idx < result.get_total_size(); ++out_flat_idx)
	{
        std::vector<size_t> current_out_coords = get_multi_dim_indices(out_flat_idx, result); // This uses result.ndim()

        T current_prod_val = static_cast<T>(1); // Initialize product to 1

        std::vector<size_t> temp_in_coords_base = current_out_coords;
        temp_in_coords_base.insert(temp_in_coords_base.begin() + axis, 0); // Placeholder

        for (size_t p = 0; p < reduced_dim_size; ++p)
		{
            temp_in_coords_base[axis] = p;

            size_t flat_input_idx = 0;
            for (size_t d = 0; d < tensor.ndim(); ++d)
                flat_input_idx += temp_in_coords_base[d] * tensor.get_strides()[d];

            T current_element = tensor.data()[flat_input_idx];
            current_prod_val *= current_element;
        }
        result.data()[out_flat_idx] = current_prod_val;
    }
    return result;
}

// --- Linear Algebra Operations ---

/**
 * @brief Performs a transpose operation on a 2D tensor (matrix).
 *
 * The rows and columns of the input matrix are swapped, creating a new matrix
 * where the element at (j, i) in the output is the element at (i, j) in the input.
 * This operation creates a copy of the data.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input 2D tensor (matrix) to be transposed.
 * @return A new Tensor<T> object representing the transposed matrix.
 * @throw DimensionError if the input tensor is not 2D.
 */
template <typename T>
Tensor<T> transpose(const Tensor<T>& tensor)
{
	if (tensor.ndim() != 2)
	{
		throw DimensionError(
			"Transpose operation currently only supports 2D tensors (matrices). "
            "Input tensor has " + std::to_string(tensor.ndim()) + " dimensions.");
	}

	const auto& input_shape = tensor.get_shape();
	size_t rows = input_shape[0];
	size_t columns = input_shape[1];

	typename Tensor<T>::shape_type output_shape = {columns, rows};
	Tensor<T> result(output_shape);
	for (size_t r = 0; r < rows; r++)
	{
		for (size_t c = 0; c < columns; c++)
			result(c, r) = tensor(r, c);
	}

	return result;
}

// Helper function to calculate total_size from shape
// This is duplicated from Tensor constructors, can be refactored into a private helper later if desired.
size_t calculate_total_size_from_shape(const typename Tensor<size_t>::shape_type& shape)
{
    size_t total_size = 1;
    if (shape.empty())
        total_size = 1; // Scalar tensor (total_size 1)
    else
	{
        for (size_t dim_size : shape)
		{
            if (dim_size == 0)
			{
                total_size = 0;
                break;
            }
            total_size *= dim_size;
        }
    }
    return total_size;
}

/**
 * @brief Creates a tensor filled with zeros.
 *
 * @tparam T The data type of the tensor elements. Must be arithmetic or boolean.
 * @param shape The desired shape of the tensor.
 * @return A new Tensor<T> object filled with zeros.
 */
template <typename T>
Tensor<T> zeros(const typename Tensor<T>::shape_type& shape)
{
	static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, bool>, "Tensor type for 'zeros' must be arithmetic or boolean.");

	size_t total_size = calculate_total_size_from_shape(shape);
	std::vector<T> data_vector(total_size);
	std::fill(data_vector.begin(), data_vector.end(), T{});
	return Tensor<T>(shape, data_vector);
}

/**
 * @brief Creates a tensor filled with ones.
 *
 * @tparam T The data type of the tensor elements. Must be arithmetic or boolean.
 * @param shape The desired shape of the tensor.
 * @return A new Tensor<T> object filled with ones.
 */
template <typename T>
Tensor<T> ones(const typename Tensor<T>::shape_type& shape)
{
	static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, bool>, "Tensor type for 'ones' must be arithmetic or boolean.");

	size_t total_size = calculate_total_size_from_shape(shape);
	std::vector<T> data_vector(total_size);
	std::fill(data_vector.begin(), data_vector.end(), static_cast<T>(1));
	return Tensor<T>(shape, data_vector);
}

/**
 * @brief Creates a tensor filled with a specified value.
 *
 * @tparam T The data type of the tensor elements.
 * @param shape The desired shape of the tensor.
 * @param fill_value The value to fill all elements with.
 * @return A new Tensor<T> object filled with `fill_value`.
 */
template <typename T>
Tensor<T> full(const typename Tensor<T>::shape_type& shape, T fill_value)
{
	size_t total_size = calculate_total_size_from_shape(shape);
	std::vector<T> data_vector(total_size);
	std::fill(data_vector.begin(), data_vector.end(), fill_value);
	return Tensor<T>(shape, data_vector);
}

/**
 * @brief Creates a 2D identity matrix (eye).
 *
 * @tparam T The data type of the matrix elements.
 * @param N The number of rows and columns (matrix will be NxN).
 * @return A new Tensor<T> object representing an NxN identity matrix.
 */
template <typename T>
Tensor<T> eye(size_t N)
{
	static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, bool>, "Tensor type for 'eye' must be arithmetic or boolean.");

	typename Tensor<T>::shape_type shape = {N, N};
	size_t total_size = N * N;
	std::vector<T> data_vector(total_size);
	std::fill(data_vector.begin(), data_vector.end(), T{});

	for (size_t i = 0; i < N; i++)
		data_vector[i * N + i] = static_cast<T>(1);

	return Tensor<T>(shape, data_vector);
}

/**
 * @brief Creates a 1D tensor with evenly spaced values within a half-open interval [start, stop).
 *        Similar to Python's range or NumPy's arange.
 *
 * @tparam T The data type of the tensor elements. Must be arithmetic.
 * @param start The start of the interval (inclusive).
 * @param stop The end of the interval (exclusive).
 * @param step The step size between values. Defaults to 1 (or 1.0f for float/double).
 * @return A new 1D Tensor<T> containing the generated sequence.
 * @throw std::invalid_argument if step is zero.
 */
template <typename T>
Tensor<T> arange(T start, T stop, T step = static_cast<T>(1))
{
	static_assert(std::is_arithmetic_v<T>, "Tensor type for 'arange' must be arithmetic.");
	
	if (step == static_cast<T>(0))
		throw std::invalid_argument("Step cannot be zero in arange.");

	if ((step > static_cast<T>(0) && start >= stop) || (step < static_cast<T>(0) && start <= stop))
		return Tensor<T>({0}); // Return an empty 1D tensor
	
	std::vector<T> data_vector;
	for (T val = start; (step > static_cast<T>(0) ? (val < stop) : (val > stop)); val += step)
		data_vector.push_back(val);

	return Tensor<T>({data_vector.size()}, data_vector);
}

/**
 * @brief Creates a 1D tensor with `num_points` evenly spaced samples,
 *        calculated over the interval [start, end] (inclusive).
 *        Similar to NumPy's linspace.
 *
 * @tparam T The data type of the tensor elements. Must be floating point.
 * @param start The starting value of the sequence.
 * @param end The end value of the sequence.
 * @param num_points The number of samples to generate. Must be non-negative.
 * @return A new 1D Tensor<T> containing the generated sequence.
 * @throw std::invalid_argument if num_points is 0.
 */
template <typename T>
Tensor<T> linspace(T start, T end, size_t num_points)
{
	static_assert(std::is_arithmetic_v<T>, "Tensor type for 'linspace' must be arithmetic.");

	if (num_points == 0)
		throw std::invalid_argument("Number of points must be non-zero in linspace");

	typename Tensor<T>::shape_type shape = {num_points};
	std::vector<T> data_vector;
	data_vector.reserve(num_points);

	if (num_points == 1)
		data_vector.push_back(start);
	else
	{
		T step_size = (end - start) / static_cast<T>(num_points - 1);
		for (size_t i = 0; i < num_points; i++)
			data_vector.push_back(start + static_cast<T>(i) * step_size);
	}

	return Tensor<T>(shape, data_vector);
}

} // namespace core
} // namespace mlib

#endif // MLIB_CORE_OPERATIONS_HPP
