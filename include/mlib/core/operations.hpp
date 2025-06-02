#ifndef MLIB_CORE_OPERATIONS_HPP
#define MLIB_CORE_OPERATIONS_HPP

#include "tensor.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>
#include <numeric>
#include <stdexcept>

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

} // namespace core
} // namespace mlib

#endif // MLIB_CORE_OPERATIONS_HPP
