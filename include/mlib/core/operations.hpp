#ifndef MLIB_CORE_OPERATIONS_HPP
#define MLIB_CORE_OPERATIONS_HPP

#include "tensor.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <functional>
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
 * @brief Overloads the + operator for element-wise tensor addition.
 * @see mlib::core::add(const Tensor<T>&, const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b)
{
	return add(a, b);
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
 * @brief Overloads the - operator for element-wise tensor subtraction.
 * @see mlib::core::subtract(const Tensor<T>&, const Tensor<T>&)
 */
template <typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b)
{
	return subtract(a, b);
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

} // namespace core
} // namespace mlib

#endif // MLIB_CORE_OPERATIONS_HPP
