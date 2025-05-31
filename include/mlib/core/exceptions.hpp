#ifndef MLIB_CORE_EXCEPTIONS_HPP
#define MLIB_CORE_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace mlib
{
namespace core
{

/**
 * @brief Base class for exceptions thrown by the mlib library.
 */
class MlibException : public std::runtime_error
{
public:
	using std::runtime_error::runtime_error; // Inherit constructors
};

/**
 * @brief Exception thrown for errors related to tensor dimensions.
 *
 * This can include shape mismatches, invalid dimension sizes, etc.
 */
class DimensionError : public MlibException
{
public:
	using MlibException::MlibException; // Inherit constructors
};

/**
 * @brief Exception thrown when an operation requires tensors of compatible
 *        shapes but they are not (e.g., element-wise addition).
 */
class ShapeMismatchError : public DimensionError
{
public:
	ShapeMismatchError(const std::string& message) : DimensionError(message) {}

	ShapeMismatchError(
			const std::vector<size_t>& shape_a,
			const std::vector<size_t>& shape_b,
			const std::string& operation_name)
		: DimensionError(build_message(shape_a, shape_b, operation_name)) {}

private:

	static std::string shape_to_string(const std::vector<size_t>& shape)
	{
		std::string s = "{";
		for(size_t i = 0; i < shape.size(); i++)
		{
			s += std::to_string(shape[i]);
			if(i < shape.size() - 1)
				s += ", ";
		}
		s += "}";
		return s;
	}

	static std::string build_message(
			const std::vector<size_t>& shape_a,
			const std::vector<size_t>& shape_b,
			const std::string& operation_name)
	{
		return "Shape mismatch in operation '" + operation_name + "': " +
			shape_to_string(shape_a) + " vs " + shape_to_string(shape_b) + ".";
	}

};

/**
 * @brief Exception for operations not supported for a given data type.
 */
class TypeError : public MlibException
{
public:
	using MlibException::MlibException;
};

} // namespace core
} // namespace mlib

#endif // MLIB_CORE_EXCEPTIONS_HPP
