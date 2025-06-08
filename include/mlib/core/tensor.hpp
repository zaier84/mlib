#ifndef MLIB_CORE_TENSOR_HPP
#define MLIB_CORE_TENSOR_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mlib
{
namespace core
{

template <typename T>
class Tensor
{
public:
    using value_type = T;
    using shape_type = std::vector<size_t>;
    using strides_type = std::vector<size_t>;
    using data_type = std::vector<T>;

    // Constructors and Destructor
    Tensor();
    explicit Tensor(const shape_type &shape);
    Tensor(const shape_type &shape, const data_type &data);
    Tensor(const shape_type &shape, T fill_value);

    // Copy semantics
    Tensor(const Tensor &other);
    Tensor &operator=(const Tensor &other);

    // Move semactics
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    ~Tensor() = default;

    // Accessors
    const shape_type &get_shape() const;
    const strides_type &get_strides() const;
    size_t ndim() const;
    size_t get_total_size() const;
    bool is_empty() const;
	bool is_scalar() const;
    bool is_contiguous() const;

    // Element access (const and non-const)
    auto at(const std::vector<size_t> &indices) -> std::conditional_t<std::is_same_v<T, bool>, bool, T&>;
	auto at(const std::vector<size_t> &indices) const -> std::conditional_t<std::is_same_v<T, bool>, bool, const T&>;

    // Variadic template for convinent access like tensor(i, j, k)
    template <typename... OperatorArgs>
	auto operator()(OperatorArgs... args) -> std::conditional_t<std::is_same_v<T, bool>, bool, T&>;
    template <typename... OperatorArgs>
	auto operator()(OperatorArgs... args) const -> std::conditional_t<std::is_same_v<T, bool>, bool, const T&>;

    // Direct data access
    T *data();
    const T *data() const;
	const data_type &get_data_vector() const;

    // Modifiers
    void fill(T value);
    void reshape(const shape_type &new_shape);

    // Utility
    void print(std::ostream &os = std::cout) const;

private:
    data_type _data;
    shape_type _shape;
    strides_type _strides;
    size_t _total_size = 0;

    // Private Helper Functions
    void calculate_strides();
    size_t calculate_flat_index(const std::vector<size_t> &indices) const;

    template <typename IndexType, typename... RestIndices>
    size_t calculate_flat_index_variadic(size_t current_dim_idx,
                                         size_t accumulated_offset,
                                         IndexType current_index,
                                         RestIndices... rest_indices) const;

    size_t calculate_flat_index_variadic(size_t current_dim_idx,
                                         size_t accumulated) const;
};

template <typename T>
void Tensor<T>::calculate_strides()
{
    if (_shape.empty())
    {
        _strides.clear();
        return;
    }
    _strides.resize(_shape.size());
    _strides.back() = 1;

    for (int i = static_cast<int>(_shape.size()) - 2; i >= 0; i--)
    {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
}

template <typename T>
size_t Tensor<T>::calculate_flat_index(const std::vector<size_t> &indices) const
{
    if (indices.size() != _shape.size())
    {
        throw std::out_of_range(
            "Number of indices does not match tensor dimensions.");
    }
    size_t flat_index = 0;
    for (size_t i = 0; i < _shape.size(); i++)
    {
        if (indices[i] >= _shape[i])
        {
            throw std::out_of_range("Index out of bound for dimension " +
                                    std::to_string(i));
        }
        flat_index += indices[i] * _strides[i];
    }
    return flat_index;
}

// Variadic index calculation (base case for recursion)
// Base case: No more indices to process (recursion termination)
template <typename T>
size_t Tensor<T>::calculate_flat_index_variadic(size_t current_dim_idx, size_t accumulated_offset) const
{
	// Check that we've processed exactly the right number of dimensions
    if (current_dim_idx != _shape.size())
        throw std::out_of_range("Too many indicies provided for tensor dimensions.");
    return accumulated_offset;
}

// Variadic index calculation (recursive step)
// Recursive case: Process current index and recurse with remaining indices
template <typename T>
template <typename IndexType, typename... RestIndices>
size_t Tensor<T>::calculate_flat_index_variadic(
    size_t current_dim_idx,
	size_t accumulated_offset,
	IndexType current_index,
    RestIndices... rest_indices) const
{
	// Check we haven't exceeded the number of dimensions
    if (current_dim_idx >= _shape.size())
        throw std::out_of_range("Too many indices provided for tensor dimension.");

	// Bounds check for current index
    if (static_cast<size_t>(current_index) >= _shape[current_dim_idx])
    {
        throw std::out_of_range(
            "Index " + std::to_string(current_index) +
            " out of bound of dimension " + std::to_string(current_dim_idx) +
            " with size " + std::to_string(_shape[current_dim_idx]));
    }

	// Add current index contribution to accumulated offset
    accumulated_offset += static_cast<size_t>(current_index) * _strides[current_dim_idx];

	// Recurse with remaining indices
    return calculate_flat_index_variadic(current_dim_idx + 1,
                                         accumulated_offset, rest_indices...);
}

// Constructors & Destructor Implementations
// 1. DEFAULT CONSTRUCTOR
template <typename T>
Tensor<T>::Tensor() : _total_size(0)
{
    // Empty tensor, _shape, _strides, _data are already empty std::vectors
	// _total_size explicitly set to 0
}

// 2. SHAPE-ONLY CONSTRUCTOR
template <typename T>
Tensor<T>::Tensor(const shape_type &shape) : _shape(shape)
{
	// Calculate total size from shape
    if (_shape.empty())
        _total_size = 1;
    else
    {
        _total_size = 1;
        for (size_t dim_size : _shape)
        {
            if (dim_size == 0)
            {
                _total_size = 0;
                break;
            }
            _total_size *= dim_size;
        }
    }
    _data.resize(_total_size, T{});
    calculate_strides();
}

// 3. SHAPE + DATA CONSTRUCTOR
template <typename T>
Tensor<T>::Tensor(const shape_type &shape, const data_type &data)
    : _shape(shape)
{
	// Handle special case: scalar tensor (empty shape, single data element){}
    if (shape.empty() && !data.empty() && data.size() == 1)
        _total_size = 1;
	// Handle empty tensor cases
    else if (shape.empty() && (data.empty() || data.size() != 1))
        _total_size = 0;
	// Normal case: calculate size from shape
    else
    {
		_total_size = 1;
        for (size_t dim_size : _shape)
        {
            if (dim_size == 0)
            {
                _total_size = 0;
                break;
            }
            _total_size *= dim_size;
        }
    }
	// Validate that data size matches calculated total size
    if (_total_size != data.size())
        throw std::invalid_argument("Data size does not match shape dimensions.");

    _data = data;  // Copy the provided data
    calculate_strides();
}

// 4. SHAPE + FILL VALUE CONSTRUCTOR
template <typename T>
Tensor<T>::Tensor(const shape_type &shape, T fill_value) : _shape(shape)
{
	// Calculate total size (same logic as shape-only constructor)
    if (shape.empty())
        _total_size = 1;
    else
    {
        _total_size = 1;
        for (size_t dim_size : _shape)
        {
            if (dim_size == 0)
            {
                _total_size = 0;
                break;
            }
            _total_size *= dim_size;
        }
    }

	// Fill data with the specified value
    _data.assign(_total_size, fill_value);
    calculate_strides();
}

// Copy semantics
// 5. COPY CONSTRUCTOR
template <typename T>
Tensor<T>::Tensor(const Tensor &other)
    : _data(other._data), _shape(other._shape), _strides(other._strides),
      _total_size(other._total_size)
{
	// Deep copy all member variables
    // No additional work needed - member initializer list handles everything
}

// 6. COPY ASSIGNMENT OPERATOR
template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor &other)
{
    if (this != &other)  // Self-assignment check
    {
        _data = other._data;
        _shape = other._shape;
        _strides = other._strides;
        _total_size = other._total_size;
    }
    return *this;
}

// Move semantics
// 7. MOVE CONSTRUCTOR
template <typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept
    : _data(std::move(other._data)),
	  _shape(std::move(other._shape)),
      _strides(std::move(other._strides)),
      _total_size(std::move(other._total_size))
{
	other._shape.clear();
	other._strides.clear();
    other._total_size = 0;  // Leave other in valid state
}

// 8. MOVE ASSIGNMENT OPERATOR
template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor &&other) noexcept
{
    if (this != &other)  // Self-assignment check
    {
        _data = std::move(other._data);
		_shape = std::move(other._shape);
        _strides = std::move(other._strides);
        _total_size = std::move(other._total_size);

		other._shape.clear();
		other._strides.clear();
		other._total_size = 0;  // Leave other in valid state
    }
    return *this;
}

// Accessor Implementation
// 1. SHAPE ACCESSOR
template <typename T>
const typename Tensor<T>::shape_type &Tensor<T>::get_shape() const { return _shape; }

// 2. STRIDES ACCESSOR
template <typename T>
const typename Tensor<T>::strides_type &Tensor<T>::get_strides() const { return _strides; }

// 3. NUMBER OF DIMENSIONS
template <typename T>
size_t Tensor<T>::ndim() const { return _shape.size(); }

// 4. TOTAL SIZE ACCESSOR
template <typename T>
size_t Tensor<T>::get_total_size() const { return _total_size; }

// 5. EMPTY CHECK
template <typename T>
bool Tensor<T>::is_empty() const
{
    return _total_size == 0 && _data.empty();
}

// 6. SCALAR CHECK
template <typename T>
bool Tensor<T>::is_scalar() const
{
	return _shape.empty() && _total_size == 1;
}

// 7. CONTIGUITY CHECK
template <typename T>
bool Tensor<T>::is_contiguous() const
{
    if (is_empty() && !is_scalar())
        return true;  // Empty tensors are considered contiguous

	if (is_scalar())
		return true;  // A scalar is always contiguous

    if (_shape.empty())
        return true;  // Scalar tensors are contiguous
	
	// Calculate what strides would be for contiguous storage
    strides_type expected_strides(_shape.size());
	if (_shape.empty() || _total_size == 0) return true;

    expected_strides.back() = 1;  // Rightmost stride is always 1
    for (int i = static_cast<int>(_shape.size()) - 2; i >= 0; i--)
    {
        expected_strides[i] = expected_strides[i + 1] * _shape[i + 1];
    }
    return _strides == expected_strides;  // Compare actual vs expected strides
}

// 8. ELEMENT ACCESS WITH VECTOR INDICES (NON-CONST)
template <typename T>
// T &Tensor<T>::at(const std::vector<size_t> &indices)
auto Tensor<T>::at(const std::vector<size_t> &indices) -> std::conditional_t<std::is_same_v<T, bool>, bool, T&>
{
	if (is_empty())
      throw std::out_of_range("Cannot access elements in an empty tensor.");
	if (is_scalar())  // at() is not for scalars, use operator()()
      throw std::out_of_range("Cannot use std::vector<size_t> indices for a scalar tensor. Use operator()().");
	
	size_t flat_idx = calculate_flat_index(indices);
	if constexpr (std::is_same_v<T, bool>)
		return static_cast<bool>(_data[flat_idx]); // Return by value
	else
		return _data[flat_idx]; // Return by reference
}

// 9. ELEMENT ACCESS WITH VECTOR INDICES (CONST)
template <typename T>
// const T &Tensor<T>::at(const std::vector<size_t> &indices) const
auto Tensor<T>::at(const std::vector<size_t> &indices) const -> std::conditional_t<std::is_same_v<T, bool>, bool, const T&>
{
	if (is_empty())
      throw std::out_of_range("Cannot access elements in an empty tensor.");
	if (is_scalar())  // at() is not for scalars, use operator()()
      throw std::out_of_range("Cannot use std::vector<size_t> indices for a scalar tensor. Use operator()().");
	
	size_t flat_idx = calculate_flat_index(indices);
	if constexpr (std::is_same_v<T, bool>)
		return static_cast<bool>(_data[flat_idx]); // Return by value
	else
		return _data[flat_idx]; // Return by reference
}

// 10. VARIADIC ELEMENT ACCESS (NON-CONST)
template <typename T>
template <typename... OperatorArgs>
// T& Tensor<T>::operator()(OperatorArgs... args) { // OLD
auto Tensor<T>::operator()(OperatorArgs... args) -> std::conditional_t<std::is_same_v<T, bool>, bool, T&>
{
	// Compile-time check: all arguments must be integral.
	static_assert((std::is_integral_v<OperatorArgs> && ...),
                "All indices for operator() must be integral types.");
	
	if constexpr (sizeof...(OperatorArgs) == 0)
	{
		if(!is_scalar())
			throw std::out_of_range("operator() with no arguments is only for scalar tensors.");

		// For bool, _data[0] returns a proxy, so we read its value.
        // For other T, _data[0] returns T&.
		if constexpr (std::is_same_v<T, bool>)
			return static_cast<bool>(_data[0]); // Return by value
		else
			return _data[0]; // Return by reference
	}
	else
	{
		// This block is compiled only if one or more arguments are passed.
		if (is_scalar()) { // Runtime check
			throw std::out_of_range("Indices provided for a scalar tensor.");
		}
		if (is_empty()){ // Runtime check for non-scalar empty tensors (e.g. {0,2})
			throw std::out_of_range("Cannot access elements in an empty (non-scalar) tensor via operator().");
		}
	
		if (sizeof...(OperatorArgs) != _shape.size())
		{
			throw std::out_of_range(
				"Number of indices (" + std::to_string(sizeof...(OperatorArgs)) +
				") in operator() does not match tensor dimensions (" +
				std::to_string(_shape.size()) + ").");
		}

		size_t flat_idx = calculate_flat_index_variadic(0, 0, args...);
		if constexpr (std::is_same_v<T, bool>)
			return static_cast<bool>(_data[flat_idx]); // Return by value
		else
			return _data[flat_idx]; // Return by reference
	}
}

// 11. VARIADIC ELEMENT ACCESS (CONST)
template <typename T>
template <typename... OperatorArgs>
// const T &Tensor<T>::operator()(OperatorArgs... args) const
auto Tensor<T>::operator()(OperatorArgs... args) const -> std::conditional_t<std::is_same_v<T, bool>, bool, const T&>
{
	// Compile-time check: all arguments must be integral.
	static_assert((std::is_integral_v<OperatorArgs> && ...),
                "All indices for operator() must be integral types.");
	
	if constexpr (sizeof...(OperatorArgs) == 0)
	{
		if(!is_scalar())
			throw std::out_of_range("operator() with no arguments is only for scalar tensors.");

		if constexpr (std::is_same_v<T, bool>)
			return static_cast<bool>(_data[0]); // Return by value
		else
			return _data[0]; // Return by const reference
	}
	else
	{
		// This block is compiled only if one or more arguments are passed.
		if (is_scalar()) { // Runtime check
			throw std::out_of_range("Indices provided for a scalar tensor.");
		}
		if (is_empty()){ // Runtime check for non-scalar empty tensors (e.g. {0,2})
			throw std::out_of_range("Cannot access elements in an empty (non-scalar) tensor via operator().");
		}
	
		if (sizeof...(OperatorArgs) != _shape.size())
		{
			throw std::out_of_range(
				"Number of indices (" + std::to_string(sizeof...(OperatorArgs)) +
				") in operator() does not match tensor dimensions (" +
				std::to_string(_shape.size()) + ").");
		}
		size_t flat_idx = calculate_flat_index_variadic(0, 0, args...);
		if constexpr (std::is_same_v<T, bool>)
			return static_cast<bool>(_data[flat_idx]); // Return by value
		else
			return _data[flat_idx]; // Return by const reference
	}
}

// 12. RAW DATA ACCESS (NON-CONST)
template <typename T>
T *Tensor<T>::data()
{
	if constexpr (std::is_same_v<T, bool>)
		throw std::runtime_error("Direct data() pointer access is not supported for Tensor<bool>. Use operator() or at().");

	if (is_empty() && !is_scalar()) return nullptr;

	return _data.data();
}

// 13. RAW DATA ACCESS (CONST)
template <typename T>
const T *Tensor<T>::data() const
{
	if constexpr (std::is_same_v<T, bool>)
		throw std::runtime_error("Direct data() pointer access is not supported for Tensor<bool>. Use operator() or at().");

	if (is_empty() && !is_scalar()) return nullptr;

	return _data.data();
}

// 14. RAW DATA VECTOR (CONST)
template <typename T>
const typename Tensor<T>::data_type &Tensor<T>::get_data_vector() const { return _data; }

// Modifier Implementations
// 1. FILL METHOD - Sets all elements to a single value
template <typename T> void Tensor<T>::fill(T value)
{
    std::fill(_data.begin(), _data.end(), value);
}

// 2. RESHAPE METHOD - Changes tensor dimensions while preserving data
template <typename T> void Tensor<T>::reshape(const shape_type &new_shape)
{
    size_t new_total_size = 1;
    if (new_shape.empty())
    {
        new_total_size = 0;
        if (_total_size == 1 && new_shape.empty())
        {
        }
        else
            new_total_size = 0;
    }
    else
    {
        for (size_t dim_size : new_shape)
        {
            if (dim_size == 0)
            {
                new_total_size = 0;
                break;
            }
            new_total_size *= dim_size;
        }
    }

    if (new_total_size != _total_size && !(_total_size == 1 && new_shape.empty() && new_total_size == 0))
    {
        if (new_shape.empty() && _total_size == 1)
        {
        }
        else if (new_total_size != _total_size)
            throw std::invalid_argument("New shape total size must match current total size of reshape.");
    }

    _shape = new_shape;
    calculate_strides();
}

// Utility Implementations
template <typename T> void Tensor<T>::print(std::ostream &os) const
{
    os << "Tensor(shape: {";
    for (size_t i = 0; i < _shape.size(); i++)
    {
        os << _shape[i] << (i == _shape.size() - 1 ? "" : ", ");
    }
    os << "}, strides: {";
    for (size_t i = 0; i < _strides.size(); i++)
    {
        os << _strides[i] << (i == _strides.size() - 1 ? "" : ", ");
    }
    os << "}, data: [";

    if (is_empty())
        os << "]";
    else if (ndim() == 0)
    {
        if (!_data.empty())
            os << _data[0];
        os << "]";
    }
    else if (ndim() == 1)
    {
        for (size_t i = 0; i < _shape[0]; i++)
            os << at({i}) << (i == _shape[0] - 1 ? "" : ", ");
        os << "]";
    }
    else if (ndim() == 2)
    {
        os << "\n";
        for (size_t i = 0; i < _shape[0]; i++)
        {
            os << "  [";
            for (size_t j = 0; j < _shape[1]; j++)
                os << at({i, j}) << (j == _shape[1] - 1 ? "" : ", ");
            os << "]\n";
        }
    }
    else
    {
        size_t count = 0;
        for (const auto &val : _data)
        {
            os << val << ", ";
            count++;
            if (count > 10 && _data.size() > 15)
            {
                os << "...";
                break;
            }
        }
        if (!_data.empty() && count <= 10)
            os << "\b\b ";
        os << "]";
    }
    os << ")\n";
}

// --- Free function operator<< for printing Tensor objects ---

/**
 * @brief Overloads the stream insertion operator (<<) to enable direct printing
 *        of Tensor objects to an output stream (e.g., std::cout).
 * @tparam T The data type of the tensor elements.
 * @param os The output stream to which the tensor will be printed.
 * @param tensor The Tensor object to print.
 * @return A reference to the output stream, allowing for chaining.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
	tensor.print(os);
	return os;
}

} // namespace core
} // namespace mlib

#endif // MLIB_CORE_TENSOR_HPP
