#ifndef MLIB_CORE_TENSOR_HPP
#define MLIB_CORE_TENSOR_HPP

#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>

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
	explicit Tensor(const shape_type& shape);
	Tensor(const shape_type& shape, const data_type& data);
	Tensor(const shape_type& shape, T fill_value);

	// Copy semantics
	Tensor(const Tensor& other);
	Tensor& operator=(const Tensor& other);

	// Move semactics
	Tensor(Tensor&& other) noexcept;
	Tensor& operator=(Tensor&& other) noexcept;

	~Tensor() = default;

	// Accessors
	const shape_type& get_shape() const;
	const strides_type& get_strides() const;
	size_t ndim() const;
	size_t get_total_size() const;
	bool is_empty() const;
	bool is_contiguous() const;

	// Element access (const and non-const)
	//
	T& at(const std::vector<size_t>& indices);
	const T& at(const std::vector<size_t>& indices) const;

	// Variadic template for convinent access like tensor(i, j, k)
	//
	template <typename... OperatorArgs>
	T& operator()(OperatorArgs... args);
	template <typename... OperatorArgs>
	const T& operator()(OperatorArgs... args) const;

	// Direct data access
	T* data();
	const T* data() const;

	// Modifiers
	void fill(T value);
	void reshape(const shape_type& new_shape);

	// Utility
	void print(std::ostream& os = std::cout) const;

private:
	data_type _data;
	shape_type _shape;
	strides_type _strides;
	size_t _total_size = 0;

	// Private Helper Functions
	void calculate_strides();
	size_t calculate_flat_index(const std::vector<size_t>& indices) const;
	
	template <typename IndexType, typename... RestIndices>
	size_t calculate_flat_index_variadic(size_t current_dim_idx, size_t accumulated_offset, IndexType current_index, RestIndices... rest_indices) const;

	size_t calculate_flat_index_variadic(size_t current_dim_idx, size_t accumulated) const;
	
};

template <typename T>
void Tensor<T>::calculate_strides()
{
	if(_shape.empty())
	{
		_strides.clear();
		return;
	}
	_strides.resize(_shape.size());
	_strides.back() = 1;

	for(int i = static_cast<int>(_shape.size()) - 2; i >= 0; i--)
	{
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

template <typename T>
size_t Tensor<T>::calculate_flat_index(const std::vector<size_t>& indices) const
{
	if (indices.size() != _shape.size())
	{
		throw std::out_of_range("Number of indices does not match tensor dimensions.");
	}
	size_t flat_index = 0;
	for(size_t i = 0; i < _shape.size(); i++)
	{
		if (indices[i] >= _shape.size())
		{
			throw std::out_of_range("Index out of bound for dimension " + std::to_string(i));
		}
		flat_index += indices[i] * _strides[i];
	}
	return flat_index;
}

// Variadic index calculation (base case for recursion)
template <typename T>
size_t Tensor<T>::calculate_flat_index_variadic(size_t current_dim_idx, size_t accumulated_offset) const
{
	if(current_dim_idx != _shape.size())
	{
		throw std::out_of_range("Too many indicies provided for tensor dimensions.");
	}
	return accumulated_offset;
}

// Variadic index calculation (recursive step)
template<typename T>
template<typename IndexType, typename... RestIndices>
size_t Tensor<T>::calculate_flat_index_variadic(size_t current_dim_idx, size_t accumulated_offset, IndexType current_index, RestIndices... rest_indices) const
{
	if(current_dim_idx != _shape.size())
	{
		throw std::out_of_range("Too many indices provided for tensor dimension.");
	}
	if(static_cast<size_t>(current_index) >= _shape[current_dim_idx])
	{
		throw std::out_of_range("Index " + std::to_string(current_index) + " out of bound of dimension " + std::to_string(current_dim_idx) + " with size " + std::to_string(_shape[current_dim_idx]));
	}
	accumulated_offset += static_cast<size_t>(current_index) * _strides[current_dim_idx];
	return calculate_flat_index_variadic(current_dim_idx + 1, accumulated_offset, rest_indices...);
}

// Constructors & Destructor Implementations
template <typename T>
Tensor<T>::Tensor() : _total_size(0)
{
	// Empty tensor, _shape, _strides, _data are already empty std::vectors
}

template <typename T>
Tensor<T>::Tensor(const shape_type& shape) : _shape(shape)
{
	if(_shape.empty())
	{
		_total_size = 0;
	}
	else
	{
		_total_size = 1;
		for(size_t dim_size : _shape)
		{
			if(dim_size == 0)
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

template <typename T>
Tensor<T>::Tensor(const shape_type& shape, const data_type& data) : _shape(shape)
{
	_total_size = 1;
	if(shape.empty() && !data.empty() && data.size() == 1)
	{
		_total_size = 1;
	}
	else if(shape.empty() && (data.empty() || data.size() != 1))
	{
		_total_size = 0;
	}
	else
	{
		for(size_t dim_size : _shape)
		{
			if(dim_size = 0)
			{
				_total_size = 0;
				break;
			}
			_total_size *= dim_size;
		}
	}
	if(_total_size != data.size())
	{
		throw std::invalid_argument("Data size does not match shape dimensions.");
	}
	_data = data;
	calculate_strides();
}

template <typename T>
Tensor<T>::Tensor(const shape_type& shape, T fill_value) : _shape(shape)
{
	if(shape.empty())
	{
		_total_size = 0;
	}
	else 
	{
		_total_size = 1;
		for(size_t dim_size : _shape)
		{
			if(dim_size == 0)
			{
				_total_size = 0;
				break;
			}
			_total_size *= dim_size;
		}
	}
	_data.assign(_total_size, fill_value);
	calculate_strides();
}

// Copy semantics
template <typename T>
Tensor<T>::Tensor(const Tensor& other)
	: _data(other._data),
	  _shape(other._shape),
	  _strides(other._strides),
	  _total_size(other._total_size)
{}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other)
{
	if(this != &other)
	{
		_data = other._data;
		_shape = other._shape;
		_strides = other._strides;
		_total_size = other._total_size;
	}
	return *this;
}

// Move semantics
template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
	: _data(std::move(other._data)),
	  _shape(std::move(other._shape)),
	  _strides(std::move(other._strides)),
	  _total_size(std::move(other._total_size))
{
	other._total_size = 0;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept
{
	if(this != &other)
	{
		_data = std::move(other._data),
		_shape = std::move(other._shape),
		_strides = std::move(other._strides),
		_total_size = std::move(other._total_size),
		other._total_size = 0;
	}
	return *this;
}

// Accessor Implementation
template <typename T>
const typename Tensor<T>::shape_type& Tensor<T>::get_shape() const { return _shape; }

template <typename T>
const typename Tensor<T>::strides_type& Tensor<T>::get_strides() const { return _strides; }

template <typename T>
size_t Tensor<T>::ndim() const { return _shape.size(); }

template <typename T>
size_t Tensor<T>::get_total_size() const { return _total_size; }

template <typename T>
bool Tensor<T>::is_empty() const { return _total_size == 0 && _data.empty(); }

template <typename T>
bool Tensor<T>::is_contiguous() const
{
	if(is_empty()) return true;

	if(_shape.empty()) return true;
	strides_type expected_strides(_shape.size());
	expected_strides.back() = 1;
	for(int i = static_cast<int>(_shape.size()) - 2; i >= 0; i--)
	{
		expected_strides[i] = expected_strides[i + 1] * _shape[i + 1];
	}
	return _strides == expected_strides;
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices)
{
	return _data[calculate_flat_index(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const
{
	return _data[calculate_flat_index(indices)];
}

template <typename T>
template <typename... OperatorArgs>
T& Tensor<T>::operator()(OperatorArgs... args)
{
	static_assert(sizeof...(OperatorArgs) > 0 || this->ndim() == 0, "operator() requires at least one index for non-scalar tensors.");
	if(this->ndim() > 0 && sizeof...(OperatorArgs) != _shape.size())
	{
		throw std::out_of_range("Number of indices in operator() does not match tensor dimensions.");
	}
	if(this->ndim() == 0)
	{
		if constexpr (sizeof...(OperatorArgs) == 0)
		{
			if(_data.empty()) throw std::runtime_error("Accessing empty scalar tensor.");
			return _data[0];
		}
		else
			throw std::out_of_range("Indices provided for a scalar tensor.");
	}
	return _data[calculate_flat_index_variadic(0, 0, args...)];
}

template <typename T>
template <typename... OperatorArgs>
const T& Tensor<T>::operator()(OperatorArgs... args) const
{
	static_assert(sizeof...(OperatorArgs) > 0 || this->ndim() == 0, "operator() requires at least one index for non-scalar tensors.");
	if(this->ndim() > 0 && sizeof...(OperatorArgs) != _shape.size())
	{
		throw std::out_of_range("Number of indices in operator() does not match tensor dimensions.");
	}
	if(this->ndim() == 0)
	{
		if constexpr (sizeof...(OperatorArgs) == 0)
		{
			if(_data.empty()) throw std::runtime_error("Accessing empty scalar tensor.");
			return _data[0];
		}
		else
			throw std::out_of_range("Indices provided for a scalar tensor.");
	}
	return _data[calculate_flat_index_variadic(0, 0, args...)];
}

template <typename T>
T* Tensor<T>::data() { return _data.data(); }

template <typename T>
const T* Tensor<T>::data() const { return _data.data(); }

// Modifier Implementations
template <typename T>
void Tensor<T>::fill(T value)
{
	std::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void Tensor<T>::reshape(const shape_type& new_shape)
{
	size_t new_total_size = 1;
	if(new_shape.empty())
	{
		new_total_size = 0;
		if(_total_size == 1 && new_shape.empty())
		{}
		else
			new_total_size = 0;

	}
	else
	{
		for(size_t dim_size : _shape)
		{
			if(dim_size == 0)
			{
				new_total_size = 0;
				break;
			}
			new_total_size *= dim_size;
		}
	}

	if(new_total_size != _total_size && !(_total_size == 1 && new_shape.empty() && new_total_size == 0))
	{
		if(new_shape.empty() && _total_size == 1)
		{}
		else if(new_total_size != _total_size)
		{
			throw std::invalid_argument("New shape total size must match current total size of reshape.");
		}
	}

	_shape = new_shape;
	calculate_strides();
}

// Utility Implementations
template <typename T>
void Tensor<T>::print(std::ostream& os) const
{
	os << "Tensor(shape: {";
	for(size_t i = 0; i < _shape.size(); i++)
	{
		os << _shape[i] << (i == _shape.size() - 1 ? "" : ", ");
	}
	os << "}, strides: {";
	for(size_t i = 0; i < _strides.size(); i++)
	{
		os << _strides[i] << (i == _strides.size() - 1 ? "" : ", ");
	}
	os << "}, data: [";

	if(is_empty())
		os << "]";
	else if(ndim() == 0)
	{
		if(!_data.empty()) os << _data[0];
		os << "]";
	}
	else if(ndim() == 1)
	{
		for(size_t i = 0; i < _shape[0]; i++)
			os << at({i}) << (i == _shape[0] -1 ? "" : ", ");
		os << "]";
	}
	else if(ndim() == 2)
	{
		os << "\n";
		for(size_t i = 0; i < _shape[0]; i++)
		{
			os << "  [";
			for(size_t j = 0; j < _shape[1]; j++)
				os << at({i, j}) << (j == _shape[1] - 1 ? "" : ", ");
			os << "]\n";
		}
	}
	else
	{
		size_t count = 0;
		for(const auto& val : _data)
		{
			os << val << ", ";
			count++;
			if(count > 10 && _data.size() > 15)
			{
				os << "...";
				break;
			}
		}
		if(!_data.empty() && count <=10) os << "\b\b ";
		os << "]";
	}
	os << ")\n";
}

} // namespace core
} // namespace mlib

#endif // MLIB_CORE_TENSOR_HPP
