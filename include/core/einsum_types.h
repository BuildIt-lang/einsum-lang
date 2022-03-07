#ifndef EINSUM_TYPES_H
#define EINSUM_TYPES_H

#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include <vector>

namespace el {

struct index {
	// index doesn't have any members	
	// All we need from this is address comparison

	// While in use, they will hold a reference to an index	
	builder::dyn_var<int> * m_iterator = nullptr;
	
};

template<typename T>
struct tensor_access;

template <typename T>
struct tensor {
	int m_dims;

	// Statically known tensor sizes
	std::vector<int> m_sizes;
	
	// Underlying data buffer
	builder::dyn_var<T*> m_buffer;

	tensor(const int dims, const std::vector<int>& sizes, const builder::dyn_var<T*>& buffer):
		m_dims(dims), m_sizes(std::move(sizes)), m_buffer(buffer) {

		assert((size_t)dims == m_sizes.size() && "Not enough sizes supplied\n");
	}

	tensor_access<T> operator [] (index &i);
	
};

template <typename T>
struct tensor_access {
	tensor<T>& m_tensor;
	std::vector<index*> m_indices;

	tensor_access(tensor<T>& _t): m_tensor(_t) {}

	// Operator for multiple indices chaining
	tensor_access<T> operator [] (index &i);	


	void create_assign(builder::dyn_var<T> &rhs) {
		builder::dyn_var<int> v = 0;
		for (builder::static_var<int> i = 0; i < m_tensor.m_dims; i++) {
			v = v * (int)(m_tensor.m_sizes[i]) + *(m_indices[i]->m_iterator);
		}
		m_tensor.m_buffer[v] = rhs;
	}

	void induce_loops(int idx, builder::dyn_var<T>& rhs) {
		// Implement a level of loop and recurse	
		for (builder::dyn_var<int> iter = 0; iter < m_tensor.m_sizes[idx]; iter = iter + 1) {
			// Register the loop variable with the index var
			m_indices[idx]->m_iterator = iter.addr();
			if ((idx + 1) < (m_tensor.m_dims)) {
				// Recurse
				induce_loops(idx + 1, rhs);	
			} else {
				// We have covered all the indices
				// Now we compute the offset and assign
				create_assign(rhs);
			}
			// While returning unset the iterators
			m_indices[idx]->m_iterator = nullptr;
		}
	}
		
	// Operator for initializing LHS to a constant
	void operator= (builder::dyn_var<T> rhs) {
		// First we will assert that we have all the indices we need 
		assert(m_indices.size() == (size_t)(m_tensor.m_dims) && "Not enough indices supplied for definition");
		
		induce_loops(0, rhs);	
	}
};


// Operator definitions
template <typename T>
tensor_access<T> tensor<T>::operator [] (index &i) {
	tensor_access<T> t (*this);
	t.m_indices.push_back(&i);	
	return t;
}

template <typename T>
tensor_access<T> tensor_access<T>::operator [] (index &i) {
	tensor_access<T> t(m_tensor);

	// We can't use this tensor access anymore after this
	t.m_indices = std::move(m_indices);
	t.m_indices.push_back(&i);
	return t;
}

}

#endif
