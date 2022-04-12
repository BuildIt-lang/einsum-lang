#ifndef EINSUM_TYPES_H
#define EINSUM_TYPES_H

#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "pipeline/extract_cuda.h"
#include <vector>
#include "core/tensor.h"

#define CTA_SIZE (512)

namespace el {


struct rhs_terms;
struct rhs_term;
struct index {
	// While in use, the iterator to use
	builder::dyn_var<int> m_iterator = 0;
	int m_index_bound = 0;	
};


std::vector<index*> get_reduce_indices(std::vector<index*> lhs_set, const rhs_terms& rhs);

// Base type for any RHS val
struct rhs_val {
	// For enabling dynamic inheritance
	virtual ~rhs_val() = default;
	virtual std::vector<index*> get_indices() const;
	virtual builder::builder get_value() const;
};


struct rhs_terms {
	std::vector<const rhs_val*> m_terms;	


	rhs_terms(rhs_terms&& other) {
		m_terms = std::move(other.m_terms);
	}	
	rhs_terms(const rhs_term &v);
	rhs_terms operator * (const rhs_term &v);

	~rhs_terms();
};

template <typename T>
struct tensor_access {
	tensor<T>& m_tensor;
	std::vector<index*> m_indices;

	tensor_access(tensor<T>& _t): m_tensor(_t) {}

	// Operator for multiple indices chaining
	tensor_access<T> operator [] (index &i);	

	builder::dyn_var<T> create_accum(int idx, const rhs_terms &rhs) {
		if (idx == 0)
			return rhs.m_terms[0]->get_value();
		return create_accum(idx - 1, rhs) * rhs.m_terms[idx]->get_value();
	}

	void create_increment(const rhs_terms &rhs, std::vector<index*> reduce_indices, builder::dyn_var<int>& buffer_index) {
		m_tensor.m_buffer[buffer_index] = create_accum(rhs.m_terms.size() -1 , rhs);
	}	

	builder::dyn_var<int> create_index(int idx) {
		if (idx == 0)
			return (m_indices[0]->m_iterator);
		return create_index(idx - 1) * (int) (m_tensor.m_sizes[idx]) + (m_indices[idx]->m_iterator);
	}

	void create_assign(const rhs_terms &rhs, std::vector<index*> reduce_indices) {
		builder::dyn_var<int> v = create_index(m_tensor.m_dims-1);
		if (reduce_indices.size())
			m_tensor.m_buffer[v] = 0;
		induce_reduce_loop(0, rhs, reduce_indices, v);	
	}

	
	void induce_reduce_loop(int idx, const rhs_terms &rhs, std::vector<index*> reduce_indices, 
		builder::dyn_var<int>& buffer_index);

	void induce_loops(int idx, const rhs_terms& rhs, std::vector<index*> reduce_indices);
	
		
	void operator= (const rhs_terms &rhs);

	template<typename T2>
	void operator = (const tensor_access<T2> &a) {
		*this = std::move((rhs_terms)a);
	}
	void operator = (const tensor_access<T> &a) {
		*this = std::move((rhs_terms)a);
	}
	rhs_terms operator * (rhs_term);
	void operator = (const T& x);
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
