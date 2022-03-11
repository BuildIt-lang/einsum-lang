#ifndef EINSUM_TYPES_H
#define EINSUM_TYPES_H

#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "pipeline/extract_cuda.h"
#include <vector>

#define CTA_SIZE (512)

namespace el {


struct rhs_terms;
struct rhs_term;
struct index {
	// index doesn't have any members	
	// All we need from this is address comparison

	// While in use, they will hold a reference to an index	
	builder::dyn_var<int> * m_iterator = nullptr;
	int m_index_bound = 0;	
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

	tensor(const std::vector<int>& sizes, const builder::dyn_var<T*>& buffer):
		m_dims(sizes.size()), m_sizes(std::move(sizes)), m_buffer(buffer) {
	}

	tensor_access<T> operator [] (index &i);
	
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

enum device_type {
	SERIAL = 0,
	CPU_PARALLEL = 1,
	GPU_PARALLEL = 2
};
extern enum device_type current_device;

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

	void create_increment(const rhs_terms &rhs, builder::dyn_var<int>& buffer_index) {
		m_tensor.m_buffer[buffer_index] = m_tensor.m_buffer[buffer_index] + create_accum(rhs.m_terms.size() -1 , rhs);
	}	

	void induce_reduce_loop(int idx, const rhs_terms &rhs, std::vector<index*> reduce_indices, 
		builder::dyn_var<int>& buffer_index) {
		if (idx == (int)reduce_indices.size()) {
			create_increment(rhs, buffer_index);
			return;
		}
		// Now add a new loop for a reduce index	
		for (builder::dyn_var<int> iter = 0; iter < reduce_indices[idx]->m_index_bound; iter = iter + 1) {
			reduce_indices[idx]->m_iterator = iter.addr();
			induce_reduce_loop(idx + 1, rhs, reduce_indices, buffer_index);
			reduce_indices[idx]->m_iterator = nullptr;
		}
	}
	
	builder::dyn_var<int> create_index(int idx) {
		if (idx == 0)
			return *(m_indices[0]->m_iterator);
		return create_index(idx - 1) * (int) (m_tensor.m_sizes[idx]) + *(m_indices[idx]->m_iterator);
	}

	void create_assign(const rhs_terms &rhs, std::vector<index*> reduce_indices) {
		builder::dyn_var<int> v = create_index(m_tensor.m_dims-1);
		m_tensor.m_buffer[v] = 0;
		induce_reduce_loop(0, rhs, reduce_indices, v);	
	}

	void induce_loops(int idx, const rhs_terms& rhs, std::vector<index*> reduce_indices) {
		if (idx == m_tensor.m_dims) {
			create_assign(rhs, reduce_indices);
			return;
		} 	
		if (idx == 0 && current_device == GPU_PARALLEL) {
			int num_cta = (m_tensor.m_sizes[idx] + CTA_SIZE - 1) / CTA_SIZE;
			builder::annotate(CUDA_ANNOTATION_STRING);
			for (builder::dyn_var<int> cta = 0; cta < num_cta; cta = cta + 1) {
				for (builder::dyn_var<int> tid = 0; tid < CTA_SIZE; tid = tid + 1) {
					builder::dyn_var<int> thread = cta * CTA_SIZE + tid;
					if ((m_tensor.m_sizes[idx] % CTA_SIZE == 0) || (bool)(thread < m_tensor.m_sizes[idx])) {
						m_indices[idx]->m_iterator = thread.addr();
						induce_loops(idx + 1, rhs, reduce_indices);	
						m_indices[idx]->m_iterator = nullptr;	
					}
				}
			}
		} else {
			// Implement a level of loop and recurse	
			if (idx == 0 && current_device == CPU_PARALLEL) {
				builder::annotate("pragma: omp parallel for");
			}
			for (builder::dyn_var<int> iter = 0; iter < m_tensor.m_sizes[idx]; iter = iter + 1) {
				// Register the loop variable with the index var
				m_indices[idx]->m_iterator = iter.addr();
				induce_loops(idx + 1, rhs, reduce_indices);	
				// While returning unset the iterators
				m_indices[idx]->m_iterator = nullptr;
			}
		}
	}
		
	// Operator for initializing LHS to a constant
	void operator= (const rhs_terms &rhs) {
		// First we will assert that we have all the indices we need 
		assert(m_indices.size() == (size_t)(m_tensor.m_dims) && "Not enough indices supplied for definition");
		std::vector<index*> reduce_indices = get_reduce_indices(m_indices, rhs);	
		induce_loops(0, rhs, reduce_indices);	
	}
	template<typename T2>
	void operator = (const tensor_access<T2> &a) {
		*this = std::move((rhs_terms)a);
	}
	void operator = (const tensor_access<T> &a) {
		*this = std::move((rhs_terms)a);
	}
	rhs_terms operator * (rhs_term);
	
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
