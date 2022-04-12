#ifndef EINSUM_TENSOR_ASSIGNMENT_H
#define EINSUM_TENSOR_ASSIGNMENT_H
#include "core/einsum_types.h"

namespace el {

// Functions for create loops on the RHS
template <typename T>
void tensor_access<T>::induce_reduce_loop(int idx, const rhs_terms &rhs, std::vector<index*> reduce_indices, 
		builder::dyn_var<int>& buffer_index) {
	if (idx == (int)reduce_indices.size()) {
		create_increment(rhs, reduce_indices, buffer_index);
		return;
	}
	builder::dyn_var<int> &iter = reduce_indices[idx]->m_iterator;
	for (iter = 0; iter < reduce_indices[idx]->m_index_bound; iter = iter + 1) 
		induce_reduce_loop(idx + 1, rhs, reduce_indices, buffer_index);
}	




// Functions to create loops on the LHS
template <typename T>
void tensor_access<T>::induce_loops(int idx, const rhs_terms& rhs, std::vector<index*> reduce_indices) {
	if (idx == m_tensor.m_dims) {
		create_assign(rhs, reduce_indices);
		return;
	} 	
	builder::dyn_var<int> &iter = m_indices[idx]->m_iterator;
	if (idx == 0 && current_device == GPU_PARALLEL) {
		int num_cta = (m_tensor.m_sizes[idx] + CTA_SIZE - 1) / CTA_SIZE;
		builder::annotate(CUDA_ANNOTATION_STRING);
		for (builder::dyn_var<int> cta = 0; cta < num_cta; cta = cta + 1) {
			for (builder::dyn_var<int> tid = 0; tid < CTA_SIZE; tid = tid + 1) {
				builder::dyn_var<int> thread = cta * CTA_SIZE + tid;
				if (m_tensor.m_sizes[idx] % CTA_SIZE == 0 || (bool) tid < m_tensor.m_sizes[idx]) {
					iter = thread;	
					induce_loops(idx + 1, rhs, reduce_indices);	
				}
			}
		}
	} else {
		if (idx == 0 && current_device == CPU_PARALLEL) {
			builder::annotate("pragma: openmp parallel for");
		}
		for (iter = 0; iter < m_tensor.m_sizes[idx]; iter = iter + 1) 
			induce_loops(idx + 1, rhs, reduce_indices);	
	}
}




// Operator over load for = 
template <typename T>
void tensor_access<T>::operator= (const rhs_terms &rhs) {
	// First we will assert that we have all the indices we need 
	assert(m_indices.size() == (size_t)(m_tensor.m_dims) && "Not enough indices supplied for definition");
	std::vector<index*> reduce_indices = get_reduce_indices(m_indices, rhs);	
	induce_loops(0, rhs, reduce_indices);	

	m_tensor.is_constant = false;
	m_tensor.constant_val = 0;
}

template <typename T>
void tensor_access<T>::operator = (const T& x) {
	*this = std::move((rhs_terms)(builder::dyn_var<T>)x);
	m_tensor.is_constant = true;
	m_tensor.constant_val = x;
}	

}

#endif
