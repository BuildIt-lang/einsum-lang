#ifndef EINSUM_TENSOR_H
#define EINSUM_TENSOR_H

#include "builder/dyn_var.h"
#include "builder/static_var.h"

namespace el {

template<typename T>
struct tensor_access;

struct index;

template <typename T>
struct tensor {
	int m_dims;

	// Statically known tensor sizes
	std::vector<int> m_sizes;

	// Underlying data buffer
	builder::dyn_var<T*> m_buffer;
		
	builder::static_var<int> is_constant = false;
	builder::static_var<T> constant_val = 0;

	tensor(const std::vector<int>& sizes, const builder::dyn_var<T*>& buffer):
		m_dims(sizes.size()), m_sizes(std::move(sizes)), m_buffer(buffer) {
	}

	tensor_access<T> operator [] (index &i);
	
};

enum device_type {
	SERIAL,
	CPU_PARALLEL,
	GPU_PARALLEL,	
};

extern enum device_type current_device;


}

#endif
