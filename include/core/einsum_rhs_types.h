#ifndef EINSUM_RHS_TYPES_H
#define EINSUM_RHS_TYPES_H

#include "core/einsum_types.h"

namespace el {
template <typename T>
struct rhs_val_const: public rhs_val {
	const builder::dyn_var<T>* m_const;
	rhs_val_const(const builder::dyn_var<T> &val) {
		// retain a reference to the constant dyn_var
		m_const = val.addr();
	}
	std::vector<index*> get_indices() const {
		return std::vector<index*>();
	}	
	builder::builder get_value() const {
		return (*m_const);
	}
	
};



template <typename T>
struct rhs_val_tensor: public rhs_val {
	const tensor_access<T>* m_tensor_access;
	rhs_val_tensor(const tensor_access<T> &val) {
		m_tensor_access = &val;
	}
	std::vector<index*> get_indices() const {
		// Attach bounds to indices and then return
		for (builder::static_var<int> i = 0; i < (int)m_tensor_access->m_indices.size(); i++) {
			m_tensor_access->m_indices[i]->m_index_bound = m_tensor_access->m_tensor.m_sizes[i];
		}
		return m_tensor_access->m_indices;
	}
	builder::dyn_var<int> create_index(int idx) const {
		if (idx == 0)
			return *(m_tensor_access->m_indices[0]->m_iterator);
		return create_index(idx - 1) * (int) (m_tensor_access->m_tensor.m_sizes[idx]) + *(m_tensor_access->m_indices[idx]->m_iterator);
	}
	builder::builder get_value() const {	
		builder::dyn_var<int> v = create_index(m_tensor_access->m_tensor.m_dims - 1);	
		return m_tensor_access->m_tensor.m_buffer[v];
	}
};

struct rhs_val_index: public rhs_val {
	const index* m_index;
	rhs_val_index(const index &i) {
		m_index = &i;
	}
	std::vector<index*> get_indices() const {
		return std::vector<index*>();
	}	
	builder::builder get_value() const {
		return *(m_index->m_iterator);
	}
};


struct rhs_term {
	const rhs_val* m_val;
	template <typename T>
	rhs_term(const tensor_access<T> &v): m_val(new rhs_val_tensor<T>(v)) {}
	template <typename T>
	rhs_term(const builder::dyn_var<T> &v): m_val(new rhs_val_const<T>(v)) {}
	rhs_term(const index &v): m_val(new rhs_val_index(v)) {}
};



template <typename T>
rhs_terms tensor_access<T>::operator *(rhs_term v) {
	return std::move((rhs_terms)*this * v);
}

}

#endif
