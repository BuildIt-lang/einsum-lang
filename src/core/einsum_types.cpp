#include "core/einsum_types.h"
#include "core/einsum_rhs_types.h"

namespace el {
builder::builder rhs_val::get_value() const {
	return (0);
}
std::vector<index*> rhs_val::get_indices() const {
	return std::vector<index*>();
}

std::vector<index*> get_reduce_indices(std::vector<index*> lhs_set, const rhs_terms& rhs) {
	// Next we will gather indices that are used on the right, but not on the left
	std::vector<index*> rhs_set;
	for (builder::static_var<size_t> i = 0; i < rhs.m_terms.size(); i++) {
		const rhs_val* val = rhs.m_terms[i];
		std::vector<index*> indices = val->get_indices();
		for (auto x: indices) {
			if (std::find(lhs_set.begin(), lhs_set.end(), x) == lhs_set.end()) {
				if (std::find(rhs_set.begin(), rhs_set.end(), x) == rhs_set.end()) {
					rhs_set.push_back(x);
				}
			}
		}
	}
	return rhs_set;
}

rhs_terms::rhs_terms(const rhs_term &v) {
	m_terms.push_back(v.m_val);
}

rhs_terms rhs_terms::operator * (const rhs_term &v) {
	m_terms.push_back(v.m_val);
	return std::move(*this);
}	

rhs_terms::~rhs_terms() {
	for (size_t i = 0; i < m_terms.size(); i++) 
		delete m_terms[i];
}

enum device_type current_device = SERIAL;
}
