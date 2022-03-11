#ifndef EINSUM_PIPELINE_H
#define EINSUM_PIPELINE_H

#include "blocks/block.h"
#include "blocks/c_code_generator.h"

namespace el {

class einsum_code_generator: public block::c_code_generator {
	using block::c_code_generator::visit;
	using block::c_code_generator::c_code_generator;
	virtual void visit(block::for_stmt::Ptr);
public:
	static void generate_code(block::block::Ptr ast, std::ostream &oss, int indent = 0) {
		einsum_code_generator generator(oss);
		generator.curr_indent = indent;
		ast->accept(&generator);
		oss << std::endl;
	}
};

extern void run_einsum_pipeline(block::block::Ptr, std::ostream &oss);

}

#endif
