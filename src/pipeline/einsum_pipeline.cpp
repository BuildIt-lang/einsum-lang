#include "pipeline/einsum_pipeline.h"
#include "blocks/rce.h"
#include "pipeline/extract_cuda.h"

namespace el {

void run_einsum_pipeline(block::block::Ptr ast, std::ostream &oss) {
	// Run a preprocessing pass to eliminate all redundant variables
	block::eliminate_redundant_vars(ast);	

	// Extract and code generate CUDA kernels one by one
	block::block::Ptr kernel;
	// This are mainly required for cooperative kernels
	// and the extra global variables they declare
	std::vector<block::decl_stmt::Ptr> new_decls;
	while (kernel = extract_single_cuda(block::to<block::func_decl>(ast)->body, new_decls)) {
		for (auto d: new_decls) {
			block::c_code_generator::generate_code(d, oss);
		}
		block::c_code_generator::generate_code(kernel, oss);
		new_decls.clear();
	}		

	einsum_code_generator::generate_code(ast, oss, 0);
}

void einsum_code_generator::visit(block::for_stmt::Ptr s) {
	std::string pragma_prefix ("pragma: ");
	if (!s->annotation.compare(0, pragma_prefix.size(), pragma_prefix)) {
		std::string pragma_value = s->annotation.substr(pragma_prefix.size());
		oss << "_Pragma(\"" << pragma_value << "\")" << std::endl;
		printer::indent(oss, curr_indent);
	}
	block::c_code_generator::visit(s);
}

}
