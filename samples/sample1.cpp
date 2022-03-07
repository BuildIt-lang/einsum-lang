#include "core/einsum_types.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

static void test(void) {
	el::index i, j;
	builder::dyn_var<int*> b1;
	el::tensor<int> t1(2, {100, 100}, b1);
	t1[i][j] = 1;
}

int main(int argc, char* argv[]) {
	builder::builder_context context;
	auto ast = context.extract_function_ast(test, "foo");
	block::c_code_generator::generate_code(ast, std::cout, 0);
	return 0;
}
