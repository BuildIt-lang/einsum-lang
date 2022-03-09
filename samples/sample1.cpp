#include "core/einsum_types.h"
#include "core/einsum_rhs_types.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"
#include "blocks/rce.h"

// test case for the expression C[i] = A[i][j] * B[j]
static void test(builder::dyn_var<int*> C, builder::dyn_var<int*> A, builder::dyn_var<int*> B, int M, int N) {	
	el::index i, j;
	el::tensor<int> c({M}, C);
	el::tensor<int> a({M, N}, A);
	el::tensor<int> b({N}, B);
	
	c[i] = a[i][j] * b[j];
}

int main(int argc, char* argv[]) {
	builder::builder_context context;
	auto ast = context.extract_function_ast(test, "foo", 128, 64);
	block::eliminate_redundant_vars(ast);
	block::c_code_generator::generate_code(ast, std::cout, 0);
	return 0;
}
