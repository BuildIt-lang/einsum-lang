#include "core/einsum_types.h"
#include "core/einsum_rhs_types.h"
#include "pipeline/einsum_pipeline.h"
#include "builder/builder_context.h"

// test case for the expression C[i] = A[i][j] * B[j]
static void matrix_vector_multiplication(builder::dyn_var<int*> C, builder::dyn_var<int*> A, builder::dyn_var<int*> B, int M, int N) {	

	el::index i, j;
	el::tensor<int> c({M}, C);
	el::tensor<int> a({M, N}, A);
	el::tensor<int> b({N}, B);
	

	b[j] = 1;	
	c[i] = a[i][j] * b[j];
}

int main(int argc, char* argv[]) {	

	el::run_einsum_pipeline(
		builder::builder_context().extract_function_ast(
			matrix_vector_multiplication, "matrix_vector", 1024, 512), 
		std::cout);


	return 0;
}
