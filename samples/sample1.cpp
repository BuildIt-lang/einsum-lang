#include "core/einsum_types.h"
#include "core/einsum_rhs_types.h"
#include "pipeline/einsum_pipeline.h"
#include "builder/builder_context.h"

// test case for the expression C[i] = A[i][j] * B[j]
static void matrix_vector_multiplication(builder::dyn_var<int*> C, builder::dyn_var<int*> A, builder::dyn_var<int*> B, int M, int N) {	
	el::current_device = el::SERIAL;

	el::index i, j;
	el::tensor<int> c({M}, C);
	el::tensor<int> a({M, N}, A);
	el::tensor<int> b({N}, B);
	
	el::current_device = el::CPU_PARALLEL;
	c[i] = a[i][j] * b[j];
	el::current_device = el::SERIAL;
}
// test case for the expression C[i][j] = A[i][k] * B[k][j]
static void matrix_matrix_multiplication(builder::dyn_var<float*> C, builder::dyn_var<float*> A, builder::dyn_var<float*> B, int M, int N, int L) {
	el::current_device = el::SERIAL;

	el::index i, j, k;
	el::tensor<float> c({M, N}, C);
	el::tensor<float> a({M, L}, A);
	el::tensor<float> b({L, N}, B);
	
	el::current_device = el::GPU_PARALLEL;
	c[i][j] = a[i][k] * b[k][j];
	el::current_device = el::SERIAL;
}

int main(int argc, char* argv[]) {	

	el::run_einsum_pipeline(
		builder::builder_context().extract_function_ast(
			matrix_vector_multiplication, "matrix_vector", 1024, 512), 
		std::cout);


	el::run_einsum_pipeline(
		builder::builder_context().extract_function_ast(
			matrix_matrix_multiplication, "matrix_matrix", 1024, 512, 1024), 
		std::cout);	

	return 0;
}
