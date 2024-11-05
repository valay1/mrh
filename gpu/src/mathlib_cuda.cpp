#if defined(_GPU_CUDA) || defined(_GPU_SYCL_CUDA)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB()
{
}

// ----------------------------------------------------------------

void MATHLIB::gemm(const char * transa, const char * transb,
	      const int * m, const int * n, const int * k,
	      const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb,
	      const double * beta, double * c, const int * ldc, cublasHandle_t & q)
{

#ifdef _SINGLE_PRECISION
  cublasSgemm(q, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
#else
  cublasDgemm(q, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
#endif
  
}

#endif
