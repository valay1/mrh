#if defined(_GPU_MKL)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB(class PM_NS::PM * pm)
{
  pm_ = pm;
}

// ----------------------------------------------------------------

void MATHLIB::gemm(const char * transa, const char * transb,
		   const int * m, const int * n, const int * k,
		   const double * alpha, const double * a, const int * lda,
		   const double * b, const int * ldb,
		   const double * beta, double * c, const int * ldc)
{  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = transpose::nontrans;
  else if(strcmp(transa, "T") == 0) ta = transpose::trans;
  else ta = transpose::conjtrans;
  
  if(strcmp(transb, "N") == 0) tb = transpose::nontrans;
  else if(strcmp(transb, "T") == 0) tb = transpose::trans;
  else tb = transpose::conjtrans;
    
#if defined(_GPU_SYCL_CUDA)
  oneapi::mkl::blas::column_major::gemm(*q, ta, tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
#else
  oneapi::mkl::blas::gemm(*q, ta, tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
#endif
  
}

// ----------------------------------------------------------------

void MATHLIB::gemm_batch(const char * transa, const char * transb,
			 const int * m, const int * n, const int * k,
			 const double * alpha, const double * a, const int * lda, const int * strideA,
			 const double * b, const int * ldb, const int * strideB,
			 const double * beta, double * c, const int * ldc, const int * strideC, const int * batchCount)
{  
  sycl::queue * q = pm_->dev_get_queue();

  using oneapi::mkl::transpose;
  
  transpose ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = transpose::nontrans;
  else if(strcmp(transa, "T") == 0) ta = transpose::trans;
  else ta = transpose::conjtrans;
  
  if(strcmp(transb, "N") == 0) tb = transpose::nontrans;
  else if(strcmp(transb, "T") == 0) tb = transpose::trans;
  else tb = transpose::conjtrans;
    
#if defined(_GPU_SYCL_CUDA)  
  oneapi::mkl::blas::column_major::gemm_batch(*q, ta, tb, *m, *n, *k, *alpha,
					      a, *lda, *strideA, b, *ldb, *strideB, *beta, c, *ldc, *strideC, *batchCount);
#else  
  oneapi::mkl::blas::gemm_batch(*q, ta, tb, *m, *n, *k, *alpha,
				a, *lda, *strideA, b, *ldb, *strideB, *beta, c, *ldc, *strideC, *batchCount);
#endif
  
}

#endif