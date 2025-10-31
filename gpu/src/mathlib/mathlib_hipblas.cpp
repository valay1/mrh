#if defined(_GPU_HIPBLAS)

#include "mathlib.h"

using namespace MATHLIB_NS;

// ----------------------------------------------------------------

MATHLIB::MATHLIB(class PM_NS::PM * pm_)
{
  pm = pm_;
}

// ----------------------------------------------------------------

MATHLIB::~MATHLIB()
{
#if defined(_PROFILE_ML)
  printf("\nLIBGPU :: PROFILE_ML\n");
  for(int i=0; i<profile_name.size(); ++i) {
    printf("LIBGPU :: PROFILE_ML :: count= %i  name= %s\n", profile_count[i], profile_name[i].c_str());
  }
#endif
}

// ----------------------------------------------------------------

int MATHLIB::create_handle()
{
  hipblasHandle_t h;

  hipblasCreate(&h);
  
  _HIP_CHECK_ERRORS();
  
  hipStream_t * s = pm->dev_get_queue();
  
  hipblasSetStream(h, *s);
  
  _HIP_CHECK_ERRORS();

  my_handles.push_back(h);

  int id = my_handles.size() - 1;

  set_handle(id);
  
  return id;
}

// ----------------------------------------------------------------

void MATHLIB::set_handle(int id)
{
  current_handle = &(my_handles[id]);
  current_handle_id = id;
}

// ----------------------------------------------------------------

void MATHLIB::set_handle()
{
  int id = pm->dev_get_device();
  
  current_handle = &(my_handles[id]);
  current_handle_id = id;
}

// ----------------------------------------------------------------

hipblasHandle_t * MATHLIB::get_handle()
{  
  return current_handle;
}

// ----------------------------------------------------------------

void MATHLIB::destroy_handle()
{
  int id = current_handle_id;
  
  hipblasDestroy(my_handles[id]);
  my_handles[id] = NULL;
}

// ----------------------------------------------------------------

void MATHLIB::memset(double * array, const int * num, const int * size)
{
#ifdef _DEBUG_ML 
  printf("Inside MATHLIB::memset()\n");
#endif
//TODO: add profiling lines related things

#if 1
  hipStream_t * s = pm->dev_get_queue();
  
  hipMemsetAsync ( array, *num, *size, *s);
#else
  hipMemset ( array, *num, *size);
#endif
  
  _HIP_CHECK_ERRORS();

#ifdef _DEBUG_ML 
  printf(" -- Leaving MATHLIB::memset()\n");
#endif

}

// ----------------------------------------------------------------

void MATHLIB::axpy(const int * n, 
                   const double * alpha, const double * x, const int * incx,
                   double * y, const int * incy)
{
#ifdef _DEBUG_ML 
  printf("Inside MATHLIB::axpy()\n");
#endif
//TODO: add profiling lines related things

  hipblasHandle_t * h = current_handle;
  
#ifdef _SINGLE_PRECISION
  hipblasSaxpy (*h , *n, alpha, x, *incx, y, *incy);   
#else
  hipblasDaxpy (*h , *n, alpha, x, *incx, y, *incy);   
#endif
  
  _HIP_CHECK_ERRORS();

#ifdef _DEBUG_ML 
  printf(" -- Leaving MATHLIB::axpy()\n");
#endif

}

// ----------------------------------------------------------------

void MATHLIB::gemv_batch(const char * transa,
		   const int * m, const int * n, 
		   const double * alpha, const double * a, const int * lda, const int * strideA,
		   const double * x, const int * incx, const int * strideX,
		   const double * beta, double * y, const int * incy, const int * strideY,
                   const int * batchCount)
{
#ifdef _DEBUG_ML
  printf("Inside MATHLIB::gemv()\n");
#endif

  hipblasHandle_t * h = current_handle;
  
  hipblasOperation_t ta;
  
  if(strcmp(transa, "N") == 0) ta = CUBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = CUBLAS_OP_T;
  else ta = CUBLAS_OP_C;

#ifdef _SINGLE_PRECISION
  hipblasSgemvStridedBatched(*h, ta, *m, *n, 
                            alpha, a, *lda, *strideA, x, *incx, *strideX, beta, y, *incy, *strideY, *batchCount);
#else
  hipblasDgemvStridedBatched(*h, ta, *m, *n, 
                            alpha, a, *lda, *strideA, x, *incx, *strideX, beta, y, *incy, *strideY, *batchCount);
#endif
  
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf(" -- Leaving MATHLIB::gemv_batch()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemv(const char * transa,
		   const int * m, const int * n, 
		   const double * alpha, const double * a, const int * lda,
		   const double * x, const int * incx,
		   const double * beta, double * y, const int * incy)
{
#ifdef _DEBUG_ML
  printf("Inside MATHLIB::gemv()\n");
#endif

//#if defined(_PROFILE_ML)
#if 0
  std::ostringstream name_;
  name_ << "gemv " << transa << " "  << *m << " " << *n << " "
	<< *lda << " " << *ldb << " " << *ldc << " " << *alpha << " " << *beta;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif
  
  hipblasHandle_t * h = current_handle;
  
  hipblasOperation_t ta;
  
  if(strcmp(transa, "N") == 0) ta = CUBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = CUBLAS_OP_T;
  else ta = CUBLAS_OP_C;

#ifdef _SINGLE_PRECISION
  hipblasSgemv(*h, ta, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
#else
  hipblasDgemv(*h, ta, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
#endif
  
  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf(" -- Leaving MATHLIB::gemv()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemm(const char * transa, const char * transb,
		   const int * m, const int * n, const int * k,
		   const double * alpha, const double * a, const int * lda,
		   const double * b, const int * ldb,
		   const double * beta, double * c, const int * ldc)
{
#ifdef _DEBUG_ML
  printf("Inside MATHLIB::gemm()\n");
#endif

#if defined(_PROFILE_ML)
  std::ostringstream name_;
  name_ << "gemm " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " "
	<< *lda << " " << *ldb << " " << *ldc << " " << *alpha << " " << *beta;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif
  
  hipblasHandle_t * h = current_handle;
  
  hipblasOperation_t ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = HIPBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = HIPBLAS_OP_T;
  else ta = HIPBLAS_OP_C;

  if(strcmp(transb, "N") == 0) tb = HIPBLAS_OP_N;
  else if(strcmp(transb, "T") == 0) tb = HIPBLAS_OP_T;
  else tb = HIPBLAS_OP_C;
  
#ifdef _SINGLE_PRECISION
  hipblasSgemm(*h, ta, tb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#else
  hipblasDgemm(*h, ta, tb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
#endif
  
  _HIP_CHECK_ERRORS();

#ifdef _DEBUG_ML
  printf(" -- Leaving MATHLIB::gemm()\n");
#endif
}

// ----------------------------------------------------------------

void MATHLIB::gemm_batch(const char * transa, const char * transb,
			 const int * m, const int * n, const int * k,
			 const double * alpha, const double * a, const int * lda, const int * strideA,
			 const double * b, const int * ldb, const int * strideB,
			 const double * beta, double * c, const int * ldc, const int * strideC,
			 const int * batchCount)
{
#ifdef _DEBUG_ML
  printf("Inside MATHLIB::gemm_batch()\n");
  printf("mnk= %i %i %i  alpha= %f  beta= %f  ld= %i %i %i  stride= %i %i %i  batchCount= %i\n",
	 *m,*n,*k,*alpha,*beta,*lda,*ldb,*ldc,*strideA,*strideB,*strideC,*batchCount);
#endif
  
#if defined(_PROFILE_ML)
  std::ostringstream name_;
  name_ << "gemm_batch " << transa << " " << transb << " " << *m << " " << *n << " " << *k << " " <<
    << *lda << " " << *ldb << " " << *ldc << " " *alpha << " " << *beta << " " << *batchCount;
  std::string name = name_.str();

  auto it_ = std::find(profile_name.begin(), profile_name.end(), name);

  int indx = it_ - profile_name.begin();

  if(indx < profile_name.size()) profile_count[indx]++;
  else {
    profile_name.push_back(name);
    profile_count.push_back(1);
  }
#endif

  hipblasHandle_t * h = current_handle;

  hipblasOperation_t ta, tb;
  
  if(strcmp(transa, "N") == 0) ta = HIPBLAS_OP_N;
  else if(strcmp(transa, "T") == 0) ta = HIPBLAS_OP_T;
  else ta = HIPBLAS_OP_C;

  if(strcmp(transb, "N") == 0) tb = HIPBLAS_OP_N;
  else if(strcmp(transb, "T") == 0) tb = HIPBLAS_OP_T;
  else tb = HIPBLAS_OP_C;
  
#ifdef _SINGLE_PRECISION
  hipblasSgemmStridedBatched(*h, ta, tb, *m, *n, *k,
			    alpha, a, *lda, *strideA, b, *ldb, *strideB, beta, c, *ldc, *strideC, *batchCount);
#else
  hipblasDgemmStridedBatched(*h, ta, tb, *m, *n, *k,
			    alpha, a, *lda, *strideA, b, *ldb, *strideB, beta, c, *ldc, *strideC, *batchCount);
#endif

  _HIP_CHECK_ERRORS();
  
#ifdef _DEBUG_ML
  printf("Leaving MATHLIB::gemm_batch()\n");
#endif
}

#endif
