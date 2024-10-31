#if defined(_GPU_CUDA)

#include <stdio.h>
#include <iostream>

#include <iomanip>
#include <vector>
#include <tuple>

#include "pm.h"

//#define _DEBUG_PM

using namespace PM_NS;

PM::PM()
{
}

//https://stackoverflow.com/questions/68823023/set-cuda-device-by-uuid
void PM::uuid_print(cudaUUID_t a){
  std::cout << "GPU";
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  for (auto t : r){
    std::cout << "-";
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      std::cout << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)a.bytes[i];
  }
  std::cout << std::endl;
}

int PM::dev_num_devices()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_num_devices()\n");
#endif
  int num_devices;
  
  cudaGetDeviceCount(&num_devices);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_num_devices() : num_devices= %i\n",num_devices);
#endif
  
  return num_devices;
}

void PM::dev_properties(int ndev)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_properties()\n");
#endif
  
  for(int i=0; i<ndev; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    _CUDA_CHECK_ERRORS();

    char name[256];
    strcpy(name, prop.name);

    printf("LIBGPU ::  [%i] Platform[ Nvidia ] Type[ GPU ] Device[ %s ]  uuid= ", i, name);
    uuid_print(prop.uuid);
    printf("\n");
  }

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_properties()\n");
#endif
}

int PM::dev_check_peer(int rank, int ngpus)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_check_peer()\n");
#endif
  
  int err = 0;
  if(rank == 0) printf("\nChecking P2P Access\n");
  for(int ig=0; ig<ngpus; ++ig) {
    cudaSetDevice(ig);
    //if(rank == 0) printf("Device i= %i\n",ig);

    int n = 1;
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) {
        int access;
        cudaDeviceCanAccessPeer(&access, ig, jg);
        n += access;

        //if(rank == 0) printf("  --  Device j= %i  access= %i\n",jg,access);
      }
    }
    if(n != ngpus) err += 1;
  }

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_peer()\n");
#endif
  
  return err;
}

void PM::dev_set_device(int id)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_set_device()\n");
#endif
  
  cudaSetDevice(id);
  _CUDA_CHECK_ERRORS();

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_set_devices()\n");
#endif
}

int PM::dev_get_device()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_get_device()\n");
#endif
  
  int id;
  cudaGetDevice(&id);
  _CUDA_CHECK_ERRORS();

#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_get_device() : id= %i\n",id);
#endif
  
  return id;
}

void * PM::dev_malloc(size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc()\n");
#endif
  
  void * ptr;
  cudaMalloc((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc()\n");
#endif
  
  return ptr;
}

void * PM::dev_malloc_async(size_t N, cudaStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_async()\n");
#endif
  
  void * ptr;
  cudaMallocAsync((void**) &ptr, N, s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_async()\n");
#endif
  
  return ptr;
}

void * PM::dev_malloc_host(size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_malloc_host()\n");
#endif
  
  void * ptr;
  cudaMallocHost((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_malloc_host()\n");
#endif
  
  return ptr;
}

void PM::dev_free(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free()\n");
#endif
  
  if(ptr) cudaFree(ptr);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free()\n");
#endif
}

void PM::dev_free_async(void * ptr, cudaStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_async()\n");
#endif
  
  if(ptr) cudaFreeAsync(ptr, s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_async()\n");
#endif
}

void PM::dev_free_host(void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_free_host()\n");
#endif
  
  if(ptr) cudaFreeHost(ptr);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_free_host()\n");
#endif
}

void PM::dev_push(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push()\n");
#endif
  
  cudaMemcpy(d_ptr, h_ptr, N, cudaMemcpyHostToDevice);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push()\n");
#endif
}

int PM::dev_push_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_push_async()\n");
#endif
  
  cudaMemcpyAsync(d_ptr, h_ptr, N, cudaMemcpyHostToDevice, s);
  _CUDA_CHECK_ERRORS2();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_push_async()\n");
#endif
  
  return 0;
}

void PM::dev_pull(void * d_ptr, void * h_ptr, size_t N)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull()\n");
#endif
  
  cudaMemcpy(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull()\n");
#endif
}

void PM::dev_pull_async(void * d_ptr, void * h_ptr, size_t N, cudaStream_t &s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_pull_async()\n");
#endif
  
  cudaMemcpyAsync(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost, s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_pull_async()\n");
#endif
}

void PM::dev_copy(void * dest, void * src, size_t N)
{ 
#ifdef _DEBUG_PM
  printf("Inside PM::dev_copy()\n");
#endif
  
  printf("correct usage dest vs. src??]n");
  cudaMemcpy(dest, src, N, cudaMemcpyDeviceToDevice);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_copy()\n");
#endif
}

void PM::dev_check_pointer(int rnk, const char * name, void * ptr)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_check_pointer()\n");
#endif
  
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);
  if(attributes.devicePointer != NULL) printf("(%i) ptr %s is devicePointer\n",rnk,name);
  if(attributes.hostPointer != NULL) printf("(%i) ptr %s is hostPointer\n",rnk,name);
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_check_pointer()\n");
#endif
}

void PM::dev_barrier()
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_barrier()\n");
#endif
  
  cudaDeviceSynchronize();
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_barrier()\n");
#endif
}

void PM::dev_stream_create(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_create()\n");
#endif
  
  cudaStreamCreate(&s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_create()\n");
#endif
}

void PM::dev_stream_destroy(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_destroy()\n");
#endif
  
  cudaStreamDestroy(s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_destroy()\n");
#endif
}

void PM::dev_stream_wait(cudaStream_t & s)
{
#ifdef _DEBUG_PM
  printf("Inside PM::dev_stream_wait()\n");
#endif
  
  cudaStreamSynchronize(s);
  _CUDA_CHECK_ERRORS();
  
#ifdef _DEBUG_PM
  printf(" -- Leaving PM::dev_stream_wait()\n");
#endif
}

#endif