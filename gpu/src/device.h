/* -*- c++ -*- */

#ifndef DEVICE_H
#define DEVICE_H

#include <chrono>
#include <math.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "pm/pm.h"
#include "mathlib/mathlib.h"
#include "pm/dev_array.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

#define _SIZE_GRID 32
#define _SIZE_BLOCK 256

#define _USE_ERI_CACHE
#define _ERI_CACHE_EXTRA 2

#define _ENABLE_P2P

//#define _DEBUG_DEVICE
//#define _DEBUG_P2P
//#define _DEBUG_FCI
//#define _TEMP_BUFSIZING
//#define _CUSTOM_FCI

#define _PUMAP_2D_UNPACK 0       // generic unpacking of 1D array to 2D matrix
#define _PUMAP_H2EFF_UNPACK 1    // unpacking h2eff array (generic?)
#define _PUMAP_H2EFF_PACK 2      // unpacking h2eff array (generic?)

#define OUTPUTIJ        1
#define INPUT_IJ        2

// pyscf/pyscf/lib/np_helper/np_helper.h
#define BLOCK_DIM    104

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

#define TRIU_LOOP(I, J) \
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) \
                for (I = 0, j1 = MIN(j0+BLOCK_DIM, n); I < j1; I++) \
                        for (J = MAX(I,j0); J < j1; J++)

extern "C" {
  void dsymm_(const char*, const char*, const int*, const int*,
	      const double*, const double*, const int*,
	      const double*, const int*,
	      const double*, double*, const int*);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
	      const int * k, const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb, const double * beta, double * c,
	      const int * ldc);
}

class Device {
  
public :
  
  Device();
  ~Device();
  
  //SETUP
  int get_num_devices();
  void get_dev_properties(int);
  void set_device(int);
  void barrier();
  void disable_eri_cache_();
  void set_verbose_(int);

  //JK
  void init_get_jk(py::array_t<double>, py::array_t<double>, int, int, int, int, int);
  void get_jk(int, int, int,
	      py::array_t<double>, py::array_t<double>, py::list &,
	      py::array_t<double>, py::array_t<double>,
	      int, int, size_t);
  void pull_get_jk(py::array_t<double>, py::array_t<double>, int, int, int);

  void getjk_rho(double *, double *, double *, int, int, int);
  void getjk_vj(double *, double *, double *, int, int, int, int);
  void getjk_unpack_buf2(double *, double *, int *, int, int, int);
  void transpose(double*, double*, int, int);
  
  void set_update_dfobj_(int);
  void get_dfobj_status(size_t, py::array_t<int>);
 
  //AO2MO
  void init_jk_ao2mo (int, int);

  void init_ppaa_papa_ao2mo (int, int);
 
  void df_ao2mo_v4 (int, int, int, int, int, int,
			    int, size_t);
  void get_bufpa(const double *, double *, int, int, int, int);
  void get_bufaa(const double *, double *, int, int, int, int);
  void transpose_120(double *, double *, int, int, int, int order = 0);
  void get_bufd(const double *, double *, int, int);
  void pull_jk_ao2mo_v4 (py::array_t<double>,py::array_t<double>,int, int);
  void pull_ppaa_papa_ao2mo_v4 (py::array_t<double>,py::array_t<double>, int, int);
  
  //ORBITAL RESPONSE
  void orbital_response(py::array_t<double>,
			py::array_t<double>, py::array_t<double>, py::array_t<double>,
			py::array_t<double>, py::array_t<double>, py::array_t<double>,
			int, int, int);

  //UPDATE H2EFF
  void update_h2eff_sub(int, int, int, int,
                        py::array_t<double>,py::array_t<double>);
  void extract_submatrix(const double *, double *, int, int, int);
  void unpack_h2eff_2d(double *, double *, int *, int, int, int);
  void transpose_2310(double *, double *, int, int);
  void transpose_3210(double *, double *, int, int);
  void pack_h2eff_2d(double *, double *, int *, int, int, int);
  
  void transpose_210(double *, double *, int, int, int);
  
  //LAS_AO2MO
  void init_eri_h2eff( int, int);//VA: new function
  void get_h2eff_df_v2 ( py::array_t<double>, 
                         int, int, int, int, int, 
                         py::array_t<double>, int, size_t);//VA: new function
  void pull_eri_h2eff(py::array_t<double>, int, int);// VA: new function
  
  //ERI for IMPURINTY HAMILTONIAN
  void init_eri_impham(int, int, int);
  void compute_eri_impham(int, int, int, int, int, size_t, int);
  void pull_eri_impham( py::array_t<double>, int, int, int);
  void compute_eri_impham_v2(int, int, int, int, int, size_t, size_t);
  void pack_eri(double *, double *, int *, int, int, int); 
  
  //PDFT
  void init_mo_grid(int, int);
  void push_ao_grid(py::array_t<double>, int, int, int);
  void compute_mo_grid(int, int, int);
  void pull_mo_grid(py::array_t<double>, int, int);
  void init_Pi(int);
  void push_cascm2 (py::array_t<double>, int); 
  void compute_rho_to_Pi (py::array_t<double>, int, int); 
  void compute_Pi (int, int, int, int); 
  void pull_Pi (py::array_t<double>, int, int); 

  //FCI
  //struct my_LinkT {
  //  unsigned int addr;
  //  uint8_t a;
  //  uint8_t i;
  //  int8_t sign;
  //  };
  void init_tdm1(int);
  void init_tdm2(int);
  void init_tdm3hab(int);
  void init_tdm1_host(int);
  void init_tdm2_host(int);
  void init_tdm3h_host(int);
  void copy_bravecs_host(py::array_t<double>, int , int, int);
  void copy_ketvecs_host(py::array_t<double>, int , int, int);
  void push_cibra_from_host(int, int , int, int);
  void push_ciket_from_host(int, int , int, int);

  void push_cibra(py::array_t<double>, int , int, int);
  void push_ciket(py::array_t<double>, int , int, int);
  void push_link_indexa(int, int , py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void push_link_indexb(int, int , py::array_t<int> ); //TODO: figure out the shape? or maybe move the compressed version 
  void compute_trans_rdm1a(int , int , int , int , int, int );
  void compute_trans_rdm1b(int , int , int , int , int, int );
  void compute_make_rdm1a(int , int , int , int , int, int );
  void compute_make_rdm1b(int , int , int , int , int, int );
  void compute_tdm12kern_a_v2(int , int , int , int , int, int );
  void compute_tdm12kern_b_v2(int , int , int , int , int, int );
  void compute_tdm12kern_ab_v2(int , int , int , int , int, int );
  void compute_rdm12kern_sf_v2(int , int , int , int , int, int );
  void compute_tdm13h_spin_v4( int , int , int , int , int , int, int,
                               int , int , int , int , int ,
                               int , int , int , int , int , int);
  void compute_tdm13h_spin_v5( int , int , int , int , int , int, int,
                               int , int , int , int , int ,
                               int , int , int , int , int , int);
  void compute_tdmpp_spin_v4( int , int , int , int , int , int, 
                               int , int , int , int , int ,
                               int , int , int , int , int , int);
  void compute_sfudm_v2( int , int , int , int , int,  
                      int , int , int , int , int ,
                      int , int , int , int , int , int);
  void compute_tdm1h_spin( int , int , int , int , int , int,
                           int , int , int , int , int ,
                           int , int , int , int , int , int);

  void reorder_rdm(int, int);
  void transpose_tdm2(int, int);
  void pull_tdm1(py::array_t<double> , int, int );
  void pull_tdm2(py::array_t<double> , int, int );

  void pull_tdm1_host(int, int, int, int, int, int, int);
  void pull_tdm2_host(int, int, int, int, int, int, int);
  void pull_tdm3h_host(int, int, int);
  void pull_tdm3hab(py::array_t<double> ,py::array_t<double> , int, int );
  void pull_tdm3hab_v2(py::array_t<double>, py::array_t<double> ,py::array_t<double> , int, int, int, int );
  void pull_tdm3hab_v2_host(int, int, int, int, int, int, int, int );

  void copy_tdm1_host_to_page(py::array_t<double> , int );
  void copy_tdm2_host_to_page(py::array_t<double> , int );

  //inner functions
  void extract_mo_cas(int, int, int);//TODO: fix the difference - changed slightly
  void get_mo_cas(const double *, double *, int, int, int);
  void pack_d_vuwM(const double *, double *, int *, int, int, int);
  void pack_d_vuwM_add(const double *, double *, int *, int, int, int);

  void push_mo_coeff(py::array_t<double>, int);

  void vecadd(const double *, double *, int); // replace with ml->daxpy()
  void vecadd_batch(const double *, double *, int, int);
  void memset_zero_batch_stride(double *, int, int, int, int);
  void get_rho_to_Pi(double *, double * ,int); // replace with gemm or element wise multiplication
  void make_gridkern(double *, double *, int, int); //replace with ml->gemm()
  void make_buf_pdft(double *, double *, double *, int, int); //replace with ml->gemm()
  void make_Pi_final(double *, double *,double *, int, int); // replace with ml->gemm()
  //FCI
  //void FCIcompress_link (my_LinkT *, int, int, int, int); 
  void set_to_zero(double *, int);
  void transpose_jikl(double *, double *, int);
  void veccopy(const double *, double *, int);
  void compute_FCItrans_rdm1a (double *, double *, double *, int, int, int, int, int *);
  void compute_FCItrans_rdm1b (double *, double *, double *, int, int, int, int, int *);
  void compute_FCItrans_rdm1a_v2 (double *, double *, double *, 
                                 int, int, 
                                 int, int, int, int,  
                                 int, int, int, int, int,  
                                 int *);
  void compute_FCItrans_rdm1b_v2 (double *, double *, double *, 
                                 int, int, 
                                 int, int, int, int,  
                                 int, int, int, int, int,  
                                 int *);
  void compute_FCImake_rdm1a (double *, double *, double *, int, int, int, int, int *);
  void compute_FCImake_rdm1b (double *, double *, double *, int, int, int, int, int *);
  void compute_FCIrdm2_a_t1ci_v2 (double *, double *, int, int, int, int, int, int*); 
  void compute_FCIrdm2_b_t1ci_v2 (double *, double *, int, int, int, int, int, int*); 
  void compute_FCIrdm3h_a_t1ci_v2 (double *, double *, int, int, int, int,
                                int, int, int, int, int*);
  void compute_FCIrdm3h_b_t1ci_v2 (double *, double *, int, int, int, int, int,
                                int, int, int, int, int*);
  void compute_FCIrdm3h_a_t1ci_v3 (double *, double *, int, int, int, int, int, int,
                                int, int, int, int, int*);
  void compute_FCIrdm3h_b_t1ci_v3 (double *, double *, int, int, int, int, int, int,
                                int, int, int, int, int*);
  void reorder(double *, double *, double *, int);
  void reduce_buf3_to_rdm(const double *, double *, int, int);
  void filter_sfudm(const double *, double *, int);
  void filter_tdmpp(const double *, double *, int, int);
  void filter_tdm1h(const double *, double *, int);
  void filter_tdm3h(double *, double *, int);
  void transpose_021(double *, double *, int);
  // multi-gpu communication (better here or part of PM?)

  void mgpu_bcast(std::vector<double *>, double *, size_t);
  void mgpu_reduce(std::vector<double *>, double *, int, bool, std::vector<double *>, std::vector<int>);

private:

  class PM * pm;

  class MATHLIB * ml;
  
  double host_compute(double *);
  void get_cores(char *);

  int verbose_level;
  
  size_t grid_size, block_size;
  
  // get_jk

  int update_dfobj;

  //  int nset;

  int size_fdrv;
  int size_buf_vj;
  int size_buf_vk;
  
  // get_jk
  
  double * rho;
  //double * vj;
  double * _vktmp;
 
  double * buf_fdrv;

  double * buf_vj;
  double * buf_vk;
  // ao2mo
  int size_buf_k_pc;
  int size_buf_j_pc;
  int size_fxpp; // remove when ao2mo_v3 is running
  int size_bufpa;
  int size_bufaa;
  int size_k_pc;
  int size_j_pc;
  int size_buf_ppaa;
  int size_buf_papa;

  double * buf_j_pc; 
  double * buf_k_pc; 
  double * pin_fxpp;//remove when ao2mo_v3 is running
  double * pin_bufpa;
  double * buf_ppaa;
  double * buf_papa;

  // h2eff_df
  int size_eri_h2eff;
  int size_buf_eri_h2eff;
  double * buf_eri_h2eff;

  // eri_impham
  int size_eri_impham;
  double * pin_eri_impham;
  
  //tdms
  int size_bravecs;
  int size_ketvecs;
  int size_dm1_full;
  int size_dm2_full;
  double * h_bravecs;
  double * h_ketvecs;
  double * h_dm1_full;
  double * h_dm2_full;
  double * h_dm2_p_full;

  
  // eri caching on device
  bool use_eri_cache;
  
  std::vector<size_t> eri_list; // addr of dfobj+eri1 for key-value pair
  
  std::vector<int> eri_count; // # times particular cache used
  std::vector<int> eri_update; // # times particular cache updated
  std::vector<int> eri_size; // # size of particular cache

  std::vector<int> eri_num_blocks; // # of eri blocks for each dfobj (i.e. trip-count from `for eri1 in dfobj.loop(blksize)`)
  std::vector<int> eri_extra; // per-block data: {naux, nao_pair}
  std::vector<int> eri_device; // device id holding cache

  std::vector<double *> d_eri_cache; // pointers for device caches
  std::vector<double *> d_eri_host; // values on host for checking if update
  
  struct my_AO2MOEnvs {
    int natm;
    int nbas;
    int *atm;
    int *bas;
    double *env;
    int nao;
    int klsh_start;
    int klsh_count;
    int bra_start;
    int bra_count;
    int ket_start;
    int ket_count;
    int ncomp;
    int *ao_loc;
    double *mo_coeff;
    //        CINTOpt *cintopt;
    //        CVHFOpt *vhfopt;
  };
    struct my_device_data {
    int device_id;
    int active; // was device used in calculation and has result to be accumulated?

    int size_rho;
    int size_vj;
    int size_vk;
    //    int size_buf;
    int size_buf1;
    int size_buf2;
    int size_buf3;
    int size_dms;
    int size_dmtril;
    int size_eri1;
    int size_ucas;
    int size_umat;
    int size_h2eff;
    int size_mo_coeff; 
    int size_mo_cas; 
    //ao2mo
    int size_j_pc;
    int size_k_cp;
    int size_k_pc;
    int size_bufd;
    int size_bufpa;
    int size_bufaa;
    // eri_h2eff
    int size_eri_unpacked;
    int size_eri_h2eff;

    //pdft
    int size_mo_grid;
    int size_ao_grid;
    int size_cascm2;
    int size_Pi;
    int size_buf_pdft;

    //fci
    int size_clinka;
    int size_clinkb;
    int size_cibra;
    int size_ciket;
    int size_tdm1;
    int size_tdm2;
    int size_tdm2_p;
    int size_pdm1;//do we need this anymore?
    int size_pdm2;//do we need this anymore?


    double * d_rho;
    double * d_vj;
    double * d_buf1;
    double * d_buf2;
    double * d_buf3;
    double * d_vkk;
    double * d_dms;
    double * d_dmtril;
    double * d_eri1;
    double * d_ucas;
    double * d_umat;
    double * d_h2eff;
    double * d_mo_coeff;
    double * d_mo_cas;
    //ao2mo
    double * d_j_pc;
    double * d_k_pc;
    double * d_bufd;
    double * d_bufpa;
    double * d_bufaa;
    double * d_ppaa;
    double * d_papa;
    // eri_h2eff
    double * d_eri_h2eff;
    //pdft
    double * d_ao_grid;
    double * d_cascm2;
    double * d_mo_grid;
    double * d_Pi;
    double * d_buf_pdft1;
    double * d_buf_pdft2;
    //fci 
    
    int * d_clinka;
    int * d_clinkb;
    double * d_cibra;
    double * d_ciket;
    double * d_tdm2;
    double * d_tdm2_p;
    double * d_tdm1;
    double * d_tdm1h;
    double * d_tdm3ha;
    double * d_tdm3hb;
    double * d_pdm2; //do we need these anymore
    double * d_pdm1; //do we need these anymore

    std::vector<int> type_pumap;
    std::vector<int> size_pumap;
    std::vector<int *> pumap;
    std::vector<int *> d_pumap;
    int * d_pumap_ptr; // no explicit allocation
    int * pumap_ptr; // no explicit allocation

    // we keep the following for now, but we don't explicitly use them anymore
    // besides, pm.h should defined a queue_t and mathlib.h a handle_t...
    
#if defined (_USE_GPU)
#if defined _GPU_CUBLAS
    cublasHandle_t handle;
    cudaStream_t stream;
#elif defined _GPU_HIPBLAS
    hipblasHandle_t handle;
    hipStream_t stream;
#elif defined _GPU_MKL
    int * handle;
    sycl::queue * stream;
#endif
#else
    int * handle;
    int * stream;
#endif
  };

  my_device_data * device_data;
  
  int * dd_fetch_pumap(my_device_data *, int, int);
  double * dd_fetch_eri(my_device_data *, double *, int, int, size_t, int);
  double * dd_fetch_eri_debug(my_device_data *, double *, int, int, size_t, int); // we'll trash this after some time

  template<class T>
  void grow_array(T * &ptr, int current_size, int & max_size, std::string name, const char * file, int line)
  {
    if(current_size > max_size) {
      max_size = current_size;
      if(ptr) pm->dev_free_async(ptr, name);
      ptr = (T *) pm->dev_malloc_async(current_size * sizeof(T), name, file, line);
    }
  }
  
  template<class T>
  void grow_array_host(T * &ptr, int current_size, int & max_size, std::string name)
  {
    if(current_size > max_size) {
      max_size = current_size;
      if(ptr) pm->dev_free_host(ptr);
      ptr = (T *) pm->dev_malloc_host(current_size * sizeof(T));
    }
  }
  
  void fdrv(double *, double *, double *,
	    int, int, int *, int *, int, double *);
  
  void ftrans(int,
	      double *, double *, double *,
	      struct my_AO2MOEnvs *);

  int fmmm(double *, double *, double *,
	   struct my_AO2MOEnvs *, int);
  
  void NPdsymm_triu(int, double *, int);
  void NPdunpack_tril(int, double *, double *, int);
/*--------------------------------------------*/
  double * t_array;
  int * count_array;

  int num_threads;
  int num_devices;
};

#endif
