#!/usr/bin/env python
import ctypes
import numpy
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec
import traceback, sys
librdm = cistring.libfci

def _make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    assert (cibra is not None and ciket is not None)
    from pyscf.lib import param
    try:
      use_gpu = param.use_gpu
      gpu = param.use_gpu
    except: 
      use_gpu = None
    if (fname in ['FCItrans_rdm1a', 'FCItrans_rdm1b', 'FCImake_rdm1a', 'FCImake_rdm1b']) and (use_gpu is not None):
        use_gpu = param.use_gpu
        gpu = param.use_gpu
    else:
        use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    cibra = numpy.ascontiguousarray(cibra)
    ciket = numpy.ascontiguousarray(ciket)
    assert (cibra.size == na*nb), '{} {} {}'.format (cibra.size, na, nb)
    assert (ciket.size == na*nb), '{} {} {}'.format (ciket.size, na, nb)
    if use_gpu and gpu_debug:
      rdm_cpu = _make_rdm1_o0(fname, cibra, ciket, norb, nelec, link_index=link_index)
      rdm_gpu = _make_rdm1_o1(fname, cibra, ciket, norb, nelec, link_index=link_index)
      if (numpy.allclose(rdm_cpu, rdm_gpu)):
        print("RDM1_spin1", fname, "TDM1s calculate correctly", flush=True)
      else: 
        print("Problem in TDM1")
      return rdm_cpu.T
    elif use_gpu:  
      rdm_gpu = _make_rdm1_o1(fname, cibra, ciket, norb, nelec, link_index=link_index)
      return rdm_gpu.T
    else:
      rdm_cpu = _make_rdm1_o0(fname, cibra, ciket, norb, nelec, link_index=link_index)
      return rdm_cpu.T

def _make_rdm1_spin_o0(fname, cibra, ciket, norb, nelec, link_index):
    rdm1 = numpy.empty((norb,norb))
    fn = getattr(librdm, fname)
    fn(rdm1.ctypes.data_as(ctypes.c_void_p),
     cibra.ctypes.data_as(ctypes.c_void_p),
     ciket.ctypes.data_as(ctypes.c_void_p),
     ctypes.c_int(norb),
     ctypes.c_int(na), ctypes.c_int(nb),
     ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
     link_indexa.ctypes.data_as(ctypes.c_void_p),
     link_indexb.ctypes.data_as(ctypes.c_void_p))
    return rdm1

def _make_rdm1_spin_o1(fname, cibra, ciket, norb, nelec, link_index):
    from mrh.my_pyscf.gpu import libgpu
    rdm1 = numpy.empty((norb,norb))
    libgpu.init_tdm1(gpu, norb)
    libgpu.push_cibra(gpu, cibra, na, nb, 0)
    libgpu.push_ciket(gpu, ciket, na, nb, 0)
    if fname == 'FCItrans_rdm1a': 
      libgpu.push_link_indexa(gpu, na, nlinka, link_indexa) #TODO: move up to direct spin1 or just generate on the fly
      libgpu.compute_trans_rdm1a(gpu, na, nb, nlinka, nlinkb, norb, 0) #TODO: update name
    elif fname == 'FCItrans_rdm1b':
      libgpu.push_link_indexb(gpu, nb, nlinkb, link_indexb) #TODO: move up to direct_spin1 or just generate on the fly
      libgpu.compute_trans_rdm1b(gpu, na, nb, nlinka, nlinkb, norb, 0) #TODO: update name
    elif fname == 'FCImake_rdm1a':
      libgpu.push_link_indexa(gpu, na, nlinka, link_indexa) #TODO: move up to direct spin1 or just generate on the fly
      libgpu.compute_make_rdm1a(gpu, na, nb, nlinka, nlinkb, norb, 0) #TODO: update name
    elif fname == 'FCImake_rdm1b':
      libgpu.push_link_indexb(gpu, nb, nlinkb, link_indexb) #TODO: move up to direct_spin1 or just generate on the fly
      libgpu.compute_make_rdm1b(gpu, na, nb, nlinka, nlinkb, norb, 0) #TODO: update name
    libgpu.pull_tdm1(gpu, rdm1, norb, 0)
    libgpu.barrier(gpu)
    return rdm1



def _reorder_rdm(rdm1, rdm2, inplace=False):
    nmo = rdm1.shape[0]
    if not inplace:
        rdm2 = rdm2.copy()
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1.T
    # Employing the particle permutation symmetry, average over two particles
    # to reduce numerical round off error
    rdm2 = lib.transpose_sum(rdm2.reshape(nmo*nmo,-1), inplace=True) * .5
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

def _make_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index=None, symm=0):
    #FCItdm12_kern_a, FCItdm12kern_b, FCItdm12kern_ab 
    #add traceback
    #    traceback.print_stack(file=sys.stdout)
    from pyscf.lib import param
    if (fname in ['FCItdm12kern_a', 'FCItdm12kern_b', 'FCItdm12kern_ab', 'FCIrdm12kern_sf']) and getattr (param, 'use_gpu', None) is not None:
       use_gpu = param.use_gpu
       gpu=param.use_gpu
    else: 
       use_gpu = None
    assert (cibra is not None and ciket is not None)
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    if link_index is None:
      neleca, nelecb = _unpack_nelec(nelec)
      link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
      if neleca != nelecb:
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
      link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    #print('link_indexa', link_indexa.shape)
    #print('link_indexb', link_indexb.shape)
    assert (cibra.size == na*nb)
    assert (ciket.size == na*nb)
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    if use_gpu and gpu_debug: 
      from mrh.my_pyscf.gpu import libgpu
      rdm1_cpu = numpy.empty((norb,norb))
      rdm2_cpu = numpy.empty((norb,norb,norb,norb))
      rdm1_gpu = numpy.empty((norb,norb))
      rdm2_gpu = numpy.empty((norb,norb,norb,norb))
      librdm.FCIrdm12_drv(getattr(librdm, fname),
                        rdm1_cpu.ctypes.data_as(ctypes.c_void_p),
                        rdm2_cpu.ctypes.data_as(ctypes.c_void_p),
                        cibra.ctypes.data_as(ctypes.c_void_p),
                        ciket.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norb),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(symm))
      libgpu.init_tdm1(gpu, norb)
      libgpu.init_tdm2(gpu, norb)
      libgpu.push_cibra(gpu, cibra, na, nb, 0)
      libgpu.push_ciket(gpu, ciket, na, nb, 0)
      libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb) #TODO: move this to direct_spin1 or generate on the fly
      rdm1_correct=True
      if fname == 'FCItdm12kern_a': 
        libgpu.compute_tdm12kern_a_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
        libgpu.pull_tdm1(gpu, rdm1_gpu, norb, 0)
        libgpu.barrier(gpu)
        rdm1_correct = numpy.allclose(rdm1_cpu, rdm1_gpu)
      if fname == 'FCItdm12kern_b': 
        libgpu.compute_tdm12kern_b_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
        libgpu.pull_tdm1(gpu, rdm1_gpu, norb, 0)
        libgpu.barrier(gpu)
        rdm1_correct = numpy.allclose(rdm1_cpu, rdm1_gpu)
      if fname == 'FCItdm12kern_ab': 
        libgpu.compute_tdm12kern_ab_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
      if fname == 'FCIrdm12kern_sf': 
        libgpu.compute_rdm12kern_sf_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
        libgpu.pull_tdm1(gpu, rdm1_gpu, norb, 0)
        libgpu.barrier(gpu)
        rdm1_correct = numpy.allclose(rdm1_cpu, rdm1_gpu)
        
      libgpu.pull_tdm2(gpu, rdm2_gpu, norb, 0)
      libgpu.barrier(gpu)
      rdm2_correct = numpy.allclose(rdm2_cpu, rdm2_gpu)
      if rdm1_correct and rdm2_correct:
        print('RDM12_spin1', fname, "TDM12 calculated correctly at GPU", gpu)
      else: 
        print('RDM12_spin1', fname, use_gpu, "Problem in TDM12", flush=True)
        if rdm1_correct: print("TDM1 correct")
        else: 
          print("Incorrect TDM1")
        if rdm2_correct: print("TDM2 correct")
        else: 
          print("Incorrect TDM2")
        exit()
      return rdm1_gpu.T, rdm2_gpu
    elif use_gpu: 
      from mrh.my_pyscf.gpu import libgpu
      rdm1_gpu = numpy.zeros((norb,norb))
      rdm2_gpu = numpy.zeros((norb,norb,norb,norb))
      libgpu.init_tdm1(gpu, norb)
      libgpu.init_tdm2(gpu, norb)
      libgpu.push_cibra(gpu, cibra, na, nb, 0)
      libgpu.push_ciket(gpu, ciket, na, nb, 0)
      libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb) #TODO: move this to direct_spin1 because it's used with both a and b
      if fname == 'FCItdm12kern_a': 
        libgpu.compute_tdm12kern_a_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
        libgpu.pull_tdm1(gpu, rdm1_gpu, norb, 0)
      if fname == 'FCItdm12kern_b': 
        libgpu.compute_tdm12kern_b_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
        libgpu.pull_tdm1(gpu, rdm1_gpu, norb, 0)
      if fname == 'FCItdm12kern_ab': 
        libgpu.compute_tdm12kern_ab_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
      if fname == 'FCIrdm12kern_sf': 
        libgpu.compute_rdm12kern_sf_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
        libgpu.pull_tdm1(gpu, rdm1_gpu, norb, 0)
      libgpu.pull_tdm2(gpu, rdm2_gpu, norb, 0)
      libgpu.barrier(gpu)
      return rdm1_gpu.T, rdm2_gpu
    else: 
      rdm1 = numpy.empty((norb,norb))
      rdm2 = numpy.empty((norb,norb,norb,norb))
      librdm.FCIrdm12_drv(getattr(librdm, fname),
                        rdm1.ctypes.data_as(ctypes.c_void_p),
                        rdm2.ctypes.data_as(ctypes.c_void_p),
                        cibra.ctypes.data_as(ctypes.c_void_p),
                        ciket.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norb),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(symm))
      return rdm1.T, rdm2


def _make_dm123(fname, cibra, ciket, norb, nelec):
    assert (cibra is not None and ciket is not None)
    from pyscf.lib import param
    try:
      use_gpu = param.use_gpu
      gpu = param.use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False

    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    if use_gpu and gpu_debug: 
      rdm1_cpu, rdm2_cpu, rdm3_cpu = _make_dm123_o0(fname, cibra, ciket, norb, nelec)
      rdm1_gpu, rdm2_gpu, rdm3_gpu = _make_dm123_o1(fname, cibra, ciket, norb, nelec)
      rdm1_correct = np.allclose(rdm1_cpu, rdm1_gpu)
      rdm2_correct = np.allclose(rdm2_cpu, rdm2_gpu)
      rdm3_correct = np.allclose(rdm3_cpu, rdm3_gpu)
      if rdm1_correct and rdm2_correct and rdm3_correct:
        print("all_rdm_correct")
      else:
        print("Issue in 3pdm")
        print("rdm1 correct?", rdm1_correct)
        print("rdm2 correct?", rdm2_correct)
        print("rdm3 correct?", rdm3_correct)
        exit()
      return rdm1_cpu.T, rdm2_cpu, rdm3_cpu

def _make_dm123_o0(fname, cibra, ciket, norb, nelec):
      rdm1 = numpy.empty((norb,)*2)
      rdm2 = numpy.empty((norb,)*4)
      rdm3 = numpy.empty((norb,)*6)
      librdm.FCIrdm3_drv(getattr(librdm, fname),
                   rdm1.ctypes.data_as(ctypes.c_void_p),
                   rdm2.ctypes.data_as(ctypes.c_void_p),
                   rdm3.ctypes.data_as(ctypes.c_void_p),
                   cibra.ctypes.data_as(ctypes.c_void_p),
                   ciket.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(norb),
                   ctypes.c_int(na), ctypes.c_int(nb),
                   ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                   link_indexa.ctypes.data_as(ctypes.c_void_p),
                   link_indexb.ctypes.data_as(ctypes.c_void_p))
      rdm3 = _complete_dm3_(rdm2, rdm3)
      return rdm1.T, rdm2, rdm3

def _make_dm123_o1(fname, cibra, ciket, norb, nelec): 
    from mrh.my_pyscf.gpu import libgpu
    rdm1 = numpy.empty((norb,)*2)
    rdm2 = numpy.empty((norb,)*4)
    rdm3 = numpy.empty((norb,)*6)
    libgpu.init_tdm1(gpu, norb)
    libgpu.init_tdm2(gpu, norb)
    libgpu.init_tdm3(gpu, norb)
    libgpu.push_cibra(gpu, cibra, na, nb, 0)
    libgpu.push_ciket(gpu, ciket, na, nb, 0)
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb)
    libgpu.compute_3pdm_kern_sf(gpu, na, nb, nlinka, nlinkb, norb, 0)
    libgpu.pull_tdm1(gpu, rdm1, norb, 0)
    libgpu.pull_tdm2(gpu, rdm2, norb, 0)
    libgpu.pull_tdm3(gpu, rdm3, norb, 0)
    rdm3 = _complete_dm3_(rdm2, rdm3)
    return rdm1.T, rdm2, rdm3

def _complete_dm3_(dm2, dm3):
    # fci_4pdm.c assumed symmetry p >= r >= t for 3-pdm <p^+ q r^+ s t^+ u>
    # Using E^r_sE^p_q = E^p_qE^r_s - \delta_{qr}E^p_s + \delta_{ps}E^r_q to
    # complete the full 3-pdm
    def transpose01(ijk, i, j, k):
        jik = ijk.transpose(1,0,2)
        jik[:,j] -= dm2[i,:,k,:]
        jik[i,:] += dm2[j,:,k,:]
        dm3[j,:,i,:,k,:] = jik
        return jik
    def transpose12(ijk, i, j, k):
        ikj = ijk.transpose(0,2,1)
        ikj[:,:,k] -= dm2[i,:,j,:]
        ikj[:,j,:] += dm2[i,:,k,:]
        dm3[i,:,k,:,j,:] = ikj
        return ikj

    # ijk -> jik -> jki -> kji -> kij -> ikj
    norb = dm2.shape[0]
    for i in range(norb):
        for j in range(i+1):
            for k in range(j+1):
                tmp = transpose01(dm3[i,:,j,:,k,:].copy(), i, j, k)
                tmp = transpose12(tmp, j, i, k)
                tmp = transpose01(tmp, j, k, i)
                tmp = transpose12(tmp, k, j, i)
                tmp = transpose01(tmp, k, i, j)
    return dm3
