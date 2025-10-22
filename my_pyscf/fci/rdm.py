import numpy as np
import math
import itertools
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring, rdm
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci import dummy
from pyscf.lib import param
from pyscf.fci import cistring

def _unpack(norb, nelec, link_index, spin=None):
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb
    else:
        return link_index

def _trans_rdm1hs (cre, cibra, ciket, norb, nelec, spin=0, link_index=None):
    '''Evaluate the one-half-particle transition density matrix between ci vectors in different
    Hilbert spaces: <cibra|r'|ciket>, where |cibra> has the same number of orbitals but one
    additional electron of the same spin as r compared to |ciket>.

    Args:
        cre: logical
            True: creation sector, <cibra|r'|ciket>
            False: destruction sector, <cibra|r|ciket>
        cibra: ndarray
            CI vector in (norb,nelec+1) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec+1 electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1h: ndarray of shape (norb,)
            One-half-particle transition density matrix between cibra and ciket.
    '''
    try: 
      use_gpu = param.use_gpu
      gpu = use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    try: custom_fci = param.custom_fci
    except: custom_fci = False
    try: custom_debug = param.custom_debug
    except: custom_debug = False
    if custom_fci and custom_debug and use_gpu:
      ### New kernel
      tdm1h = _trans_rdm1hs_o0(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
      tdm1h_c = _trans_rdm1hs_o1(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
      ### Old kernel
      tdm1_correct = np.allclose(tdm1h, tdm1h_c)
      if tdm1_correct: 
        print('Trans RDM1hs calculated correctly')
      else:
        print('Trans RDM1hs calculated incorrectly')
        exit()
    elif custom_fci and use_gpu: 
      tdm1h = _trans_rdm1hs_o1(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
    else:
      tdm1h = _trans_rdm1hs_o0(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
    tdm1h = tdm1h[-1,:-1]
    if not cre: tdm1h = tdm1h.conj ()
    return tdm1h



def _trans_rdm1hs_o0(cre, cibra, ciket, norb, nelec, spin=0, link_index=None):
    nelec = list (_unpack_nelec (nelec))
    if not cre:
        cibra, ciket = ciket, cibra
        nelec[spin] -= 1
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = [x for x in nelec]
    nelec_bra[spin] += 1
    linkstr = _unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=(1-spin), occ_b=spin)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=0)
    fn = ('FCItrans_rdm1a', 'FCItrans_rdm1b')[spin]
    tdm1h = rdm.make_rdm1_spin1 (fn, cibra, ciket, norb+1, nelec_bra, linkstr)
    return tdm1h

def _trans_rdm1hs_o1(cre, cibra, ciket, norb, nelec, spin=0, link_index=None):
    from mrh.my_pyscf.gpu import libgpu
    gpu=param.use_gpu
    nelec = list (_unpack_nelec (nelec))
    if not cre:
        cibra, ciket = ciket, cibra
        nelec[spin] -= 1
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = [x for x in nelec]
    nelec_bra[spin] += 1
    linkstr = _unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg

    ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket = dummy.dummy_orbital_params(norb, nelec_ket, occ_a =(1-spin), occ_b = spin)
    ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra = dummy.dummy_orbital_params(norb, nelec_bra, occ_a = 0, occ_b = 0)
    na, nlinka = linkstr[0].shape[:2] 
    nb, nlinkb = linkstr[1].shape[:2] 
    na_bra, nb_bra = cibra.shape
    na_ket, nb_ket = ciket.shape
    libgpu.push_cibra(gpu, cibra, na_bra, nb_bra, 0)
    libgpu.push_ciket(gpu, ciket, na_ket, nb_ket, 0)
    tdm1h = np.empty((norb+1, norb+1))
    libgpu.init_tdm1(gpu, norb+1)
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])

    libgpu.compute_tdm1h_spin(gpu, na, nb, nlinka, nlinkb, norb+1, spin, 
                               ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,
                               ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, 0) #TODO: write a better name
    libgpu.pull_tdm1(gpu, tdm1h, norb+1, 0)
    return tdm1h.T
   

def trans_rdm1ha_cre (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-up creation case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm1hs (True, cibra, ciket, norb, nelec, spin=0, link_index=link_index)

def trans_rdm1hb_cre (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-down creation case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm1hs (True, cibra, ciket, norb, nelec, spin=1, link_index=link_index)

def trans_rdm1ha_des (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-up destruction case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm1hs (False, cibra, ciket, norb, nelec, spin=0, link_index=link_index)

def trans_rdm1hb_des (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-down destruction case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm1hs (False, cibra, ciket, norb, nelec, spin=1, link_index=link_index)

def _trans_rdm13hs (cre, cibra, ciket, norb, nelec, spin=0, link_index=None, reorder=True):
    ''' Evaluate the one-half- and three-half-particle transition density matrices between ci
    vectors in different Hilbert spaces: <cibra|r'p'q|ciket> and <cibra|r'|ciket>, where |cibra>
    has the same number of orbitals but one additional electron of the same spin as r compared to
    |ciket>.

    Args:
        cre: logical
            True: creation sector, <cibra|r'|ciket> and <cibra|r'p'q|ciket>
            False: destruction sector, <cibra|r|ciket> and <cibra|p'qr|ciket>
        cibra: ndarray
            CI vector in (norb,nelec+1) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec+1 electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1h: ndarray of shape (norb,)
            One-half-particle transition density matrix between cibra and ciket.
        (tdm3ha, tdm3hb): ndarrays of shape (norb,norb,norb,)
            Three-half-particle transition density matrix between cibra and ciket, spin-up and
            spin-down cases of the full electron. Returned in Mulliken order with the half-electron
            always first and the full electron always second:
            tdm3ha[r,p,q] = <cibra|r'p'q|ciket> or <cibra|p'qr|ciket>
    '''
    try: 
      use_gpu = param.use_gpu
      gpu = use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    try: custom_fci = param.custom_fci
    except: custom_fci = False
    try: custom_debug = param.custom_debug
    except: custom_debug = False
    if custom_fci and custom_debug and use_gpu:
      ### Old kernel
      tdm1h, tdm3ha, tdm3hb = _trans_rdm13hs_o0(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index, reorder = reorder)
      ### New kernel
      tdm1h_c, tdm3ha_c, tdm3hb_c = _trans_rdm13hs_o5(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index, reorder = reorder)
      tdm1_correct = np.allclose(tdm1h, tdm1h_c)
      tdm3ha_correct = np.allclose(tdm3ha, tdm3ha_c)
      tdm3hb_correct = np.allclose(tdm3hb, tdm3hb_c)
      if tdm1_correct and tdm3ha_correct and tdm3hb_correct: 
        print('Trans RDM13hs calculated correctly')
      else:
        print('TDM RDM13hs calculated incorrectly')
        print('TDM1h correct?', tdm1_correct)
        print('TDM3ha correct?', tdm3ha_correct)
        print('TDM3hb correct?', tdm3hb_correct)
        exit()
    elif custom_fci and use_gpu: 
      tdm1h, tdm3ha, tdm3hb = _trans_rdm13hs_o5(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index, reorder = reorder)
    else:
      tdm1h, tdm3ha, tdm3hb = _trans_rdm13hs_o0(cre, cibra, ciket, norb, nelec, spin=spin, link_index=link_index, reorder = reorder)
    return tdm1h, (tdm3ha, tdm3hb)

def _trans_rdm13hs_o0(cre, cibra, ciket, norb, nelec, spin=0, link_index=None, reorder=True):
    nelec = list (_unpack_nelec (nelec))
    if not cre:
        cibra, ciket = ciket, cibra
        nelec[spin] -= 1
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = [x for x in nelec]
    nelec_bra[spin] += 1
    linkstr = _unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=(1-spin), occ_b=spin)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=0)
    fn_par = ('FCItdm12kern_a', 'FCItdm12kern_b')[spin]
    fn_ab = 'FCItdm12kern_ab'

    tdm1h, tdm3h_par = rdm.make_rdm12_spin1 (fn_par, cibra, ciket, norb+1, nelec_bra, linkstr, 2)
    if reorder: tdm1h, tdm3h_par = rdm.reorder_rdm (tdm1h, tdm3h_par, inplace=True)
    if spin:
      tdm3ha = rdm.make_rdm12_spin1 (fn_ab, ciket, cibra, norb+1, nelec_bra, linkstr, 0)[1]
      tdm3ha = tdm3ha.transpose (3,2,1,0)
      tdm3hb = tdm3h_par
    else:
      tdm3ha = tdm3h_par
      tdm3hb = rdm.make_rdm12_spin1 (fn_ab, cibra, ciket, norb+1, nelec_bra, linkstr, 0)[1]

    tdm1h = tdm1h[-1,:-1]
    tdm3ha = tdm3ha[:-1,-1,:-1,:-1]
    tdm3hb = tdm3hb[:-1,-1,:-1,:-1]
    if not cre: 
        tdm1h = tdm1h.conj ()
        tdm3ha = tdm3ha.conj ().transpose (0,2,1)
        tdm3hb = tdm3hb.conj ().transpose (0,2,1)
    return tdm1h, tdm3ha, tdm3hb 

def _trans_rdm13hs_o5(cre, cibra, ciket, norb, nelec, spin=0, link_index=None, reorder=True):
    '''GPU accelerated _trand_rdm13hs with custom FCI kernel'''
    from mrh.my_pyscf.gpu import libgpu
    gpu=param.use_gpu
    nelec = list (_unpack_nelec (nelec))
    if not cre:
        cibra, ciket = ciket, cibra
        nelec[spin] -= 1
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = [x for x in nelec]
    nelec_bra[spin] += 1
    linkstr = _unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket = dummy.dummy_orbital_params(norb, nelec_ket, occ_a = (1-spin), occ_b = spin)
    ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra = dummy.dummy_orbital_params(norb, nelec_bra, occ_a = 0, occ_b = 0)
    na, nlinka = linkstr[0].shape[:2] 
    nb, nlinkb = linkstr[1].shape[:2] 
    na_bra, nb_bra = cibra.shape
    na_ket, nb_ket = ciket.shape
    libgpu.push_cibra(gpu, cibra, na_bra, nb_bra, 0)
    libgpu.push_ciket(gpu, ciket, na_ket, nb_ket, 0)
    #print("cibra: size", na_bra, "x",nb_bra, "=", na_bra*nb_bra)
    #print("ciket: size", na_ket, "x",nb_ket, "=", na_ket*nb_ket)
    

    tdm1h = np.empty((norb))
    tdm3ha = np.empty((norb, norb, norb))
    tdm3hb = np.empty((norb, norb, norb))
    libgpu.init_tdm1(gpu, norb+1)
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
    libgpu.init_tdm3hab(gpu, norb+1)
    libgpu.compute_tdm13h_spin_v4(gpu, na, nb, nlinka, nlinkb, norb+1, spin, reorder,
                                   ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,
                                   ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, 0) #TODO: write a better name
    if reorder: libgpu.reorder_rdm(gpu, norb+1, 0)
    libgpu.pull_tdm3hab_v2(gpu, tdm1h, tdm3ha, tdm3hb, norb, cre, spin, 0)
    return tdm1h, tdm3ha, tdm3hb 

def _trans_rdm13hs_o6(cre, cibra, ciket, norb, nelec, spin=0, link_index=None, reorder=True):
    '''GPU accelerated _trand_rdm13hs with custom FCI kernel'''
    from mrh.my_pyscf.gpu import libgpu
    gpu=param.use_gpu
    nelec = list (_unpack_nelec (nelec))
    if not cre:
        cibra, ciket = ciket, cibra
        nelec[spin] -= 1
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = [x for x in nelec]
    nelec_bra[spin] += 1
    linkstr = _unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket = dummy.dummy_orbital_params(norb, nelec_ket, occ_a = (1-spin), occ_b = spin)
    ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra = dummy.dummy_orbital_params(norb, nelec_bra, occ_a = 0, occ_b = 0)
    na, nlinka = linkstr[0].shape[:2] 
    nb, nlinkb = linkstr[1].shape[:2] 
    na_bra, nb_bra = cibra.shape
    na_ket, nb_ket = ciket.shape
    libgpu.push_cibra(gpu, cibra, na_bra, nb_bra, 0)
    libgpu.push_ciket(gpu, ciket, na_ket, nb_ket, 0)

    tdm1h = np.empty((norb))
    tdm3ha = np.empty((norb, norb, norb))
    tdm3hb = np.empty((norb, norb, norb))
    libgpu.init_tdm1(gpu, norb+1)
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
    libgpu.init_tdm3hab(gpu, norb+1)
    libgpu.compute_tdm13h_spin_v5(gpu, na, nb, nlinka, nlinkb, norb+1, spin, reorder,
                                   ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,
                                   ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, 0) #TODO: write a better name
    if reorder: libgpu.reorder_rdm(gpu, norb+1, 0)
    libgpu.pull_tdm3hab_v2(gpu, tdm1h, tdm3ha, tdm3hb, norb, cre, spin, 0)
    return tdm1h, tdm3ha, tdm3hb 




def trans_rdm13ha_cre (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-up creation case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm13hs (True, cibra, ciket, norb, nelec, spin=0, link_index=link_index)

def trans_rdm13hb_cre (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-down creation case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm13hs (True, cibra, ciket, norb, nelec, spin=1, link_index=link_index)

def trans_rdm13ha_des (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-up destruction case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm13hs (False, cibra, ciket, norb, nelec, spin=0, link_index=link_index)

def trans_rdm13hb_des (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-down destruction case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm13hs (False, cibra, ciket, norb, nelec, spin=1, link_index=link_index)

def trans_sfudm1 (cibra, ciket, norb, nelec, link_index=None):
    ''' Evaluate the spin-flip-up single-electron transition density matrix: <cibra|a'b|ciket>.

    Args:
        cibra: ndarray
            CI vector in (norb,(neleca+1,nelecb-1)) Hilbert space
        ciket: ndarray
            CI vector in (norb,(neleca,nelecb)) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the (neleca+1,nelecb) electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        sfudm1: ndarray of shape (norb,norb)
            Spin-flip up transition density matrix between cibra and ciket
    '''
    try: 
      use_gpu = param.use_gpu
      gpu = use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    try: custom_fci = param.custom_fci
    except: custom_fci = False
    try: custom_debug = param.custom_debug
    except: custom_debug = False
    if custom_fci and custom_debug and use_gpu:
      ### Old kernel
      dum2dm = _trans_sfudm1_o0 (cibra, ciket, norb, nelec, link_index=link_index)
      ### New kernel
      dum2dm_c = _trans_sfudm1_o2 (cibra, ciket, norb, nelec, link_index=link_index)
      tdm2_correct = np.allclose(dum2dm, dum2dm_c)
      if tdm2_correct: 
        print('Trans SFUDM1 calculated correctly')
      else:
        print('Trans SFUDM1 calculated incorrectly')
        exit()
    elif custom_fci and use_gpu: 
      dum2dm = _trans_sfudm1_o2 (cibra, ciket, norb, nelec, link_index=link_index)
    else:
      dum2dm = _trans_sfudm1_o0 (cibra, ciket, norb, nelec, link_index=link_index)
    return dum2dm


def _trans_sfudm1_o0(cibra, ciket, norb, nelec, link_index=None):
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[0] += 1
    nelec_bra[1] -= 1
    nelecd = [nelec_bra[0], nelec_ket[1]]
    linkstr = _unpack (norb+1, nelecd, link_index)
    errmsg = ("For the spin-flip transition density matrix functions, the linkstr must be for "
              "(neleca+1,nelecb) electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelecd[i]*(norb-nelecd[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=1, occ_b=0)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=1)
    fn = 'FCItdm12kern_ab'
    dm2dum = rdm.make_rdm12_spin1 (fn, ciket, cibra, norb+1, nelecd, linkstr, 0)[1] #### NOTE: ciket and cibra is switched!!
    return -dm2dum[-1, :-1, :-1, -1]

def _trans_sfudm1_o2(cibra,ciket,norb, nelec, link_index=None):
    from mrh.my_pyscf.gpu import libgpu
    gpu=param.use_gpu
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[0] += 1
    nelec_bra[1] -= 1
    nelecd = [nelec_bra[0], nelec_ket[1]]
    linkstr = _unpack (norb+1, nelecd, link_index)
    errmsg = ("For the spin-flip transition density matrix functions, the linkstr must be for "
              "(neleca+1,nelecb) electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelecd[i]*(norb-nelecd[i]+2))), errmsg
    ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket = dummy.dummy_orbital_params(norb, nelec_ket, occ_a = 1, occ_b = 0)
    ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra = dummy.dummy_orbital_params(norb, nelec_bra, occ_a = 0, occ_b = 1)
    na, nlinka = linkstr[0].shape[:2] 
    nb, nlinkb = linkstr[1].shape[:2] 
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
    dm2dum = np.zeros((norb,norb))
    na_bra, nb_bra = cibra.shape
    na_ket, nb_ket = ciket.shape
    libgpu.push_cibra(gpu, ciket, na_ket, nb_ket, 0)
    libgpu.push_ciket(gpu, cibra, na_bra, nb_bra, 0)
    libgpu.init_tdm1(gpu, norb)#actually pulling
    libgpu.init_tdm2(gpu, norb+1)#for storing result on gpu
    libgpu.compute_sfudm_v2(gpu, na, nb, nlinka, nlinkb, norb+1, 
                         ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket,
                         ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, 0)
    libgpu.pull_tdm1(gpu, dm2dum, norb, 0)#when filtering
    return dm2dum

def trans_sfddm1 (cibra, ciket, norb, nelec, link_index=None):
    ''' Evaluate the spin-flip-down single-electron transition density matrix: <cibra|b'a|ciket>.

    Args:
        cibra: ndarray
            CI vector in (norb,(neleca-1,nelecb+1)) Hilbert space
        ciket: ndarray
            CI vector in (norb,(neleca,nelecb)) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the (neleca,nelecb+1) electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        sfdm1: ndarray of shape (norb,norb)
            Spin-flip-down transition density matrix between cibra and ciket
    '''
    nelec = list(_unpack_nelec (nelec))
    nelec[0] -= 1
    nelec[1] += 1
    return trans_sfudm1 (ciket, cibra, norb, nelec, link_index=link_index).conj ().T

def trans_ppdm (cibra, ciket, norb, nelec, spin=0, link_index=None):
    ''' Evaluate the pair-creation single-electron transition density matrix: <cibra|p'q'|ciket>.

    Args:
        cibra: ndarray
            CI vector in (norb,nelec+2) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space
        spin: integer
            Spin of created pair. 0 = aa, 1 = ab, 2 = bb

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec+2 electrons in norb+1/norb+2 (for ab/other spin) orbitals
            Hilbert space. See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        ppdm: ndarray of shape (norb,norb)
            Pair-creation single-electron transition density matrix
    '''
    try: 
      use_gpu = param.use_gpu
      gpu = use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    try: custom_fci = param.custom_fci
    except: custom_fci = False
    try: custom_debug = param.custom_debug
    except: custom_debug = False
    ndum = 2 - (spin%2)
    if custom_fci and custom_debug and use_gpu:
      tdmhh_c = _trans_ppdm_o3(cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
      tdmhh = _trans_ppdm_o0(cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
      tdmhh_correct = np.allclose(tdmhh, tdmhh_c)
      if tdmhh_correct: 
        print('trans TDMpp calculated correctly')
      else:
        print('Trans TDMpp incorrect')
        print(tdmhh_c)
        print(tdmhh)
        exit()
    elif custom_fci and use_gpu: 
      tdmhh = _trans_ppdm_o3(cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
    else:
      tdmhh = _trans_ppdm_o0(cibra, ciket, norb, nelec, spin=spin, link_index=link_index)
    return tdmhh

def _trans_ppdm_o0(cibra, ciket, norb, nelec, spin = 0, link_index = None):
    s1 = int (spin>1)
    s2 = int (spin>0)
    ndum = 2 - (spin%2)
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[s1] += 1
    nelec_bra[s2] += 1
    occ_a, occ_b = int (spin<2), int (spin>0)
    linkstr = _unpack (norb+ndum, nelec_bra, link_index)
    errmsg = ("For the pair-creation transition density matrix functions, the linkstr must "
              "be for nelec+2 electrons occupying norb+1/norb+2 (ab/other spin case) orbitals.")
    assert (linkstr[0].shape[1]==(nelec_bra[0]*(norb+ndum-nelec_bra[0]+1))), errmsg
    assert (linkstr[1].shape[1]==(nelec_bra[1]*(norb+ndum-nelec_bra[1]+1))), errmsg
    nelecd = [nelec_ket[0], nelec_ket[1]]
    for i in range (ndum):
        ciket = dummy.add_orbital (ciket, norb+i, nelecd, occ_a=occ_a, occ_b=occ_b)
        nelecd[0] += occ_a
        nelecd[1] += occ_b
        cibra = dummy.add_orbital (cibra, norb+i, nelec_bra, occ_a=0, occ_b=0)
    fn = ('FCItdm12kern_a', 'FCItdm12kern_ab', 'FCItdm12kern_b')[spin]
    fac = (2,0,2)[spin]
    dumdm1, dumdm2 = rdm.make_rdm12_spin1 (fn, cibra, ciket, norb+ndum, nelec_bra, linkstr, fac)
    if (spin%2)==0: dumdm1, dumdm2 = rdm.reorder_rdm (dumdm1, dumdm2, inplace=True)
    return dumdm2[:-ndum,-1,:-ndum,-ndum]

def _trans_ppdm_o3(cibra, ciket, norb, nelec, spin = 0, link_index = None):
    from mrh.my_pyscf.gpu import libgpu
    gpu=param.use_gpu
    s1 = int (spin>1)
    s2 = int (spin>0)
    ndum = 2 - (spin%2)
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[s1] += 1
    nelec_bra[s2] += 1
    occ_a, occ_b = int (spin<2), int (spin>0)
    linkstr = _unpack (norb+ndum, nelec_bra, link_index)
    errmsg = ("For the pair-creation transition density matrix functions, the linkstr must "
              "be for nelec+2 electrons occupying norb+1/norb+2 (ab/other spin case) orbitals.")
    assert (linkstr[0].shape[1]==(nelec_bra[0]*(norb+ndum-nelec_bra[0]+1))), errmsg
    assert (linkstr[1].shape[1]==(nelec_bra[1]*(norb+ndum-nelec_bra[1]+1))), errmsg
    nelecd = [nelec_ket[0], nelec_ket[1]]
    nelecd_copy = nelecd.copy()
    na_bra, nb_bra = cibra.shape
    na_ket, nb_ket = ciket.shape
    ia_bra = ia_ket = ib_bra = ib_ket = 0
    ja_bra, jb_bra, ja_ket, jb_ket = na_bra, nb_bra, na_ket, nb_ket
    sgn_bra = sgn_ket = 1
    for i in range (ndum):
        ia_ket_new, ja_ket_new, ib_ket_new, jb_ket_new, sgn_ket_new = dummy.dummy_orbital_params(norb+i, nelecd_copy, occ_a=occ_a, occ_b=occ_b)
        nelecd_copy[0] +=occ_a
        nelecd_copy[1] +=occ_b 
        ia_bra_new, ja_bra_new, ib_bra_new, jb_bra_new, sgn_bra_new = dummy.dummy_orbital_params(norb+i, nelec_bra, occ_a = 0, occ_b=0)
        ia_bra += ia_bra_new
        ib_bra += ib_bra_new
        ia_ket += ia_ket_new
        ib_ket += ib_ket_new
        ja_bra = ia_bra + na_bra
        jb_bra = ib_bra + nb_bra
        ja_ket = ia_ket + na_ket
        jb_ket = ib_ket + nb_ket
        sgn_bra *= sgn_bra_new
        sgn_ket *= sgn_ket_new
    #if spin!= 1:
    #  dumdm1 = np.empty((norb+ndum, norb+ndum))
    #else: 
    #  dumdm1 = np.zeros((norb+ndum, norb+ndum))
    libgpu.init_tdm1(gpu, norb)
    libgpu.init_tdm2(gpu, norb+ndum)
    na, nlinka = linkstr[0].shape[:2] 
    nb, nlinkb = linkstr[1].shape[:2] 
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
    libgpu.push_cibra(gpu, cibra, na_bra, nb_bra, 0)
    libgpu.push_ciket(gpu, ciket, na_ket, nb_ket, 0)
    #print(cibra.shape, ciket.shape)
    #print("cibra")
    #print(cibra)
    #print("ciket")
    #print(ciket)
    libgpu.compute_tdmpp_spin_v4(gpu, na, nb, nlinka, nlinkb, norb+ndum, spin, 
                              ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, 
                              ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, 0) #TODO: write a better name
    ##VA 10/2: Reorder doesn't do anything ... 
    #dumdm2 = np.empty((norb+ndum, norb+ndum, norb+ndum, norb+ndum))
    #libgpu.pull_tdm2(gpu, dumdm2, norb+ndum)
    #return dumdm2[:-ndum,-1,:-ndum,-ndum]
    dumdm2 = np.empty((norb, norb))
    libgpu.pull_tdm1(gpu, dumdm2, norb, 0)
    return dumdm2

def trans_hhdm (cibra, ciket, norb, nelec, spin=0, link_index=None):
    ''' Evaluate the pair-destruction single-electron transition density matrix: <cibra|pq|ciket>.

    Args:
        cibra: ndarray
            CI vector in (norb,nelec-2) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space
        spin: integer
            Spin of created pair. 0 = aa, 1 = ab, 2 = bb

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec electrons in norb+1/norb+2 (for ab/other spin) orbitals
            Hilbert space. See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        hhdm: ndarray of shape (norb,norb)
            Pair-destruction single-electron transition density matrix
    '''
    nelec = list(_unpack_nelec (nelec))
    nelec[int (spin>1)] -= 1
    nelec[int (spin>0)] -= 1
    return trans_ppdm (ciket, cibra, norb, nelec, spin=spin, link_index=link_index).conj ().T



