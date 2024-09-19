import sys
import numpy as np
import itertools
from scipy import linalg
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.lo.orth import vec_lowdin
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.spin_op import contract_sdown, contract_sup, mdown, mup
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver
from mrh.my_pyscf.lassi.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.lassi.spaces import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.spaces import _spin_shuffle, list_spaces
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.lassi.spaces import orthogonal_excitations, combine_orthogonal_excitations
from mrh.my_pyscf.lassi.lassi import LASSI

# TODO: split prepare_states into three steps
# 1. Compute the number of unique fragment CI vectors to be computed (including sz-flips but not
#    including repeated references in the final CI table), the total number of model states in the
#    final step, and report the memory footprint. Prepare the corresponding tables of CI vectors.
# 2. Optimize the unique, unfrozen CI vectors. Include the option to initialize them from stored
#    values.
# 3. Combine the optimized CI vectors into a single ci table in the format the LASSI kernel expects.
#    Use references, not copies.

def prepare_model_states (lsi, ci_ref, ci_sf, ci_ch):
    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.new_logger (lsi, lsi.verbose)
    las = lsi._las.get_single_state_las (state=0)
    # Write in ci_ref
    space0 = list_spaces (las)[0]
    for i in range (las.nfrags):
        nelec, smult = (space0.neleca[i], space0.nelecb[i]), space0.smults[i]
        las.ci[i][0] = space0.ci[i] = mdown (ci_ref[i], las.ncas_sub[0], nelec, smult)
    # Make spin flip objects
    spin_flips = []
    for i in range (las.nfrags):
        smult = space0.smults[i]
        ci1 = []
        spins1 = []
        smults1 = []
        if ci_sf[i][0] is not None:
            smults1.append (smult-2)
            spins1.append (smult-3)
            ci1.append (ci_sf[i][0])
        if ci_sf[i][1] is not None:
            smults1.append (smult+2)
            spins1.append (smult+1)
            ci1.append (ci_sf[i][1])
        spin_flips.append (SpinFlips (ci1, space0.nlas[i], space0.nelec[i], spins1, smults1))
    # Make charge-hop objects
    spaces2 = [space0]
    for i, a in itertools.product (range (lsi.nfrags), repeat=2):
        for s in range (4):
            ci_i, ci_a = ci_ch[i][a][s]
            if ci_i is None or ci_a is None: continue
            dsi = -1 + (s//2)*2
            dsa = -1 + (s%2)*2
            spaces2.append (space0.get_single_any_m (i, a, dsi, dsa, ci_i=ci_i, ci_a=ci_a))
    if lsi.nfrags > 3:
        spaces2 = charge_excitation_products (lsi, spaces2, nroots_ref=1)
    spaces3 = spin_flip_products (las, spaces2, spin_flips, nroots_ref=1)
    weights = [space.weight for space in spaces3]
    charges = [space.charges for space in spaces3]
    spins = [space.spins for space in spaces3]
    smults = [space.smults for space in spaces3]
    ci3 = [[space.ci[ifrag] for space in spaces3] for ifrag in range (lsi.nfrags)]
    las3 = las.state_average (weights=weights, charges=charges, spins=spins, smults=smults, assert_no_dupes=False)
    las3.ci = ci3
    las3.lasci (_dry_run=True)
    log.timer ("LASSIS model space preparation", *t0)
    return las3

def prepare_states (lsi, ncharge=1, nspin=0, sa_heff=True, deactivate_vrv=False, crash_locmin=False):
    # TODO: make states_energy_elec capable of handling lroots and address inconsistency
    # between definition of e_states array for neutral and charge-separated rootspaces
    t0 = (logger.process_clock (), logger.perf_counter ())
    ham_2q = lsi.ham_2q ()
    log = logger.new_logger (lsi, lsi.verbose)
    las = lsi._las.get_single_state_las (state=0)
    # 1. Spin shuffle step
    if np.all (get_space_info (las)[2]==1):
        # If all singlets, skip the spin shuffle and the unnecessary warning below
        las1 = las
    else:
        las1 = spin_shuffle (las, equal_weights=True)
        # TODO: memory efficiency; the line below makes copies
        las1.ci = spin_shuffle_ci (las1, las1.ci)
        las1.converged = las.converged
    nroots_ref = las1.nroots
    if las1.nroots==1:
        log.info ("LASSIS reference spaces: 0")
    else:
        log.info ("LASSIS reference spaces: 0-%d", nroots_ref-1)
    for ix, space in enumerate (list_spaces (las1)):
        log.info ("Reference space %d:", ix)
        space.table_printlog ()
    # 2. Spin excitations part 1
    spin_flips = all_spin_flips (lsi, las1, nspin=nspin, ham_2q=ham_2q) if nspin else None
    las1.e_states = las1.energy_nuc () + np.array (las1.states_energy_elec ())
    # 3. Charge excitations
    # TODO: Store the irreducible degrees of freedom of the charge excitations more transparently,
    # like spin_flips above.
    if ncharge:
        las2 = all_single_excitations (las1)
        converged, spaces2 = single_excitations_ci (
            lsi, las2, las1, ncharge=ncharge, sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
            spin_flips=spin_flips, crash_locmin=crash_locmin, ham_2q=ham_2q
        )
    else:
        converged = las1.converged
        spaces2 = list_spaces (las1)
    if lsi.nfrags > 3:
        spaces2 = charge_excitation_products (lsi, spaces2, nroots_ref=las1.nroots)
    # 4. Spin excitations part 2
    if nspin:
        spaces3 = spin_flip_products (las1, spaces2, spin_flips, nroots_ref=nroots_ref)
    else:
        spaces3 = spaces2
    weights = [space.weight for space in spaces3]
    charges = [space.charges for space in spaces3]
    spins = [space.spins for space in spaces3]
    smults = [space.smults for space in spaces3]
    ci3 = [[space.ci[ifrag] for space in spaces3] for ifrag in range (lsi.nfrags)]
    las3 = las1.state_average (weights=weights, charges=charges, spins=spins, smults=smults, assert_no_dupes=False)
    las3.ci = ci3
    las3.lasci (_dry_run=True)
    log.timer ("LASSIS model space preparation", *t0)
    return converged, las3

def single_excitations_ci (lsi, las2, las1, ncharge=1, sa_heff=True, deactivate_vrv=False,
                           spin_flips=None, crash_locmin=False, ham_2q=None):
    log = logger.new_logger (lsi, lsi.verbose)
    mol = lsi.mol
    nfrags = lsi.nfrags
    e_roots = np.append (las1.e_states, np.zeros (las2.nroots-las1.nroots))
    spaces = list_spaces (las2)
    ncsf = las2.get_ugg ().ncsf_sub
    auto_singles = False
    if isinstance (ncharge, np.ndarray):
        ncharge=ncharge[None,:]
    elif isinstance (ncharge, str):
        if 's' in ncharge.lower ():
            auto_singles = True
            ncharge = ncsf
        else:
            raise RuntimeError ("Valid ncharge values are integers or 's'")
    lroots = np.minimum (ncharge, ncsf)
    if ham_2q is None:
        h0, h1, h2 = lsi.ham_2q ()
    else:
        h0, h1, h2 = ham_2q
    t0 = (logger.process_clock (), logger.perf_counter ())
    converged = True
    # Prefilter spin-shuffles
    spaces = [spi for i, spi in enumerate (spaces)
              if (i<las1.nroots) or not any (
                [spi.is_spin_shuffle_of (spj) for spj in spaces[:i]]
              )]
    log.info ("LASSIS electron hop spaces: %d-%d", las1.nroots, len (spaces)-1)
    keys = set ()
    for i in range (las1.nroots, len (spaces)):
        # compute lroots
        psref_ix = [j for j, space in enumerate (spaces[:las1.nroots])
                    if spaces[i].is_single_excitation_of (space)]
        psref = [spaces[j] for j in psref_ix]
        excfrags = np.zeros (nfrags, dtype=bool)
        for space in psref: excfrags[spaces[i].excited_fragments (space)] = True
        nref_pure = len (psref)
        psref = _spin_flip_products (psref, spin_flips, nroots_ref=len(psref),
                                               frozen_frags=(~excfrags))
        psref = [space for space in psref if spaces[i].is_single_excitation_of (space)]
        if auto_singles:
            lr = spaces[i].compute_single_excitation_lroots (psref)
            lroots[:,i][excfrags] = np.minimum (lroots[:,i][excfrags], lr)
        lroots[:,i][~excfrags] = 1
        # logging after setup
        log.info ("Electron hop space %d:", i)
        spaces[i].table_printlog (lroots=lroots[:,i])
        log.info ("is connected to reference spaces:")
        for j in psref_ix:
            log.info ('%d by %s', j, spaces[i].single_excitation_description_string (spaces[j]))
            key = spaces[i].single_excitation_key (spaces[j])
        if len (psref) > nref_pure:
            log.info ("as well as spin-excited spaces:")
            for space in psref[nref_pure:]:
                space.table_printlog ()
                log.info ('by %s', spaces[i].single_excitation_description_string (space))
        assert (key not in keys), 'Problem enumerating model states! Talk to Matt about it!'
        keys.add (key)
        # throat-clearing into ExcitationPSFCISolver
        ciref = [[] for j in range (nfrags)]
        for k in range (nfrags):
            for space in psref: ciref[k].append (space.ci[k])
        spaces[i].set_entmap_(psref[0])
        psref = [space.get_product_state_solver () for space in psref]
        psexc = ExcitationPSFCISolver (psref, ciref, las2.ncas_sub, las2.nelecas_sub,
                                       stdout=mol.stdout, verbose=mol.verbose,
                                       crash_locmin=crash_locmin, opt=lsi.opt)
        psexc._deactivate_vrv = deactivate_vrv
        norb = spaces[i].nlas
        neleca = spaces[i].neleca
        nelecb = spaces[i].nelecb
        smults = spaces[i].smults
        for k in np.where (excfrags)[0]:
            weights = np.zeros (lroots[k,i])
            if sa_heff: weights[:] = 1.0 / len (weights)
            else: weights[0] = 1.0
            psexc.set_excited_fragment_(k, (neleca[k],nelecb[k]), smults[k], weights=weights)
        ifrag, afrag, spin = key
        norb_i, norb_a, smult_i, smult_a = norb[ifrag], norb[afrag], smults[ifrag], smults[afrag]
        nelec_i, nelec_a = (neleca[ifrag],nelecb[ifrag]), (neleca[afrag],nelecb[afrag])
        # Going into psexc.kernel, they have to be in lexical order
        ci0 = lsi.ci_charge_hops[ifrag][afrag][spin]
        if ci0[0] is not None: ci0[0] = mdown (ci0[0], norb_i, nelec_i, smult_i)
        if ci0[1] is not None: ci0[1] = mdown (ci0[1], norb_a, nelec_a, smult_a)
        ci0 = [ci0[int (afrag<ifrag)], ci0[int (ifrag<afrag)]]
        conv, e_roots[i], ci1 = psexc.kernel (h1, h2, ecore=h0, ci0=ci0,
                                              max_cycle_macro=lsi.max_cycle_macro,
                                              conv_tol_self=lsi.conv_tol_self)
        lsi.ci_charge_hops[ifrag][afrag][spin] = [
            mup (ci1[ifrag], norb_i, nelec_i, smult_i),
            mup (ci1[afrag], norb_a, nelec_a, smult_a)
        ]
        if len (psref)>1:
            for k in np.where (~excfrags)[0]: ci1[k] = ci1[k][0]
        spaces[i].ci = ci1
        if not conv: log.warn ("CI vectors for charge-separated rootspace %d not converged", i)
        converged = converged and conv
        t0 = log.timer ("Space {} excitations".format (i), *t0)
    return converged, spaces

class SpinFlips (object):
    '''For a single fragment, bundle the ci vectors of various spin-flipped states with their
       corresponding quantum numbers. Instances of this object are stored together in a list
       where position indicates fragment identity.'''
    def __init__(self, ci, norb, nelec, spins, smults):
        self.norb = norb
        self.nelec = nelec
        self.ci = ci
        self.spins = spins
        self.smults = smults
        # Assumes you only assigned the m=s case
        for i in range (len (ci)):
            ci, spin, smult = self.ci[i], self.spins[i], self.smults[i]
            if smult>1:
                neleca = (self.nelec + (smult-1)) // 2
                nelecb = (self.nelec - (smult-1)) // 2
                ci_list = list (ci)
                for ms in range (smult-1):
                    ci_list = [contract_sdown (c, norb, (neleca,nelecb)) for c in ci_list]
                    neleca -= 1
                    nelecb += 1
                    self.ci.append (np.array (ci_list))
                    self.smults.append (smult)
                    self.spins.append (neleca-nelecb)
                
            

def all_spin_flips (lsi, las, nspin=1, ham_2q=None):
    # NOTE: this actually only uses the -first- rootspace in las, so it can be done before
    # the initial spin shuffle
    log = logger.new_logger (lsi, lsi.verbose)
    norb_f = las.ncas_sub
    spaces = list_spaces (las)
    if len (spaces) > 1:
        assert (all ([np.all(spaces[i].nelec==spaces[i-1].nelec) for i in range (1,len(spaces))]))
        assert (all ([np.all(spaces[i].smults==spaces[i-1].smults) for i in range (1,len(spaces))]))
    norb0 = las.ncas_sub
    nelec0 = spaces[0].nelec
    spins0 = spaces[0].spins
    smults0 = spaces[0].smults
    nfrags = spaces[0].nfrag
    smults1 = []
    spins1 = []
    ci1 = []
    if ham_2q is None:
        h0, h1, h2 = lsi.ham_2q ()
    else:
        h0, h1, h2 = ham_2q
    casdm1s = las.make_casdm1s ()
    f1 = h1 + np.tensordot (h2, casdm1s.sum (0), axes=2)
    f1 = f1[None,:,:] - np.tensordot (casdm1s, h2, axes=((1,2),(2,1)))
    i = 0
    auto_singles = isinstance (nspin, str) and 's' in nspin.lower ()
    nup0 = np.minimum (spaces[0].nelecd, spaces[0].nholeu)
    ndn0 = np.minimum (spaces[0].nelecu, spaces[0].nholed)
    if not auto_singles: # integer supplied by caller
        nup0[:] = nspin
        ndn0[:] = nspin
    for ifrag, (norb, nelec, spin, smult) in enumerate (zip (norb0, nelec0, spins0, smults0)):
        j = i + norb
        h2_i = h2[i:j,i:j,i:j,i:j]
        lasdm1s = casdm1s[:,i:j,i:j]
        h1_i = (f1[:,i:j,i:j] - np.tensordot (h2_i, lasdm1s.sum (0))[None,:,:]
                + np.tensordot (lasdm1s, h2_i, axes=((1,2),(2,1))))
        def cisolve (sm, nroots, ci0):
            neleca = (nelec + (sm-1)) // 2
            nelecb = (nelec - (sm-1)) // 2
            solver = csf_solver (las.mol, smult=sm).set (nelec=(neleca,nelecb), norb=norb)
            solver.check_transformer_cache ()
            nroots = min (nroots, solver.transformer.ncsf)
            ci_list = solver.kernel (h1_i, h2_i, norb, (neleca,nelecb), ci0=ci0, nroots=nroots)[1]
            if nroots==1: ci_list = [ci_list,]
            ci_arr = np.array (ci_list)
            return ci_arr
        smults1_i = []
        spins1_i = []
        ci1_i = []
        if smult > 2: # spin-lowered
            log.info ("LASSIS fragment %d spin down (%de,%do;2S+1=%d)",
                      ifrag, nelec, norb, smult-2)
            smults1_i.append (smult-2)
            spins1_i.append (smult-3)
            ci0 = lsi.ci_spin_flips[ifrag][0]
            ci1_i_down = cisolve (smult-2, ndn0[ifrag], ci0)
            lsi.ci_spin_flips[ifrag][0] = ci1_i_down
            ci1_i.append (ci1_i_down)
        min_npair = max (0, nelec-norb)
        max_smult = (nelec - 2*min_npair) + 1
        if smult < max_smult: # spin-raised
            log.info ("LASSIS fragment %d spin up (%de,%do;2S+1=%d)",
                      ifrag, nelec, norb, smult+2)
            smults1_i.append (smult+2)
            spins1_i.append (smult+1)
            ci0 = lsi.ci_spin_flips[ifrag][1]
            ci1_i_up = cisolve (smult+2, nup0[ifrag], ci0)
            lsi.ci_spin_flips[ifrag][1] = ci1_i_up
            ci1_i.append (ci1_i_up)
        smults1.append (smults1_i)
        spins1.append (spins1_i)
        ci1.append (ci1_i)
        i = j
    spin_flips = [SpinFlips (c,no,ne,m,s) for c,no,ne,m,s in zip (ci1,norb0,nelec0,spins1,smults1)]
    return spin_flips

def _spin_flip_products (spaces, spin_flips, nroots_ref=1, frozen_frags=None):
    # NOTE: this actually only uses the -first- rootspace in las, so it can be done before
    # the initial spin shuffle
    '''Combine spin-flip excitations in all symmetrically permissible ways'''
    if spin_flips is None or len (spin_flips)==0: return spaces
    spaces_ref = spaces[:nroots_ref]
    spins3 = [she.spins for she in spin_flips]
    smults3 = [she.smults for she in spin_flips]
    ci3 = [she.ci for she in spin_flips]
    nelec0 = spaces[0].nelec
    smults0 = spaces[0].smults
    nfrags = spaces[0].nfrag
    spin = spaces[0].spins.sum ()
    if frozen_frags is None: frozen_frags = np.zeros (nfrags, dtype=bool)
    for ifrag in range (nfrags):
        if frozen_frags[ifrag]: continue
        new_spaces = []
        m3, s3, c3 = spins3[ifrag], smults3[ifrag], ci3[ifrag]
        for space in spaces:
            # I want to inject the spin-flip into all distinct references,
            # but if two references differ only in ifrag then this would
            # generate duplicates. The two lines below filter this case.
            if space.nelec[ifrag] != nelec0[ifrag]: continue
            if space.smults[ifrag] != smults0[ifrag]: continue
            for m3i, s3i, c3i in zip (m3, s3, c3):
                new_spaces.append (space.single_fragment_spin_change (
                    ifrag, s3i, m3i, ci=c3i))
        spaces += new_spaces
    # Filter by ms orthogonality
    spaces = [space for space in spaces if space.spins.sum () == spin]
    # Filter by smult orthogonality
    spaces = [space for space in spaces 
              if (not (all (space.is_orthogonal_by_smult (spaces_ref))))]
    seen = set ()
    # Filter duplicates!
    spaces = [space for space in spaces if not ((space in seen) or seen.add (space))]
    return spaces

def _spin_shuffle_ci_(spaces, spin_flips, nroots_ref, nroots_refc):
    '''Memory-efficient version of the function spaces._spin_shuffle_ci_.
    Based on the fact that we know there has only been one independent set
    of vectors per fragment Hilbert space and that all possible individual
    fragment spins must be accounted for already, so we are just recombining
    them.'''
    old_idx = []
    new_idx = []
    nfrag = spaces[0].nfrag
    for ix, space in enumerate (spaces):
        if space.has_ci ():
            old_idx.append (ix)
        else:
            assert (ix >= nroots_refc)
            new_idx.append (ix)
            space.ci = [None for ifrag in range (space.nfrag)]
    # Prepare charge-hop szrots
    spaces_1c = spaces[nroots_ref:nroots_refc]
    spaces_1c = [space for space in spaces_1c if len (space.entmap)==1]
    ci_szrot_1c = []
    for ix, space in enumerate (spaces_1c):
        ifrag, jfrag = space.entmap[0] # must be a tuple of length 2
        ci_szrot_1c.append (space.get_ci_szrot (ifrags=(ifrag,jfrag)))
    charges0 = spaces[0].charges
    smults0 = spaces[0].smults
    # Prepare reference szrots
    ci_szrot_ref = spaces[0].get_ci_szrot ()
    for ix in new_idx:
        idx = spaces[ix].excited_fragments (spaces[0])
        space = spaces[ix]
        for ifrag in np.where (~idx)[0]:
            space.ci[ifrag] = spaces[0].ci[ifrag]
        for ifrag in np.where (idx)[0]:
            if space.charges[ifrag] != charges0[ifrag]: continue
            if space.smults[ifrag] != smults0[ifrag]:
                sf = spin_flips[ifrag]
                iflp = sf.smults == space.smults[ifrag]
                iflp &= sf.spins == space.spins[ifrag]
                assert (np.count_nonzero (iflp) == 1)
                iflp = np.where (iflp)[0][0]
                space.ci[ifrag] = sf.ci[iflp]
            else: # Reference-state spin-shuffles
                space.ci[ifrag] = ci_szrot_ref[ifrag][space.spins[ifrag]]
        for (ci_i, ci_j), sp_1c in zip (ci_szrot_1c, spaces_1c):
            ijfrag = sp_1c.entmap[0]
            if ijfrag not in spaces[ix].entmap: continue
            if np.any (sp_1c.charges[list(ijfrag)] != space.charges[list(ijfrag)]): continue
            if np.any (sp_1c.smults[list(ijfrag)] != space.smults[list(ijfrag)]): continue
            ifrag, jfrag = ijfrag
            assert (space.ci[ifrag] is None)
            assert (space.ci[jfrag] is None)
            space.ci[ifrag] = ci_i[space.spins[ifrag]]
            space.ci[jfrag] = ci_j[space.spins[jfrag]]
        assert (space.has_ci ()), '{} {} {} {}'.format (space.charges, space.smults, space.spins, charges0)
    return spaces

def spin_flip_products (las, spaces, spin_flips, nroots_ref=1):
    '''Inject spin-flips into las2 in all possible ways'''
    log = logger.new_logger (las, las.verbose)
    las2_nroots = len (spaces)
    spaces = _spin_flip_products (spaces, spin_flips, nroots_ref=nroots_ref)
    nfrags = spaces[0].nfrag
    spaces = _spin_shuffle (spaces)
    spaces = _spin_shuffle_ci_(spaces, spin_flips, nroots_ref, las2_nroots)
    log.info ("LASSIS spin-excitation spaces: %d-%d", las2_nroots, len (spaces)-1)
    for i, space in enumerate (spaces[las2_nroots:]):
        if np.any (space.nelec != spaces[0].nelec):
            log.info ("Spin/charge-excitation space %d:", i+las2_nroots)
        else:
            log.info ("Spin-excitation space %d:", i+las2_nroots)
        space.table_printlog ()
    return spaces

def charge_excitation_products (lsi, spaces, nroots_ref=0, space0=None):
    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.new_logger (lsi, lsi.verbose)
    mol = lsi.mol
    nfrags = lsi.nfrags
    if space0 is None: space0 = spaces[0]
    i0, j0 = i, j = nroots_ref, len (spaces)
    for product_order in range (2, (nfrags//2)+1):
        seen = set ()
        for i_list in itertools.combinations (range (i,j), product_order):
            p_list = [spaces[ip] for ip in i_list]
            nonorth = False
            for p, q in itertools.combinations (p_list, 2):
                if not orthogonal_excitations (p, q, space0):
                    nonorth = True
                    break
            if nonorth: continue
            p = p_list[0]
            for q in p_list[1:]:
                p = combine_orthogonal_excitations (p, q, space0)
            spaces.append (p)
            log.info ("Electron hop product space %d (product of %s)", len (spaces) - 1, str (i_list))
            spaces[-1].table_printlog ()
    assert (len (spaces) == len (set (spaces)))
    log.timer ("LASSIS charge-hop product generation", *t0)
    return spaces

def as_scanner(lsi):
    '''Generating a scanner for LASSIS PES.
    
    The returned solver is a function. This function requires one argument
    "mol" as input and returns total LASSIS energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters of LASSIS object
    are automatically applied in the solver.
    
    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.
    ''' 
    if isinstance(lsi, lib.SinglePointScanner):
        return lsi
        
    logger.info(lsi, 'Create scanner for %s', lsi.__class__)
    name = lsi.__class__.__name__ + LASSIS_Scanner.__name_mixin__
    return lib.set_class(LASSIS_Scanner(lsi), (LASSIS_Scanner, lsi.__class__), name)
        
class LASSIS_Scanner(lib.SinglePointScanner):
    def __init__(self, lsi, state=0):
        self.__dict__.update(lsi.__dict__)
        self._las = lsi._las.as_scanner()
        self._scan_state = state

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
    
        self.reset (mol)
        for key in ('with_df', 'with_x2c', 'with_solvent', 'with_dftd3'):
            sub_mod = getattr(self, key, None)
            if sub_mod:
                sub_mod.reset(mol)

        las_scanner = self._las
        las_scanner(mol)
        self.mol = mol
        self.mo_coeff = las_scanner.mo_coeff
        e_tot = self.kernel()[0][self._scan_state]
        if hasattr (e_tot, '__len__'):
            e_tot = np.average (e_tot)
        return e_tot

class LASSIS (LASSI):
    def __init__(self, las, ncharge='s', nspin='s', sa_heff=True, deactivate_vrv=False,
                 crash_locmin=False, opt=1, **kwargs):
        '''
        Key attributes:
            _las : instance of class `LASCINoSymm`
                The encapsulated LASSCF wave function. The CI vectors of the reference state are,
                i.e., _las.get_single_state_las (state=0).ci.
            ci_spin_flips : doubly nested list of ndarrays
                Element [i][s] are the spin-flip CI vectors of the ith fragment in the direction
                s = 0,1 = -,+.
            ci_charge_hops: quadruply nested list of ndarrays
                Element [i][a][s][p] are charge-hop CI vectors for an electron hopping from the
                ith to the ath fragment for spin case s = 0,1,2,3 = --,-+,+-,++, and fragment
                p = 0,1 = i,a.
        '''
        self.ncharge = ncharge
        self.nspin = nspin
        self.sa_heff = sa_heff
        self.deactivate_vrv = deactivate_vrv
        self.crash_locmin = crash_locmin
        self.e_states_meaningless = True # a tag to silence an invalid warning
        LASSI.__init__(self, las, opt=opt, **kwargs)
        self.max_cycle_macro = 50
        self.conv_tol_self = 1e-6
        self.ci_spin_flips = [[None for s in range (2)] for i in range (self.nfrags)]
        self.ci_charge_hops = [[[[None,None] for s in range (4)]
                                for a in range (self.nfrags)]
                               for i in range (self.nfrags)]
        self._cached_ham_2q = None
        if las.nroots>1:
            logger.warn (self, ("Only the first LASSCF state is used by LASSIS! "
                                "Other states are discarded!"))

    def ham_2q (self, *args, **kwargs):
        if self._cached_ham_2q is not None: return self._cached_ham_2q
        return super().ham_2q (*args, **kwargs)

    def kernel (self, ncharge=None, nspin=None, sa_heff=None, deactivate_vrv=None,
                crash_locmin=None, **kwargs):
        t0 = (logger.process_clock (), logger.perf_counter ())
        log = logger.new_logger (self, self.verbose)
        h0, h1, h2 = self.ham_2q ()
        t1 = log.timer ("LASSIS integral transformation", *t0)
        with lib.temporary_env (self, _cached_ham_2q=(h0,h1,h2)):
            self.converged = self.prepare_states_(ncharge=ncharge, nspin=nspin,
                                                  sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
                                                  crash_locmin=crash_locmin)
            t1 = log.timer ("LASSIS state preparation", *t1)
            self.e_roots, self.si = self.eig (**kwargs)
            t1 = log.timer ("LASSIS diagonalization", *t1)
        log.timer ("LASSIS", *t0)
        return self.e_roots, self.si

    def get_ci_ref (self):
        las = self._las.get_single_state_las (state=0)
        space0 = list_spaces (las)[0]
        ci_ref = []
        for i in range (self.nfrags):
            ci_ref.append (mup (space0.ci[i], space0.nlas[i], (space0.neleca[i], space0.nelecb[i]),
                                space0.smults[i]))
        return ci_ref

    def prepare_states_(self, ncharge=None, nspin=None, sa_heff=None, deactivate_vrv=None,
                        crash_locmin=None, **kwargs):
        if ncharge is None: ncharge = self.ncharge
        if nspin is None: nspin = self.nspin
        if sa_heff is None: sa_heff = self.sa_heff
        if deactivate_vrv is None: deactivate_vrv = self.deactivate_vrv
        if crash_locmin is None: crash_locmin = self.crash_locmin
        log = logger.new_logger (self, self.verbose)
        self.converged, las = self.prepare_states (ncharge=ncharge, nspin=nspin,
                                                   sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
                                                   crash_locmin=crash_locmin)
        ci_ref = self.get_ci_ref ()
        ci_sf = self.ci_spin_flips
        ci_ch = self.ci_charge_hops
        las = self.prepare_model_states (ci_ref, ci_sf, ci_ch)
        #self.__dict__.update(las.__dict__) # Unsafe
        self.fciboxes = las.fciboxes
        self.ci = las.ci
        self.nroots = las.nroots
        self.weights = las.weights
        self.e_lexc = las.e_lexc
        self.e_states = las.e_states
        log.info ('LASSIS model state summary: %d rootspaces; %d model states; converged? %s',
                  self.nroots, self.get_lroots ().prod (0).sum (), str (self.converged))
        return self.converged

    eig = LASSI.kernel
    as_scanner = as_scanner
    prepare_states = prepare_states
    prepare_model_states = prepare_model_states

