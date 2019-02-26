import numpy as np
import scipy
from mrh.util import params

# A collection of simple manipulations of matrices that I somehow can't find in numpy

def is_matrix_zero (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_zero = np.zeros (test_matrix.shape, dtype=test_matrix.dtype)
    return np.allclose (test_matrix, test_zero, rtol=rtol, atol=atol)

def is_matrix_eye (test_matrix, matdim=None, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    if (test_matrix.shape[0] != test_matrix.shape[1]):
        return False
    test_eye = np.eye (test_matrix.shape[0], dtype=test_matrix.dtype)
    return np.allclose (test_matrix, test_eye, atol=atol, rtol=rtol)

def is_matrix_idempotent (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    if (test_matrix.shape[0] != test_matrix.shape[1]):
        return False
    test_m2 = np.dot (test_matrix, test_matrix)
    return np.allclose (test_matrix, test_m2, atol=atol, rtol=rtol)

def is_matrix_diagonal (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_diagonal = np.diag (np.diag (test_matrix))
    return np.allclose (test_matrix, test_diagonal, atol=atol, rtol=rtol)

def is_matrix_hermitian (test_matrix, rtol=params.num_zero_rtol, atol=params.num_zero_atol):
    test_adjoint = np.transpose (np.conjugate (test_matrix))
    return np.allclose (test_matrix, test_adjoint, atol=atol, rtol=rtol)

def assert_matrix_square (test_matrix, matdim=None):
    if (matdim == None):
        matdim = test_matrix.shape[0]
    assert ((test_matrix.ndim == 2) and (test_matrix.shape[0] == matdim) and (test_matrix.shape[1] == matdim)), "Matrix shape is {0}; should be ({1},{1})".format (test_matrix.shape, matdim)
    return matdim

def matrix_svd_control_options (the_matrix, full_matrices=False, only_nonzero_vals=False, sort_vecs=-1,
    lspace=None, rspace=None, lsymm=None, rsymm=None, symmetry=None,
    lspace_symmetry=None, rspace_symmetry=None, strong_symm=False,
    num_zero_rtol=params.num_zero_rtol, num_zero_atol=params.num_zero_atol):
    ''' Perform SVD of a matrix using scipy's linalg driver with a lot of pre-/post-processing

        Args:
            the_matrix: ndarray of shape (M,N) or scalar
                If a scalar, is treated as I * the_matrix, where I has length P given by the
                number of rows in "lspace" if lspace is provided as a vector block or
                the number of rows in "rspace" otherwise.
                

        Kwargs:
            full_matrices: logical
                If true, lvecs and rvecs include the null space and have shapes (M,M) and (N,N).
                Otherwise, lvecs and rvecs omit the null space and have shapes (M,K) and (N,K).
            only_nonzero_vals: logical
                If true, the formal [max (M,N) - K] and numerical (sval == 0)
                null spaces are both omitted from the returned singular values
                and possibly (depending on the value of full_matrices)
                the left- and right-singular vectors: K = count_nonzero (svals).
                Otherwise: only the formal null space is omitted: K = min (M,N)
            sort_vecs: integer
                Defines overall sorting of non-degenerate eigenvalues. -1 means from largest
                to smallest; +1 means from smallest to largest. Play with other values at
                your peril
            lspace: index list for accessing Mprime elements array of shape (M,)
            or ndarray of shape (M,Mprime)
            or None
                Defines a subspace for the rows in which SVD is performed.
            rspace: index list for accessing Nprime elements array of shape (N,)
            or ndarray of shape (N,Nprime)
            or None
                Defines a subspace for the columns in which SVD is performed.
            lsymm: list of block labels of length M
            or list of non-square matrices of shape (M,P), sum_P = M
            or None
                Formal symmetry information for the rows of the matrix. Neither the matrix nor the
                subspace need to be symmetry adapted, and unless strong_symm=True,
                symmetry is used only to define a gauge convention within
                degenerate manifolds. Orthonormal linear combinations of coupled left- and right-singular
                vectors with the highest possible projections onto any symmetry
                block are sequentially generated using repeated singular value decompositions
                in successively smaller subspaces. Vectors within degenerate
                manifolds are therefore sorted from least to most symmetry-breaking;
                quantitatively symmetry-adapted vectors are grouped by block 
                with the blocks in a currently arbitrary order.
            rsymm: list of block labels of length N
            or list of non-square matrices of shape (N,P), sum_P = N
            or None
                Formal symmetry information for the columns of the matrix. See lsymm
            symmetry: list of block labels of length ?
            or list of non-square matrices of shape (?,P), sum_P = ?
            or None
                Convenience argument to set both lsymm and rsymm to the same value. Overridden
                by either lsymm or rsymm. User beware: no dimensionality checking.
            lspace_symmetry: list of block labels of length Mprime
            or list of non-square matrices of shape (Mprime,P), sum_P = Mprime
            or None
                Formal symmetry information of the row subspace. Set to None if lspace is None.
                If not None, "lsymm" is ignored (overrides "symmetry"). 
            rspace_symmetry: list of block labels of length Nprime
            or list of non-square matrices of shape (Nprime,P), sum_P = Nprime
            or None
                Formal symmetry information of the column subspace. Set to None if rspace is None.
                If not None, "rsymm" is ignored (overrides "symmetry").
            num_zero_atol: float
                Absolute tolerance for numpy's "isclose" function and its relatives.
            num_zero_rtol: float
                Relative tolerance for numpy's "isclose" function and its relatives.

        Returns:
            lvecs: ndarray of shape (M,M) or (M,K), K <= M (see full_matrices and only_nonzero_vals kwargs)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning (Mprime -> M)
            svals: ndarray of shape (K,), K <= min (M,N) (see only_nonzero_vals kwarg)
            rvecs: ndarray of shape (N,N) or (N,K), K <= N (see full_matrices and only_nonzero_vals kwargs)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning (Nprime -> N)
            llabels: list of length M or K, K <= M
                Only returned if lsymm or symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each left-singular vector.
                Does not guarantee that the vectors are symmetry-adapted.
            rlabels: list of length N or K, K <= N
                Only returned if rsymm or symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each left-singular vector.
                Does not guarantee that the vectors are symmetry-adapted.


    '''

    # Interpret subspace information
    lspace = None if lspace is None else np.asarray (lspace)
    lspace_isvectorblock = False if lspace is None else lspace.ndim == 2
    rspace = None if rspace is None else np.asarray (rspace)
    rspace_isvectorblock = False if rspace is None else rspace.ndim == 2

    # Interpret subspace symmetry information
    if strong_symm:
        if lsymm is None and symmetry is None and lspace_symmetry is None: 
            raise RuntimeError ("Can't do SVD by symmetry blocks if no row symmetry information provided!")
        if rsymm is None and symmetry is None and rspace_symmetry is None: 
            raise RuntimeError ("Can't do SVD by symmetry blocks if no column symmetry information provided!")
    if lspace is None: lspace_symmetry = None
    if rspace is None: rspace_symmetry = None
    return_llabels = not (lsymm is None) or not (symmetry is None) or not (lspace_symmetry is None)
    return_rlabels = not (rsymm is None) or not (symmetry is None) or not (rspace_symmetry is None)
    lspace_symm_isvectorblock = False if lspace_symmetry is None else isinstance (rspace_symmetry[0], np.ndarray)
    rspace_symm_isvectorblock = False if rspace_symmetry is None else isinstance (rspace_symmetry[0], np.ndarray)
    if lspace_symmetry is not None: lsymm = None
    if rspace_symmetry is not None: rsymm = None

    # Interpret full-space symmetry information
    if lsymm is None and lspace_symmetry is None and symmetry is not None: lsymm = symmetry
    if rsymm is None and rspace_symmetry is None and symmetry is not None: rsymm = symmetry
    lsymm_isvectorblock = False if lsymm is None else isinstance (lsymm[0], np.ndarray)
    rsymm_isvectorblock = False if rsymm is None else isinstance (rsymm[0], np.ndarray)

    # Shape construction and zero matrix escape
    def _interpret_shape (mat, spc, axis):
        Pbasis = P = None
        if isinstance (spc, np.ndarray) and spc.ndim == 2:
            Pbasis, P = spc.shape
        elif isinstance (spc, np.ndarray) and np.can_cast (spc, np.bool_):
            Pbasis = spc.size
            P = np.count_nonzero (spc)
        elif isinstance (spc, np.ndarray):
            P = spc.size
        if isinstance (mat, np.ndarray) and mat.ndim == 2:
            Pbasis = mat.shape[axis]
        if Pbasis is not None and P is None: P = Pbasis
        return Pbasis, P
    Mbasis, M = _interpret_shape (the_matrix, lspace, 0)
    Nbasis, N = _interpret_shape (the_matrix, rspace, 1)
    if Mbasis is None and Nbasis is None: raise RuntimeError ("Insufficient information to determine shape of matrix")
    if Nbasis is None: Nbasis = Mbasis
    if Mbasis is None: Mbasis = Nbasis
    if M is None: M = Mbasis
    if N is None: N = Nbasis
    if 0 in (M, N):
        if full_matrices: return np.zeros ((Mbasis,M)), np.zeros ((0)), np.zeros ((Nbasis,N))
        return np.zeros ((Mbasis,0)), np.zeros ((K)), np.zeros ((Nbasis,0))

    # If subspace symmetry is provided as a vector block, transform subspace into a symmetry-adapted form
    # No recursion necessary because the eigenvectors are meant to be provided in the full basis :)
    def _symmadapt_subspace (space, symm):
        if symm is None or symm.ndim < 2: return space, symm
        if space.ndim > 1:
            space = space @ np.concatenate (symm, axis=1)
        else:
            space = np.concatenate (symm, axis=1)[space,:]
        symm = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (symm)])
        return space, symm
    lspace, lspace_symmetry = _symmadapt_subspace (lspace, lspace_symmetry)
    rspace, rspace_symmetry = _symmadapt_subspace (rspace, rspace_symmetry)

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    def _symmadapt_recurse (symm, space, space_isvectorblock, axis):
        Pbasis = Nbasis if axis else Mbasis
        symm_umat = np.concatenate (symm, axis=1)
        assert (symm_umat.shape == tuple((Pbasis, Pbasis))), "I can't guess how to map symmetry blocks to different bases"
        symm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (symm)])
        if isinstance (the_matrix, np.ndarray):
            symm_matr = symm_umat.conjugate ().T @ the_matrix if axis else the_matrix @ symm_umat
        else:
            symm_matr = symm_umat.conjugate ().T * the_matrix if axis else the_matrix * symm_umat
        if space is not None:
            # I have to turn the subspace into a vector block!
            if space_isvectorblock: symm_subs = symm_umat.conjugate ().T @ space
            else: symm_subs = symm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (symm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        if axis:
            rspace, rsymm = symm_subs, symm_lbls
        else:
            lspace, lsymm = symm_subs, symm_lbls
        rets = matrix_svd_control_options (symm_matr, only_nonzero_vals=only_nonzero_vals,
            full_matrices=full_matrices, sort_vecs=sort_vecs,
            lspace=lspace, rspace=rspace, lsymm=lsymm, rsymm=rsymm, symmetry=None,
            lspace_symmetry=lspace_symmetry, rspace_symmetry=rspace_symmetry,
            num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        rets = [symm_umat @ x if idx == axis*2 else x for idx, x in enumerate (rets)]
        return rets
    if lsymm_isvectorblock: return _symmadapt_recurse (lsymm, lspace, lspace_isvectorblock, 0)
    if rsymm_isvectorblock: return _symmadapt_recurse (rsymm, rspace, rspace_isvectorblock, 1)

    # Recurse from strong symmetry enforcement to SVD over individual symmetry blocks
    if strong_symm:

        # If a subspace is being diagonalized, recurse into symmetry blocks via the subspaces
        def _unpack_space_symm (space, symm, space_symmetry):
            if space is None:
                symm_lbls = symm
                space = np.ones (M, dtype=np.bool_)
            elif space_symmetry is not None:
                symm_lbls = space_symmetry
            elif space_isvectorblock:
                space, symm_lbls = align_vecs (space, symm, rtol=num_zero_rtol, atol=num_zero_atol)
            else:
                symm_lbls = symm[space]
            return space, symm_lbls
        lspace, lsymm_lbls = _unpack_space_symm (lspace, lsymm, lspace_symmetry)
        rspace, rsymm_lbls = _unpack_space_symm (rspace, rsymm, rspace_symmetry)
            
        uniq_common_lbls = np.unique (np.append (lsymm_lbls, rsymm_lbls))
        uniq_null_lbls = [np.setdiff1d (np.unique (x), uniq_common_lbls) for x in (lsymm_lbls, rsymm_lbls)]
        svals = []
        vecs = [[], []]
        vecs_null = [[], []]
        labels = [[], []]
        labels_null = [[], []]
        vecs_blk = [[], []]
        for lbl in uniq_common_lbls:
            lsubs_blk = lspace[...,lsymm_lbls==lbl]
            rsubs_blk = rspace[...,rsymm_lbls==lbl]
            vecs_blk[0], svals_blk, vecs_blk[1] = matrix_svd_control_options (the_matrix,
                lsymm=None, rsymm=None, symmetry=None, strong_symm=False, lspace_symmetry=None, rspace_symmetry=None,
                lspace=lsubs_blk, rspace=rsubs_blk, only_nonzero_vals=only_nonzero_vals, full_matrices=full_matrices,
                sort_vecs=sort_vecs, num_zero_rtol=num_zero_rtol, num_zero_atol=num_zero_atol)
            nvals = len (svals_blk)
            nnull = [x.shape[1] - nvals for x in vecs_blk]
            svals.append (svals_blk)
            for i in range (2):
                vecs[i].append (vecs_blk[i][:,:nvals])
                vecs_null[i].append (vecs_blk[i][:,nvals:])
                labels[i].extend ([lbl for ix in range (nvals)])
                labels_null[i].extend ([lbl for ix in range (nnull[i])])
        svals = np.concatenate (svals)
        vecs = [np.concatenate (x, axis=1) for x in vecs]
        labels = [np.asarray (x) for x in labels]
        vecs_null = [np.concatenate (x, axis=1) for x in vecs_null]
        labels_null = [np.asarray (x) for x in labels_null]
        if sort_vecs:
            idx = svals.argsort ()[::sort_vecs]
            svals = svals[idx]
            vecs = [x[:,idx] for x in vecs]
            labels = [x[idx] for x in labels]
        if full_matrices:
            vecs = [np.append (x, y) for x, y in zip (vecs, vecs_null)]
            labels = [np.append (x, y) for x, y in zip (llabels, llabels_null)]
            def _add_symm_null (vecs, labels, null_lbls, symm_lbls, space, space_isvectorblock, Pbasis): 
                for lbl in null_lbls:
                    nnull = symm_lbls==lbl
                    if space_isvectorblock: 
                        vecs_null = space[:,symm_lbls==lbl]
                    else:
                        vecs_null = np.eye (Pbasis)[:,space][:,symm_lbls==lbl]
                    vecs = np.append (vecs, vecs_null, axis=1)
                    labels = np.append (labels, [lbl for ix in range (nnull)])
                return vecs, labels
            vecs[0], labels[0] = _add_symm_null (vecs[0], labels[0], uniq_null_lbls[0],
                lsymm_lbls, lspace, lspace_isvectorblock, Mbasis)
            vecs[1], labels[1] = _add_symm_null (vecs[1], labels[1], uniq_null_lbls[1],
                rsymm_lbls, rspace, rspace_isvectorblock, Nbasis)
        return vecs[0], svals, vecs[1], labels[0], labels[1]

    # Wrap in subspaces (Have to do both vector-block forms first)
    if lspace_isvectorblock:
        if isinstance (the_matrix, np.ndarray):
            the_matrix = lspace.conjugate ().T @ the_matrix 
        else:
            the_matrix = lspace.conjugate ().T * the_matrix
    if rspace_isvectorblock:
        if isinstance (the_matrix, np.ndarray):
            the_matrix = the_matrix @ rspace
        else:
            the_matrix = the_matrix * rspace
    elif rspace is not None:
        the_matrix = the_matrix[:,rspace]
    if not lspace_isvectorblock and lspace is not None:
        the_matrix = the_matrix[lspace,:]

    # Kernel
    lvecs, svals, r2q = scipy.linalg.svd (the_matrix, full_matrices=full_matrices)
    rvecs = r2q.conjugate ().T
    nsvals = len (svals)
    if only_nonzero_vals:
        idx = np.isclose (svals, 0, atol=num_zero_atol, rtol=num_zero_rtol)
        svals = svals[~idx]
        if full_matrices:
            lvecs[:,:nsvals] = np.append (lvecs[:,:nsvals][:,~idx], lvecs[:,:nsvals][:,idx], axis=1)
            rvecs[:,:nsvals] = np.append (rvecs[:,:nsvals][:,~idx], rvecs[:,:nsvals][:,idx], axis=1)
        else:
            lvecs = lvecs[:,~idx]
            rvecs = rvecs[:,~idx]
        nsvals = len (svals)
    if sort_vecs:
        idx = (np.abs (svals)).argsort ()[::sort_vecs]
        svals = svals[idx]
        rvecs[:,:nsvals] = rvecs[:,:nsvals][:,idx]
        lvecs[:,:nsvals] = lvecs[:,:nsvals][:,idx]

    # Align and label eigenvectors using SUBSPACE symmetry information
    lvecs, rvecs, llabels, rlabels = align_degenerate_coupled_vecs (lvecs, svals, rvecs, lspace_symmetry, rspace_symmetry, rtol=num_zero_rtol, atol=num_zero_atol)

    # Wrap out subspaces
    if lspace_isvectorblock:
        lvecs = lspace @ lvecs
    elif lspace is not None:
        subs_lvecs = lvecs.copy ()
        lvecs = np.zeros ((Mbasis, lvecs.shape[1]), dtype=lvecs.dtype)
        lvecs[lspace,:] = subs_lvecs
    if rspace_isvectorblock:
        rvecs = rspace @ rvecs
    elif rspace is not None:
        subs_rvecs = rvecs.copy ()
        rvecs = np.zeros ((Nbasis, rvecs.shape[1]), dtype=rvecs.dtype)
        rvecs[rspace,:] = subs_rvecs

    # Align and label eigenvectors using FULL SPACE symmetry information
    lvecs, rvecs, llabels, rlabels = align_degenerate_coupled_vecs (lvecs, svals, rvecs, lsymm, rsymm, rtol=num_zero_rtol, atol=num_zero_atol)

    if return_llabels and return_rlabels:
        return lvecs, svals, rvecs, llabels, rlabels
    elif return_llabels:
        return lvecs, svals, rvecs, llabels
    elif return_rlabels:
        return lvecs, svals, rvecs, rlabels
    return lvecs, svals, rvecs

def matrix_eigen_control_options (the_matrix, b_matrix=None, symmetry=None, strong_symm=False, 
    subspace=None, subspace_symmetry=None, sort_vecs=-1, only_nonzero_vals=False, round_zero_vals=False, 
    num_zero_atol=params.num_zero_atol, num_zero_rtol=params.num_zero_rtol):
    ''' Diagonalize a matrix using scipy's driver and also a whole lot of pre-/post-processing,
        most significantly sorting, throwing away numerical null spaces, 
        subspace projections, and symmetry alignments.
        
        Args:
            the_matrix: square ndarray with M rows

        Kwargs:
            b_matrix: square ndarray with M rows
                The second matrix for the generalized eigenvalue problem
            symmetry: list of block labels of length M
            or list of non-square matrices of shape (M,P), sum_P = M
            or None
                Formal symmetry information. Neither the matrix nor the
                subspace need to be symmetry adapted, and unless strong_symm=True,
                symmetry is used only to define a gauge convention within
                degenerate manifolds. Orthonormal linear combinations of degenerate
                eigenvectors with the highest possible projections onto any symmetry
                block are sequentially generated using repeated singular value decompositions
                in successively smaller subspaces. Eigenvectors within degenerate
                manifolds are therefore sorted from least to most symmetry-breaking;
                quantitatively symmetry-adapted eigenvectors are grouped by block 
                with the blocks in a currently arbitrary order.
            strong_symm: logical
                If true, the actual diagonalization is carried out symmetry-blockwise.
                Requires symmetry or subspace_symmetry.
                Eigenvectors will be symmetry-adapted but this does not
                check that the whole matrix is actually diagonalized by them so user beware!
                Extra risky if a subspace is used without subspace_symmetry
                because the vectors of the subspace
                are assigned to symmetry blocks in the same way as degenerate eigenvectors,
                and the symmetry labels of the final eigenvectors are inherited from
                the corresponding subspace symmetry block without double-checking.
                (Subspace always outranks symmetry.)
            subspace: index list for accessing Mprime elements array of shape (M,)
            or ndarray of shape (M,Mprime)
            or None
                Defines a subspace in which the matrix is diagonalized. Note
                that symmetry is applied to the matrix, not the subspace states.
                Subspace always outranks symmetry, meaning that the eigenvectors are
                guaranteed within round-off error to be contained within the
                subspace, but using a subspace may decrease the reliability
                of symmetry assignments, even if strong_symm==True.
            subspace_symmetry: list of block labels of length Mprime
            or list of non-square matrices of shape (Mprime,P), sum_P = Mprime
            or None
                Formal symmetry information of the subspace. Set to None if subspace is None.
                If not None, "symmetry" is ignored. 
            sort_vecs: integer
                Defines overall sorting of non-degenerate eigenvalues. -1 means from largest
                to smallest; +1 means from smallest to largest. Play with other values at
                your peril
            only_nonzero_vals: logical
                If true, only the K <= M nonzero eigenvalues and corresponding eigenvectors
                are returned
            round_zero_vals: logical
                If true, sets all eigenvalues of magnitude less than num_zero_atol to identically zero
            num_zero_atol: float
                Absolute tolerance for numpy's "isclose" function and its relatives.
                Used in determining what counts as a degenerate manifold.
            num_zero_rtol: float
                Relative tolerance for numpy's "isclose" function and its relatives.
                Used in determining what counts as a degenerate manifold.

        Returns:
            evals: ndarray of shape (K,); K <= M (see only_nonzero_vals kwarg)
            evecs: ndarray of shape (M,K); K <= M (see only_nonzero_vals kwarg)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning
            labels: list of length K; K<=M
                Only returned if symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each eigenvector unless
                strong_symm==True, in which case the labels are derived
                from labels of subspace vectors computed in this way.. Does not
                guarantee that the eigenvector is symmetry-adapted.
                
            

    '''

    # Interpret subspace information
    subspace = None if subspace is None else np.asarray (subspace)
    subspace_isvectorblock = False if subspace is None else subspace.ndim == 2

    # Interpret subspace symmetry information
    if strong_symm and (symmetry is None) and (subspace_symmetry is None): raise RuntimeError ("Can't do eigen by symmetry blocks if no symmetry information provided!")
    if subspace is None: subspace_symmetry = None
    return_labels = not (symmetry is None) or not (subspace_symmetry is None)
    subs_symm_isvectorblock = False if subspace_symmetry is None else isinstance (subspace_symmetry[0], np.ndarray)
    if subspace_symmetry is not None:
        symmetry = None
        if subs_symm_isvectorblock:
            if len (subspace_symmetry) == 1: subspace_symmetry = None
        elif len (np.unique (subspace_symmetry)) == 1: subspace_symmetry = None
    if subspace_symmetry is None: subs_symm_isvectorblock = False

    # Interpret (full space) symmetry information (discarded if subspace symmetry is provided!)
    symm_isvectorblock = False if symmetry is None else isinstance (symmetry[0], np.ndarray)
    if symmetry is not None and symm_isvectorblock and len (symmetry) == 1: symmetry = None
    if symmetry is not None and not symm_isvectorblock and len (np.unique (symmetry)) == 1: symmetry = None
    if symmetry is None: symm_isvectorblock = False
    if (symmetry is None) and (subspace_symmetry is None): strong_symm = False

    # Zero matrix escape
    M = subspace.shape[-1] if subspace is not None else the_matrix.shape[0]
    Mbasis = subspace.shape[0] if subspace_isvectorblock else the_matrix.shape[0]
    if not M:
        if return_labels: return np.zeros ((0)), np.zeros ((Mbasis,0)), np.ones ((0))
        return np.zeros ((0)), np.zeros ((Mbasis,0))

    # If subspace symmetry is provided as a vector block, transform subspace into a symmetry-adapted form
    # No recursion necessary because the eigenvectors are meant to be provided in the full basis :)
    if subs_symm_isvectorblock:
        if subspace_isvectorblock:
            subspace = subspace @ np.concatenate (subspace_symmetry, axis=1)
        else:
            subspace = np.concatenate (subspace_symmetry, axis=1)[subspace,:]
        subspace_symmetry = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (subspace_symmetry)])

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    if symm_isvectorblock:
        symm_umat = np.concatenate (symmetry, axis=1)
        symm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (symmetry)])
        if isinstance (the_matrix, np.ndarray):
            symm_matr = symm_umat.conjugate ().T @ the_matrix @ symm_umat
        else:
            symm_matr = (symm_umat.conjugate ().T * the_matrix) @ symm_umat
        symm_bmat = symm_umat.conjugate ().T @ b_matrix @ symm_umat if b_matrix is not None else None
        if subspace is not None:
            # Since symm_isidx == False, I have to turn the subspace into a vector block too! Dang!
            if subspace_isvectorblock: symm_subs = symm_umat.conjugate ().T @ subspace
            else: symm_subs = symm_umat.conjugate ().T [:,subspace]
        # Be 200% sure that this recursion can't trigger this conditional block again!
        assert (not (isinstance (symm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
        evals, symm_evecs, labels = matrix_eigen_control_options (symm_matr, symmetry=symm_lbls, strong_symm=strong_symm,
            subspace=symm_subs, sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals, round_zero_vals=round_zero_vals,
            b_matrix=symm_bmat, num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        evecs = symm_umat @ symm_evecs
        return evals, evecs, labels

    # Recurse from strong symmetry enforcement to diagonalization over individual symmetry blocks
    if strong_symm:

        # If a subspace is being diagonalized, recurse into symmetry blocks via the subspace
        if subspace is None:
            symm_lbls = symmetry
            subspace = np.ones (the_matrix.shape[0], dtype=np.bool_)
        elif subspace_symmetry is not None:
            symm_lbls = subspace_symmetry
        elif subspace_isvectorblock:
            subspace, symm_lbls = align_vecs (subspace, symmetry, rtol=num_zero_rtol, atol=num_zero_atol)
        else:
            symm_lbls = symmetry[subspace]
            
        # Recurse into diagonalization of individual symmetry blocks via the "subspace" option!
        uniq_lbls = np.unique (symm_lbls)
        evals = []
        evecs = []
        labels = []
        for lbl in uniq_lbls:
            subs_blk = subspace[...,symm_lbls==lbl]
            evals_blk, evecs_blk = matrix_eigen_control_options (the_matrix, b_matrix=b_blk, subspace=subs_blk,
                symmetry=None, subspace_symmetry=None, strong_symm=False,
                sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals, round_zero_vals=round_zero_vals,
                num_zero_rtol=num_zero_rtol, num_zero_atol=num_zero_atol)
            evals.append (evals_blk)
            evecs.append (evecs_blk)
            labels.extend ([lbl for ix in range (len (evals_blk))])
        evals = np.concatenate (evals)
        evecs = np.concatenate (evecs, axis=1)
        labels = np.asarray (labels)
        if sort_vecs:
            idx = evals.argsort ()[::sort_vecs]
            evals = evals[idx]
            evecs = evecs[:,idx]
            labels = labels[idx]
        return evals, evecs, labels

    # Wrap in subspace projection. This should be triggered if and only if strong_symm==False!
    if subspace is not None:
        assert (not strong_symm)
        if subspace_isvectorblock:
            if isinstance (the_matrix, np.ndarray):
                the_matrix = subspace.conjugate ().T @ the_matrix @ subspace
            else:
                the_matrix = (subspace.conjugate ().T * the_matrix) @ subspace
            b_matrix   = subspace.conjugate ().T @ b_matrix @ subspace if b_matrix is not None else None
        else:
            idx = np.ix_(subspace,subspace)
            ndim_full = the_matrix.shape[0]
            the_matrix = the_matrix[idx]
            b_matrix   = b_matrix[idx] if b_matrix is not None else None

    # Now for the actual damn kernel            
    # Subtract a diagonal average from the matrix to fight rounding error
    diag_avg = np.eye (the_matrix.shape[0]) * np.mean (np.diag (the_matrix))
    pMq = np.asmatrix (the_matrix - diag_avg)
    qSr = None if b_matrix is None else np.asmatrix (b_matrix)
    # Use hermitian diagonalizer if possible and don't do anything if the matrix is already diagonal
    evals = np.diagonal (pMq)
    evecs = np.asmatrix (np.eye (len (evals), dtype=evals.dtype))
    if not is_matrix_diagonal (pMq):
        evals, evecs = scipy.linalg.eigh (pMq, qSr) if is_matrix_hermitian (pMq) else scipy.linalg.eig (pMq, qSr)
    # Add the diagonal average to the eigenvalues when returning!
    evals = evals + np.diag (diag_avg)
    if only_nonzero_vals:
        idx = np.where (np.abs (evals) > num_zero_atol)[0]
        evals = evals[idx]
        evecs = evecs[:,idx]
    if sort_vecs:
        idx = evals.argsort ()[::sort_vecs]
        evals = evals[idx]
        evecs = evecs[:,idx]
    if round_zero_vals:
        idx = np.where (np.abs (evals) < num_zero_atol)[0]
        evals[idx] = 0
    evals, evecs = (np.asarray (output) for output in (evals, evecs))

    # Align and label eigenvectors using SUBSPACE symmetry information
    evecs, labels = align_degenerate_vecs (evals, evecs, subspace_symmetry, rtol=num_zero_rtol, atol=num_zero_atol)

    if subspace is not None:
        assert (not strong_symm)
        if subspace_isvectorblock:
            evecs = subspace @ evecs
        else:
            subs_evecs = evecs.copy ()
            evecs = np.zeros ((ndim_full, subs_evecs.shape[1]), dtype=subs_evecs.dtype)
            evecs[subspace,:] = subs_evecs
    
    # Align and label eigenvectors using FULL SPACE symmetry information
    if labels is None: evecs, labels = align_degenerate_vecs (evals, evecs, symmetry, rtol=num_zero_rtol, atol=num_zero_atol)

    # Dummy labels if no symmetry provided
    if labels is None: labels = np.zeros (len (evals))

    if return_labels: return evals, evecs, labels
    return evals, evecs

def align_degenerate_vecs (vals, vecs, symm, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    if symm is None:
        return vecs, None
    uniq_labels = np.unique (symm)
    idx_unchk = np.ones (len (vals), dtype=np.bool_)
    labels = np.empty (len (vals), dtype=uniq_labels.dtype)
    while np.count_nonzero (idx_unchk):
        chk_1st_eval = vals[idx_unchk][0]
        idx_degen = np.isclose (vals, chk_1st_eval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx_degen) > 1:
            vecs[:,idx_degen], labels[idx_degen] = align_vecs (vecs[:,idx_degen], symm, atol=atol, rtol=rtol)
        else:
            symmweight = [np.square (vecs[symm==lbl,idx_degen]).sum () for lbl in uniq_labels]
            labels[idx_degen] = uniq_labels[np.argmax (symmweight)]
        idx_unchk[idx_degen] = False
    return vecs, labels

def align_vecs (vecs, row_labels, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    col_labels = np.empty (vecs.shape[1], dtype=row_labels.dtype)
    uniq_labels = np.unique (row_labels)
    i = 0
    while i < vecs.shape[1]:
        svdout = [matrix_svd_control_options (1, lspace=(row_labels==lbl), rspace=vecs[:,i:],
            sort_vecs=-1, only_nonzero_vals=False, full_matrices=True) for lbl in uniq_labels]
        # This argmax identifies the single best irrep assignment possible for all of vecs[:,i:]
        symm_label_idx = np.argmax ([svals[0] for lvecs, svals, rvecs in svdout])
        lvecs, svals, rvecs = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose (svals, svals[0], atol=atol, rtol=rtol))
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        # This puts the best-aligned vector at position i and causes all of vecs[:,j:] to be orthogonal to it
        vecs[:,i:] = rvecs
        assert (j > i)
        i = j
        # This is a trick to grab a whole bunch of degenerate vectors at once (i.e., numerically symmetry-adapted vectors with sval = 1)
        # It may improve numerical stability
    return vecs, col_labels

def align_degenerate_coupled_vecs (lvecs, svals, rvecs, lsymm, rsymm, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    nvals = len (svals)
    if lsymm is None and rsymm is None:
        return lvecs, rvecs, None, None
    if lsymm is None: lsymm = []
    if rsymm is None: rsymm = []
    uniq_labels = np.unique (np.append (lsymm, rsymm))
    idx_unchk = np.ones (len (svals), dtype=np.bool_)
    llabels = np.empty (lvecs.shape[1], dtype=uniq_labels.dtype)
    rlabels = np.empty (rvecs.shape[1], dtype=uniq_labels.dtype)
    lv = lvecs[:,:nvals]
    rv = rvecs[:,:nvals]
    ll = llabels[:nvals]
    while np.count_nonzero (idx_unchk):
        chk_1st_sval = svals[idx_unchk][0]
        idx = np.isclose (svals, chk_1st_sval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx) > 1:
            lv[:,idx], rv[:,idx], ll[idx] = align_coupled_vecs (
                lv[:,idx], rv[:,idx], lsymm, rsymm, rtol=rtol, atol=atol)
        else:
            proj = lv[:,idx] @ rv[:,idx].conjugate ().T
            symmweight = []
            for lbl in uniq_labels:
                if len (lsymm) > 0: proj = proj[lsymm==lbl,:]
                if len (rsymm) > 0: proj = proj[:,rsymm==lbl]
                symmweight.append (linalg.norm (proj))
            ll[idx] = uniq_labels[np.argmax (symmweight)]
        idx_unchk[idx_degen] = False
    rlabels = llabels
    if lvecs.shape[1] > nvals and len (lsymm) > 0:
        lvecs[:,nvals:], llabels[nvals:] = align_vecs (lvecs[:,nvals:], lsymm, rtol=rtol, atol=atol)
    if rvecs.shape[1] > nvals and len (rsymm) > 0:
        rvecs[:,nvals:], rlabels[nvals:] = align_vecs (rvecs[:,nvals:], rsymm, rtol=rtol, atol=atol)
    if len (lsymm) == 0: llabels = None
    if len (rsymm) == 0: rlabels = None
    return lvecs, rvecs, llabels, rlabels

def align_coupled_vecs (lvecs, rvecs, lrow_labels, rrow_labels, rtol=params.num_zero_rtol, atol=params.num_zero_atol, assert_tol=0):
    assert (lvecs.shape[1] == rvecs.shape[1])
    npairs = lvecs.shape[1]
    col_labels = np.empty (npairs, dtype=rrow_labels.dtype)
    if lrow_labels is None: lrow_labels = []
    if rrow_labels is None: rrow_labels = []
    uniq_labels = np.unique (np.concatenate (lrow_labels, rrow_labels))
    i = 0
    coupl = lvecs @ rvecs.conjugate ().T
    while i < npairs:
        svdout = []
        for lbl in uniq_labels:
            metric = coupl.copy ()
            if len (lrow_labels) > 0: metric[lrow_labels!=lbl,:] = 0
            if len (rrow_labels) > 0: metric[:,rrow_labels!=lbl] = 0
            svdout.append (matrix_svd_control_options (metric, sort_vecs=-1, only_nonzero_vals=False, full_matrices=True,
                lspace=lvecs[:,i:], rspace=rvecs[:,i:]))
        symm_label_idx = np.argmax ([svals[0] for lu, svals, ru in svdout])
        lvecs[:,i:], svals, rvecs[:,i:] = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose (svals, svals[0], atol=atol, rtol=rtol))
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        assert (j > i)
        i = j
    return lvecs, rvecs, col_labels

def assign_blocks_weakly (the_states, the_blocks):
    projectors = [blk @ blk.conjugate ().T for blk in the_blocks]
    vals = np.stack ([((proj @ the_states) * the_states).sum (0) for proj in projectors], axis=-1)
    return np.argmax (vals, axis=1)





