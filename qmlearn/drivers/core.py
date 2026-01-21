import numpy as np
import scipy
from sklearn.decomposition import PCA
from rmsd.calculate_rmsd import (
    kabsch,
    quaternion_rotate,
    reorder_brute,
    reorder_distance,
    reorder_hungarian,
    reorder_inertia_hungarian,
    )

from qmlearn.utils.utils import matrix_deviation, unitary_decompose, wedge
from qmlearn.data import REFLECTION
from scipy.optimize import linear_sum_assignment

class Engine(object):
    r"""Abstract Base class for the External calculator.
    Until now we have implemented: PySCF and Psi4.

    Attributes
    ----------

    vext : ndarray
        External Potential.
    gamma : ndarray
        1-body reduced density matrix (1-RDM).
    gamma2 : ndarray
        2-body reduced density matrix (2-RDM).
    etotal : float
        Total electronic energy
    forces : ndarray
        Atomic forces
    ovlp : ndarray
        Overlap Matrix.
    kop : ndarray
        Kinetic Energy Operator.
    eri : ndarray
        2-electron integrals: 8-fold or 4-fold ERIs or complex integral array with N^4 elements
        (N is the number of orbitals).
    orb : ndarray
        Molecular orbital coefficients
    """
    def __init__(self, mol = None, method = 'rks', basis = '6-31g', xc = None, **kwargs):
        self.options = locals()
        self.options.update(kwargs)
        self.options.pop('self', None)
        self.options.pop('kwargs', None)
        #-----------------------------------------------------------------------
        self._vext = None
        self._gamma = None
        self._tgamma = None
        self._gamma2 = None
        self._gamma2c = None
        self._etotal = None
        self._etotal_c = None
        self._forces = None
        self._occ = None
        self._mo_energy = None
        self._eig_gamma2 = None
        self._eig_gamma2c = None
        self._delta_gamma = None
        self._gamma2cum = None
        self._gamma_00 = None
        self._gamma_01 = None
        self._gamma_10 = None
        self._gamma_11 = None
        self._occ_dg = None
        #
        self._kop = None
        self._ovlp = None
        self._eri = None
        self._orb = None
        self._ovlp_x = None
        self._ovlp_x_inv = None

    def init(self, *args, **kwargs):
        r""" Initialize ABC class."""
        pass

    def run(self, *args, **kwargs):
        r""" ABC to check run function."""
        pass

    @property
    def gamma(self):
        r""" 1-body reduced density matrix (1-RDM). """
        pass

    @property
    def gamma2(self):
        r""" 2-body reduced density matrix (2-RDM). """
        pass

    @property
    def tgamma(self):
        r""" 1-body transition density matrix. """
        pass

    @property
    def etotal(self):
        r""" Total Energy. """
        pass
    @property
    def etotal_c(self):
        r"""Correlated Energy. """
        pass

    @property
    def forces(self):
        r""" Atomic Forces.  """
        pass

    @property
    def vext(self):
        r""" External Potential. """
        pass

    @property
    def kop(self):
        r""" Kinetic Energy Operator. """
        pass

    @property
    def ovlp(self):
        r"""Overlap Matrix."""
        pass

    @property
    def ncharge0(self):
        r""" Calculated number of electrons. """
        pass

    @property
    def ovlp_x(self):
        if self._ovlp_x is None :
            self.init_ovlp_x()
        return self._ovlp_x

    @property
    def ovlp_x_inv(self):
        if self._ovlp_x_inv is None :
            self.init_ovlp_x()
        return self._ovlp_x_inv

    def calc_gamma(self, orb=None, occs=None):
        r"""Calculate the 1-body reduced density matrix (1-RDM).

        Parameters
        ----------
        orb : ndarray
           Orbital Coefficients. Each column is one orbital
        occs : ndarray
            Occupancy

        Returns
        -------
        1-RDM : ndarray
        """
        pass

    def calc_ncharge(self, gamma, ovlp = None):
        r""" Get calculated total number of electrons

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        ovlp : ndarray
            Overlap Matrix

        Returns
        -------
        ncharge : int
            Calcualted number of electrons

        """
        if ovlp is None : ovlp = self.ovlp
        ncharge = np.einsum('ji,ij->', gamma, ovlp)
        return ncharge

    def calc_ke(self, gamma, kop = None):
        r""" Get Total kinetic energy

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        kop : ndarray
            Kinetic energy operator

        Returns
        -------
        ke : float
            Total kinetic energy

        """
        if kop is None : kop = self.kop
        ke = np.einsum('ji,ij->', gamma, kop)
        return ke

    def calc_idempotency(self, gamma, ovlp=None, kind=1):
        r""" Check idempotency of gamma

        .. math::
           \gamma S \gamma = 2 \gamma

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        ovlp : ndarray
            Overlap matrix

        Attributes
        ----------
        kind :  int

            | 1 : Level 1
            | 2 : Level 2
            | 3 : Level 3

        Returns
        -------
        errorsum : float
            Sum over the Absolute difference among matrix 1 and 2.

        """
        # Only for nspin=1
        if ovlp is None : ovlp = self.ovlp
        if kind==1:
            g = gamma@ovlp@gamma/2
        if kind==2:
            g = (3*gamma@ovlp@gamma@ovlp@gamma - 2*gamma@ovlp@gamma)/8
        if kind==3:
            g = (4*gamma@ovlp@gamma@ovlp@gamma -gamma@ovlp@gamma@ovlp@gamma@ovlp@gamma)/8
        return matrix_deviation(gamma, g)

    def calc_occupations(self, gamma, ovlp_x=None, ovlp=None, type=1):
        r"""Calculate occupations for gamma

        Parameters
        ----------
        gamma: ndarray
           1-RDM
        type: int
            It specifies the problem type to be solved::

                1 => x@a@x@v = w@v => a = x^{-1}@(w@v@v.T)@x^{-1}
                2 =>   a@p@v = w@v => a = w@v@v.T

        Returns
        -------
        occs: ndarray
            Occupancy, all but last
        orbs: ndarray
            Orbital coefficients, all but last. Each column is one orbital
        """

        if type==1:
            if ovlp_x is None or ovlp_x.size == 0:
                ovlp_x = self.ovlp_x
            occs, orbs = np.linalg.eigh(ovlp_x@gamma@ovlp_x)
        else:
            if ovlp is None : ovlp = self.ovlp
            occs, orbs = scipy.linalg.eigh(gamma, b=ovlp, type=2)
        return occs[::-1], orbs[:,::-1]

    def init_ovlp_x(self, ovlp = None):
        r""" Initiate overlap calculation

        Parameters
        ----------
        ovlp: ndarray
           Overlap matrix

        Returns
        -------
        einsum: float
           Sum of eigenvalues for overlap and inverted overlap matrices
        """

        ovlp = ovlp or self.ovlp
        svel, svec = np.linalg.eigh(ovlp)
        self._ovlp_x = np.einsum('ik,jk->ij', svec, svec*np.sqrt(svel))
        self._ovlp_x_inv = np.einsum('ik,jk->ij', svec, svec/np.sqrt(svel))

    def purify_gamma(self, gamma, occs=None, nelectron=None, method='aufbau', smearing='fermi', sigma=0.1, type=1, **kwargs):
        """purify_gamma.

        Parameters
        ----------
        gamma : ndarray
            1-body reduced density matrix (1-RDM).
        occs : ndarray
            Occupation numbers
        nelectron : int
            Total number of electrons
        method : str
            method for purification, options are

                | aufbau : Aufbau
                | smearing : Fermi-Dirac distribution
        """
        nelectron = nelectron or self.nelectron
        ovlp_x_inv = self.ovlp_x_inv
        occs_g, orbs = self.calc_occupations(gamma, type=type)
        #print(occs_g)
        if occs is not None:
            occs_i = occs
        elif method == 'aufbau':
            occs = occs_g/2.0
            occs_i = np.rint(occs)*2.0
            occs_i[occs_i<0.0] = 0.0
            occs_i[occs_i>2.0] = 2.0
        elif method == 'smearing':
            mo_energy = self.calc_delta_mo_energy(occs_g, sigma=sigma)
            fermi, occs_i = self.get_occupations(mo_energy, smearing=smearing, nelectron=nelectron, sigma=sigma, **kwargs)
        elif method == 'delta':
            occs_i = occs_g
            occs_i[occs_i<0.0] = 0.0
        else:
            raise AttributeError("Please give occupations or a supported method.")
        if method != 'delta':
          if abs(occs_i.sum() - nelectron) > 1E-6:
              print('Number of e: ', occs_i.sum(),nelectron)
              raise ValueError('The occupations does not match the number of electrons')
        #print(occs_i)
        if type == 1:
            gamma = ovlp_x_inv@np.einsum('ik,jk->ij', orbs, orbs*occs_i)@ovlp_x_inv
        else:
            gamma = np.einsum('ik,jk->ij', orbs*occs_i, orbs)
        return gamma

    @staticmethod
    def calc_delta_mo_energy(occs, sigma=0.1):
        mask = occs <= 0
        mask2 = occs >= 2
        occs[mask] = 1.0
        occs[mask2] = 1.0
        mo_energy = np.log(2/occs -1.0)
        mo_energy *= sigma
        mo_energy[mask] = 1E10
        mo_energy[mask2] = -1E10
        return mo_energy

    def get_occupations(self, mo_energy, smearing='fermi', nelectron=None, fermi=None, sigma=0.1):
        nelectron = nelectron or self.nelectron
        nocc = nelectron/2
        if fermi is None: fermi = mo_energy[max(0, int(nocc-1))]
        if hasattr(smearing, '__call__'):
            fs = smearing
        elif smearing=='fermi':
            def fs(mu, mo_energy, sigma):
                occ = np.zeros_like(mo_energy)
                de = (mo_energy - mu) / sigma
                occ[de<40] = 1.0/(np.exp(de[de<40])+1.0)
                return occ
        else:
            raise AttributeError(f"{smearing} not supported")

        def func(mu, mo_energy, sigma, nocc):
            ne = fs(mu, mo_energy, sigma).sum()
            return (ne-nocc)**2

        res = scipy.optimize.minimize(func, fermi, args=(mo_energy, sigma, nocc), method='Powell',
                options={'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 10000})
        fermi = res.x
        mo_occs = fs(fermi, mo_energy, sigma)*2.0
        return fermi, mo_occs

    def purify_gamma2(self, gamma2=None, occs=None, gamma2c=None):
        if gamma2 is None: gamma2 = gamma2c
        eigv_, coeff = self.eigs_gamma2(gamma2)
        eigv = occs
        gamma2_ = np.einsum('ik,jk->ij', coeff, coeff*eigv)
        shape = np.shape(gamma2)
        gamma2_p = np.transpose(gamma2_.reshape(shape),[0, 2, 1, 3])

        return gamma2_p

    def update_gamma2(self, gamma2, gamma=None, trace=None, ao_repr=True, nelectron=None, identity=None):
        nelectron = nelectron or self.nelectron
        if trace is None:
            trace = nelectron * (nelectron-1)
        nelectron = nelectron or self.nelectron
        if gamma is not None: gamma = gamma * (nelectron-1)
        if ao_repr :
            if gamma is not None :
                gamma = self.gamma_ao2mo(gamma)
            gamma2 = self.gamma2_ao2mo(gamma2)

        A0, A1, A2 = unitary_decompose(gamma2)
        A0m, A1m, A2m = unitary_decompose(gamma2, a = gamma, trace=trace, identity=identity)
        gamma2_m = A0m + A1m + A2
        gamma2_m = self.gamma2_mo2ao(gamma2_m)
        return gamma2_m

    def eigs_gamma2(self, gamma2, ao_repr=True):
        from scipy import linalg
        if ao_repr:
            shape = np.shape(gamma2)[0]
            gamma2_reshape = np.transpose(gamma2,[0,2,1,3]).reshape((shape**2,shape**2))
            ovlp_aug = np.einsum('vu,st->vust',self.ovlp,self.ovlp)
            ovlp_aug_reshape = np.transpose(ovlp_aug,[0,2,1,3]).reshape((shape**2,shape**2))

            eigv, coeff = linalg.eigh(gamma2_reshape,ovlp_aug_reshape,type=2)
            trace_gamma2 = np.einsum('mnst,mn,st',gamma2,self.ovlp,self.ovlp)
        else:
            shape = np.shape(gamma2)[0]
            gamma2_reshape = gamma2.reshape((shape**2,shape**2))
            eigv, coeff = linalg.eigh(gamma2_reshape)
            trace_gamma2 = np.einsum('mnmn->',gamma2)
         
        if np.any(np.round(eigv,3)<0): #round 3 round e-4
            print('2RDM is NOT positive semidefinite')
        else:
            print('2RDM is positive semidefinite')
        if not np.allclose(trace_gamma2, 0, atol=1e-3):
            print('2RDM trace is not ZERO -> N', trace_gamma2,'!=', 0)
        else:
            print('2RDM trace is ZERO')

        return eigv[::-1], coeff[:,::-1]

    def eigs_gamma1(self,gamma1):
        ovlp = self.ovlp
        from scipy import linalg
        eigv, coeff = linalg.eigh(gamma1,ovlp,type=2)
        if np.any(np.round(eigv,6)<0):
          print('1RDM is NOT positive semidefinite')
        else:
          print('1RDM is positive semidefinite')

        trace_gamma1 = np.trace(gamma1@ovlp)
        if not np.allclose(trace_gamma1, 0, atol=1e-3):
           print('1RDM trace is not ZERO -> N', trace_gamma1,'!=', 0)
        else:
           print('1RDM trace is ZERO')
        return eigv[::-1], coeff[:,::-1]
        
    def purify_d_gamma(self, gamma_d=None, gamma_hf=None,pure=True,method='smearing'):

        r""" Function to Purify \delta gamma1 = gamma^1_{FCI} - gamma^1_{HF}

        Returns:
        \delta gamma1 and gamma^1_{FCI}

        """
        if gamma_hf is None:
           self.mf.run()
           gamma_hf = self.mf.make_rdm1()
        #ove = self.ovlp
        
        gamma_full = gamma_hf + gamma_d
        if pure:
          gamma_f_p = self.purify_gamma(gamma_full,method=method)
        else:
          gamma_f_p = gamma_full

        gamma_d = gamma_f_p - gamma_hf

        return gamma_f_p, gamma_d

    def purify_gamma2c(self,gamma=None,gamma2c=None,gamma_hf=None,pure=True):
        if gamma_hf is None:
           self.mf.run()
           gamma_hf = self.mf.make_rdm1()

        a = np.einsum('pq,rs->pqrs',gamma_hf,gamma_hf,optimize=True)
        b = np.einsum('pq,rs->psrq',gamma_hf,gamma_hf,optimize=True)
        gamma2_r = gamma2c + .5*(2*a-b)

        if pure:
          trace = self.nelectron * (self.nelectron-1)
          gamma2_f_p = self.update_gamma2(gamma2_r, gamma=gamma, trace = trace)
        else:
          gamma2_f_p = gamma2_r

        gamma2c_p = gamma2_f_p - .5*(2*a-b)

        return gamma2_f_p, gamma2c_p

    def purify_gamma2p(self,gamma=None,gamma2c=None,pure=True):

        if pure:
          trace = 0
          gamma2_f_p = self.update_gamma2(gamma2c, gamma=gamma, trace = trace)
        else:
          gamma2_f_p = gamma2_r

        return gamma2_f_p


    def gamma1_f_gamma2(self,gamma2, gamma1=None):
        gamma_f = np.einsum('mnst,st->mn',gamma2,self.ovlp,optimize=True)/(self.nelectron-1)
        if gamma1 is not None:
          print('Check gamma1 from gamma2 == given gamma1: ', np.allclose(gamma1,gamma_f))
        return gamma_f

    def particle_hole_duality(self,gamma1=None,gamma2=None,r=None,identity=None, ao_repr=True,
                              a=None,type_rdm='q'):
        if ao_repr :
            if gamma1 is not None :
                gamma1 = self.gamma_ao2mo(gamma1)
            gamma2 = self.gamma2_ao2mo(gamma2)

        gamma2_t = gamma2.transpose([0,2,1,3])

        if r is None: r = gamma2.shape[0]
        if identity is None: identity = np.eye(r)
        if a is None: a = np.einsum('ikjk->ij', gamma2_t, optimize=True)/(self.nelectron-1)
        N = self.nelectron
        n_d = N*(N-1)
        n_q = (2*r-N)*(2*r-N-1) #for close-shell
        n_g = N*(2*r-N+1)  #for close-shell
        delta = identity
        print('n_d: ',n_d,'\n','n_q: ',n_q,'\n','n_g: ',n_g)

        if type_rdm == 'g':
           g = np.einsum('jl,ik->ijkl',delta,a)*2 - gamma2_t.transpose([0,3,2,1])
           mat_f = g                                           
         
        if type_rdm == 'q':
           d_jl_d_ik = 4*np.einsum('jl,ik->ijkl',delta,delta)
           d_il_d_jk = 2*np.einsum('il,jk->ijkl',delta,delta)
           d_jl_g_ik = 2*np.einsum('jl,ik->ijkl',delta, a)
           d_il_g_jk = np.einsum('il,jk->ijkl',delta, a)
           d_jk_g_il = np.einsum('jk,il->ijkl',delta,a)
           d_ik_g_jl = 2*np.einsum('ik,jl->ijkl',delta,a)
           q = gamma2_t + d_jl_d_ik - d_il_d_jk - d_jl_g_ik + d_il_g_jk + d_jk_g_il - d_ik_g_jl
           mat_f = q
        return mat_f

    def particle_df(self,gamma1,gamma2, type_rdm ='q'):
        if type_rdm =='q':
          q = self.particle_hole_duality(gamma1 = gamma1,gamma2 = gamma2, type_rdm='q')
          results = q
        else:
          g = self.particle_hole_duality(gamma1 = gamma1,gamma2 = gamma2, type_rdm='g')
          results = g
        return results

def atoms_rmsd(target, atoms, transform = True, **kwargs) :
    r""" Function to return RMSD : Root mean square deviation between atoms and target:transform atom object. And the target atom coordinates.

    Parameters
    ----------
    target : :obj: ASE atoms object
        Reference atoms.
    atoms : :obj: ASE atoms object
        Atoms to be transformed (Rotate y/o translate).

    Attributes
    ----------
    keep : bool
        If True keep a copy of atoms.
    transform : bool
        If True minimize rotation and translation between refatoms and atoms.
        (See ASE documentation: Minimize RMSD between atoms and target.)

    Returns
    -------
    rmsd : float
        RMSD betwwen reference and transform atoms.
    atoms : :obj: ASE atoms object
        Transformed target molecule.
    """
    if transform :
        op_rotate, op_translate, op_indices = minimize_rmsd_operation(target, atoms, **kwargs)
        positions = np.dot(atoms.positions,op_rotate)+op_translate
        atoms = atoms[op_indices]
        atoms.set_positions(positions[op_indices])
        rmsd = rmsd_coords(target.positions, atoms.positions)
    else :
        rmsd = rmsd_coords(target.positions, atoms.positions)
    return rmsd, atoms

def rmsd_coords(target, pos, **kwargs):
    r""" Function to return RMSD : Root mean square deviation between pos and target atomic positions.

    Parameters
    ----------
    pos : :obj: ASE atoms object
        Reference atoms.
    target : :obj: ASE atoms object
        Atoms to be transformed (Rotate y/o translate).

    Returns
    -------
    rmsd : float
        RMSD between pos and target atomic positions  """
    return diff_coords(target, pos, diff_method='rmsd', **kwargs)

def diff_coords(target, pos = None, weights = None, diff_method = 'rmsd'):
    r""" Function to return mean base on method error/deviation between pos and target atomic positions.

    Parameters
    ----------
    pos : :obj: ASE atoms object
        Reference atoms.
    target : :obj: ASE atoms object
        Atoms to be transformed (Rotate y/o translate).

    Attributes
    ----------
    diff_method : str

        | RMSD(root-mean-square deviation) : 'rsmd'
        | RMSE(root-mean-square error) : 'rmse'
        | MSD(mean-square deviation) : 'msd'
        | MSE(mean-square error) : 'mse'
        | MAE(mean absolute error) : 'mae'

    Returns
    -------
    rmsd : float
        Mean base on method error/deviation between pos and target atomic positions """
    if pos is None :
        diff = target
    else :
        diff = pos - target
    if weights is not None :
        weights = np.asarray(weights)
        if weights.ndim == 1 and len(weights) > 1 :
            weights = weights[:, None]
        diff *= weights
    if diff_method in ['msd', 'mse'] : # mean squared deviation (MSD) or mean squared error (MSE)
        rmsd = np.sum(diff*diff)/len(diff)
    elif diff_method in ['rmsd', 'rmse'] : # root-MSD or root-MSE
        rmsd = np.sqrt(np.sum(diff*diff)/len(diff))
    elif diff_method == 'mae' : # mean absolute error (MAE)
        rmsd = np.sum(np.abs(diff))/len(diff)
    return rmsd

def atoms2bestplane(atoms, direction = None):
    r""" Apply Principal Component Analysis (PCA) to atoms and re-oriente them.

    Parameters
    ----------
    atoms : :obj: ASE atoms object
       Atoms coordinates

    Attributes
    ----------
    direction : ndarray
        1D-vector to re-orient the atoms positions

    Returns
    -------
    atoms : ndarray
        Reoriented atom positions
    """
    pca = PCA()
    pos = pca.fit_transform(atoms.positions)
    atoms.set_positions(pos)
    if direction is not None :
        atoms = atoms2newdirection(atoms, a=(0,0,1), b=direction)
    return atoms

def get_atoms_axes(atoms):
    r""" Get PCA components of atomic positions.

    Parameters
    ----------
    atoms : :obj: ASE object

    Returns
    -------
    axes : ndarry
        Vector containing the PCA components of the atomic positions """
    pca = PCA(n_components=3)
    pca.fit(atoms.positions)
    axes = pca.components_
    return axes

def atoms2newdirection(atoms, a=(0,0,1), b=(1,0,0)):
    r""" Function to re-orientate the atom positions.
    If Vector is None, the default rotation is around X-axis.

    Parameters
    ----------
    atoms : :obj: ASE atom object

    Returns
    -------
    atoms : ndarray
        Rotated atoms coordinates
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    if np.allclose(a, b) : return atoms
    c = np.cross(a, b)
    deg= np.rad2deg(np.arccos(np.clip(np.dot(a,b),-1.0,1.0)))
    atoms.rotate(deg,c)
    return atoms

def minimize_rmsd_operation(target, atoms, stereo = True, rotate_method = 'kabsch',
        reorder_method = 'inertia-hungarian', use_reflection = True, rmsd_cut = None, ignore_hydrogen = False):
    r""" Function to create Rotation Matrix and Translation Vector
    of reference atoms with respect to initialize atoms.

    Parameters
    ----------
    target : :obj: ASE atoms object
        References atoms
    atoms : :obj: ASE atoms object
        Initialize atoms
    reorder_method : str

        | 'hungarian'
        | 'inertia-hungarian'
        | 'brute'
        | 'distance'

    rotate_method: str

        | None : 'none'
        | Kabsch : 'kabsch'
        | Quaternion : 'quaternion'

    use_reflection : bool
        If True it applies a reflection on your molecule.

    ignore_hydrogen : bool
        If True will ignore the hydrogen atoms, and the reorder_method will not working.

    Returns
    -------
    rotate : ndarray
        Rotation Matrix
    translate : ndarray
        Translation Vector
    rmsd_final_indices : ndarry
        Reorderred atom indices
    """
    # return _minimize_rmsd_operation_v0(target, atoms)
    #
    if ignore_hydrogen :
        atoms1 = target[~(target.symbols == 'H')]
        atoms2 = atoms[~(atoms.symbols == 'H')]
        reorder_method = 'none'
    else :
        atoms1 = target.copy()
        atoms2 = atoms.copy()
    symbols = atoms2.get_chemical_symbols()
    #
    pos_t = atoms1.get_positions()
    c_t = np.mean(pos_t, axis=0)
    pos_t = pos_t - c_t

    pos = atoms2.get_positions()
    c = np.mean(pos, axis=0)
    pos = pos - c

    #print('Initial positions: (Target) (Atoms) ', pos_t,pos)
    #print('Initial conditions (Reflection, Rotation): ', use_reflection, rotate_method)
    #-----------------------------------------------------------------------
    atoms1.positions[:] = pos_t
    #-----------------------------------------------------------------------
    if use_reflection :
        if stereo :
            srot = REFLECTION.srot_stereo
        else :
            srot = REFLECTION.srot
    else :
        srot=[np.eye(3)]
    #-----------------------------------------------------------------------
    rmsd_final_min = np.inf
    for ia, rot in enumerate(srot):
        atoms2.set_positions(np.dot(pos, rot))
        atoms2.set_chemical_symbols(symbols)
        #
        indices = reorder_atoms_indices(atoms1, atoms2, reorder_method=reorder_method)
        atoms2 = atoms2[indices]
        rotate = get_match_rotate(atoms1, atoms2, rotate_method = rotate_method)
        atoms2.positions[:] = np.dot(atoms2.positions[:], rotate)
        rmsd = diff_coords(atoms1.positions, atoms2.positions)
        print('indices: ', indices, rmsd, 'rotated_method: ', rotate_method)
        if rmsd < rmsd_final_min :
            rmsd_final_min = rmsd
            rmsd_final_rotate = rotate
            rmsd_final_reflection = rot
            rmsd_final_indices = indices
        if rmsd_cut is not None :
            if rmsd < rmsd_cut : break
    rotate = np.dot(rmsd_final_reflection, rmsd_final_rotate)
    translate = c_t - np.dot(c, rotate)
    print('rmsd_final_min', rmsd_final_min)
    if ignore_hydrogen : rmsd_final_indices = np.arange(len(atoms))
    return rotate, translate, rmsd_final_indices

def reflect_atoms(atoms, stereo = True, tol=1E-8, **kwargs):
    r""" Function to create all reflection atoms"""
    pos_t = atoms.get_positions()
    c_t = np.mean(pos_t, axis=0)
    pos_t = pos_t - c_t
    #-----------------------------------------------------------------------
    atoms0 = atoms.copy()
    atoms0.positions[:] = pos_t
    #-----------------------------------------------------------------------
    if stereo :
        srot = REFLECTION.srot_stereo
    else :
        srot = REFLECTION.srot
    data = []
    for ia, rot in enumerate(srot):
        atoms1 = atoms0.copy()
        atoms1.positions[:] = np.dot(atoms1.positions, rot)
        rmsd = diff_coords(atoms0.positions, atoms1.positions, diff_method = 'mae')
        if rmsd > tol :
            data.append(atoms1)
    return data

def get_match_rotate(target, atoms, rotate_method = 'kabsch'):
    r""" Rotate Atoms with a customize method

    Parameters
    ----------
    target : :obj: ASE atoms object
        References atoms
    atoms : :obj: ASE atoms object
        Initialize atoms
    rotate_method: str

        | None : 'none'
        | Kabsch : 'kabsch'
        | Quaternion : 'quaternion'

    Returns
    -------
    rotate : ndarray
        Rotated atom positions """
    if rotate_method is None or rotate_method == 'none':
        rotate = np.eye(3)
    else :
        if not hasattr(rotate_method, '__call__'):
            if rotate_method == 'kabsch' :
                rotate_method = kabsch
            elif rotate_method == 'quaternion' :
                rotate_method = quaternion_rotate
                raise AttributeError(f"Sorry, not support '{rotate_method}'.")
        rotate = rotate_method(atoms.positions, target.positions)
    return rotate

def reorder_atoms_indices(target, atoms, reorder_method='hungarian'):
    r""" Reorder atoms indices

    Parameters
    ----------
    target : :obj: ASE atoms object
        References atoms
    atoms : :obj: ASE atoms object
        Initialize atoms
    reorder_method : str

        | 'hungarian'
        | 'inertia-hungarian'
        | 'brute'
        | 'distance'

    Returns
    -------
    indices : ndarray
        Reordered atom indices """
    if reorder_method is None or reorder_method == 'none':
        indices = np.arange(len(atoms))
    else :
        if not hasattr(reorder_method, '__call__'):
            if reorder_method == 'hungarian' :
                reorder_method = reorder_hungarian
            elif reorder_method == 'inertia-hungarian' :
                reorder_method = reorder_inertia_hungarian
            elif reorder_method == 'brute' :
                reorder_method = reorder_brute
            elif reorder_method == 'distance' :
                reorder_method = reorder_distance
            else :
                raise AttributeError(f"Sorry, not support '{reorder_method}'.")
        indices = reorder_method(np.asarray(target.get_chemical_symbols()),
                np.asarray(atoms.get_chemical_symbols()), target.positions, atoms.positions)
    return indices


class MOMOrbitalTracker:

    """
        mol : New mol PySCF object
        mo_ref: Reference MO
        """
    
    def __init__(self, mol, mo_ref):
        self.S = mol.intor('int1e_ovlp')
        self.mo_ref = mo_ref.copy()

    def reorder(self, mo_coeff):
        # overlap matrix
        O = np.abs(self.mo_ref.conj().T @ self.S @ mo_coeff)

        # Hungarian algorithm: maximize total overlap
        row, col = linear_sum_assignment(-O)

        mo_new = mo_coeff[:, col]

        # update reference
        self.mo_ref = mo_new.copy()

        return mo_new, col
