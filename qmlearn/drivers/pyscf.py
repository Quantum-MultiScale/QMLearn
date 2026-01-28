import os
import itertools
import numpy as np
from scipy.linalg import eig
from scipy.spatial.transform import Rotation
from ase import Atoms, io
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto, ao2mo
from pyscf import dft, scf, mp, fci, ci, cc, mcscf, lib
from pyscf.symm import Dmatrix
from pyscf.scf import addons
from functools import reduce
from pyscf.scf import cphf
from pyscf.grad.mp2 import _shell_prange

from qmlearn.drivers.core import Engine, MOMOrbitalTracker

methods_pyscf = {
        'dft' : dft.RKS,
        'hf' : scf.RHF,
        'rks' : dft.RKS,
        'rhf' : scf.RHF,
        'mp2' : mp.MP2,
        'cisd': ci.CISD,
        'fci' : fci.FCI,
        'ccsd' : cc.CCSD,
        'ccsd(t)' : cc.CCSD,
        'casci' : mcscf.CASCI,
        'casscf' : mcscf.CASSCF,
        }

class EnginePyscf(Engine):
    r""" PySCF calculator

    Attributes
    ----------

    vext : ndarray
        External Potential.
    gamma : ndarray
        1-body reduced density matrix (1-RDM).
    gamma2 : ndarray
        2-body reduced density matrix (2-RDM).
    etotal : float
        Total electronic energy (Atomic Units a.u.)
    forces : ndarray
        Atomic forces (Atomic Units a.u.)
    ovlp : ndarray
        Overlap Matrix.
    kop : ndarray
        Kinetic Energy Operator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init()

    def init(self, **kwargs):
        r""" Function to initialize mf PySCF object

        Parameters
        ----------
        mol : :obj: PySCF or ASE atom object
            Molecular geometry. Coordinates in Angstroms.
        basis : dict or str
            To define basis set.
        xc : dict or str
            To define xchange-correlation functional
        verbose :int
            Printing level
        method : str

            | DFT : 'dft'
            | HF : 'hf'
            | RKS : 'rks'
            | RHF : 'rhf
            | MP2 : 'mp2'
            | CISD : 'cisd'
            | FCI : 'fci'

        charge : int
            Total charge of the molecule
        """
        mol = self.options.get('mol', None)
        mf = self.options.get('mf', None)
        basis = self.options.get('basis', '6-31g')
        xc = self.options.get('xc', None) or ''
        verbose = self.options.get('verbose', 0)
        self.method = self.options.get('method', 'rks').lower()
        charge = self.options.get('charge', None)
        symmetry = self.options.get('symmetry',False)
        smearing = self.options.get('smearing',False)
        #ao_repr = self.options.get('ao_repr', True)
        #
        if isinstance(self.method, str):
            if self.method in ['mp2', 'cisd', 'fci', 'ccsd', 'ccsd(t)', 'casci', 'casscf'] :
                self.method = 'rhf+' + self.method
        if self.method.count('+') > 1 :
            raise AttributeError("Sorry, only support two methods at the same time.")
        for fm in self.method.split('+') :
            if fm not in methods_pyscf :
                raise AttributeError(f"Sorry, not support the {fm} method now")
        if mf is None :
            mol = self.init_mol(mol, basis=basis, charge = charge,symmetry=symmetry)
            mf = methods_pyscf[self.method.split('+')[0]](mol)
            mf.xc = xc
            mf.verbose = verbose
        if smearing:
#            print(f"Using smearing: {smearing}")
            mf = scf.addons.smearing_(mf, sigma=0.094, method='fermi')
        self.mf = mf
        self.mol = self.mf.mol
        self.h1e = self.mf.get_hcore()
        self.h2e = self.mf.mol.intor('int2e')

    def init_mol(self, mol, basis, charge = 0,symmetry=False):
        r""" Function to create PySCF atom object

        Parameters
        ----------
        mol : list or str (From PySCF) or ASE atom object
            To define molecluar structure.  The internal format is

            | atom = [[atom1, (x, y, z)],
            |         [atom2, (x, y, z)],
            |         ...
            |         [atomN, (x, y, z)]]

        basis:  dict or str
            To define basis set.

        Returns
        -------
        atoms : :obj: PySCF atom object
           Molecular Structure, basis, and charge definition into gto PySCF atom object
        """
        ase_atoms = None
        if isinstance(mol, Atoms):
            ase_atoms = mol
        if isinstance(mol, (str, os.PathLike)):
            ase_atoms = io.read(mol)
        else:
            atom = mol
        if ase_atoms is not None :
            atom = atoms_from_ase(ase_atoms)

        if isinstance(atom, gto.Mole):
            mol = atom
        else :
            mol = gto.M(atom = atom, basis=basis, charge = charge,symmetry=symmetry, parse_arg = False)
        return mol

    def run(self, properties = ('energy', 'forces'), ao_repr=True, eig=True, singlet=True, mom=False, refqmmol=None, **kwargs):
        r""" Caculate electronic properties using PySCF.

        Parameters
        ----------
        properties : str

            | Total electronic energy : 'energy'
            | Total atomic forces : 'forces'

        If 'energy' is choosen the following properties are also calculated:

            | 1-RDM (1-body reduced density matrix) : 'gamma'
            | 2-RDM (2-body reduced density matrix) : 'gamma2'
            | Occupation number : 'occs'
            | Molecular orbitals : 'orb'

        ao_repr : bool
            If True, will use atomic orbitals for calculating electronic properties

        """
        if 'energy' in properties or self._gamma is None :
            # (dft.uks.UKS, dft.rks.RKS, scf.uhf.UHF, scf.hf.RHF))
            dm0_ref = self.options.get('dm0',None)
            if dm0_ref is not None:
                self.mf.run(dm0=dm0_ref)
            else:
                self.mf.run()

            self._mo_energy = self.mf.mo_energy
            smearing = self.options.get('smearing',False)
            if smearing:
                self.occs = self.mf.get_occ(mo_energy=self._mo_energy)
            else:
                self.occs = self.mf.get_occ()
            norb = self.occs.shape[0]
            
            #
            if '+' in self.method :
                method2 = self.method.split('+')[1]
                if method2 == 'fci':
                    nroots = self.options.get('nroots', 1)
                    if nroots is None : nroots = 1
                    mf2 = methods_pyscf[method2](self.mf)
                    mf2.verbose = self.mf.verbose
                    #
                    if nroots > 1:
                        mf2.nroots = nroots
                        es, cis = mf2.kernel()
                        e, ci_ = es[0], cis[0]
                        #
                        if 'tgamma' in properties:
                            self._tgamma = {}
                            for i,j in itertools.combinations_with_replacement(range(nroots), 2):
                                rdm = mf2.trans_rdm1(cis[i], cis[j], norb=norb, nelec=self.mol.nelec)
                                if ao_repr:
                                    rdm = self.gamma_mo2ao(rdm, mo_coeff=self.mf.mo_coeff)
                                self._tgamma[(i,j)] = rdm
                        #
                    else:
                        e, ci_ = mf2.kernel()
                    self._orb = self.mf.mo_coeff
                    self._etotal = e
                    #
                    if 'gamma2' in properties or 'gamma2c' in properties :
                        self._gamma, self._gamma2 = mf2.make_rdm12(fcivec=ci_, norb=norb, nelec=self.mol.nelec)
                        if ao_repr:
                           print('AO Representation')
                           self._gamma = self.gamma_mo2ao(self._gamma, mo_coeff=self.mf.mo_coeff)
                           self._gamma2 = cc.ccsd_rdm._rdm2_mo2ao(self._gamma2.transpose(1,0,3,2), self._orb)
                           if eig:
                              self._eig_gamma2 = self.eigs_gamma2(self._gamma2)[0]
                        else:
                           print('MO Representation')
                        self._occ = self.calc_occupations(self._gamma)[0]

                    else :
                        self._gamma = mf2.make_rdm1(fcivec=ci_, norb=norb, nelec=self.mol.nelec)
                        self._occ = np.diagonal(self._gamma)
                        if ao_repr is True:
                            self._gamma = self.gamma_mo2ao(self._gamma, mo_coeff=self.mf.mo_coeff)
                            self._occ = self.calc_occupations(self._gamma)[0]

                    if 'gamma2c' in properties or 'gamma2cum' in properties:
                         use_cumulant = 'gamma2cum' in properties
                         print('Cumulant Gamma2' if use_cumulant else 'Correlated Gamma2')

                         gamma_ref = self._gamma if use_cumulant else self.mf.make_rdm1(ao_repr = ao_repr)
                         gamma_a = np.einsum('pq,rs->pqrs',gamma_ref,gamma_ref)
                         gamma_b = np.einsum('pq,rs->psrq',gamma_ref,gamma_ref)

                         self._gamma2c = self._gamma2 - .5*(2*gamma_a-gamma_b)
                         self._delta_gamma = self._gamma - self.mf.make_rdm1(ao_repr = ao_repr)
                         self._occ_dg = self.calc_occupations(self._delta_gamma)[0]
                         if eig:
                            self._eig_gamma2c = self.eigs_gamma2(self._gamma2c)[0]

                    print('FCI energy: ', self._etotal)
                    self.mf2 = mf2

                elif method2 == 'cisd':
                    mf2 = ci.CISD(self.mf)
                    eris = cc.ccsd._make_eris_outcore(mf2, self.mf.mo_coeff)
                    ecisd, civec = mf2.kernel(eris=eris)
                    self._orb = self.mf.mo_coeff

                    if 'gamma2' in properties or 'gamma2c' in properties:
                         self._gamma = mf2.make_rdm1(civec,ao_repr=ao_repr)
                         self._gamma2 = mf2.make_rdm2(civec,ao_repr=ao_repr)
                         if eig:
                            self._eig_gamma2 = self.eigs_gamma2(self._gamma2)[0]
                         if ao_repr:
                            print('AO Representation')
                         else:
                            print('MO Representation')
                    else :
                         self._gamma = mf2.make_rdm1(civec,ao_repr = ao_repr, **kwargs)

                    if 'gamma2c' in properties or 'gamma2cum' in properties:
                         use_cumulant = 'gamma2cum' in properties
                         print('Cumulant Gamma2' if use_cumulant else 'Correlated Gamma2')

                         gamma_ref = self._gamma if use_cumulant else self.mf.make_rdm1(ao_repr = ao_repr)
                         gamma_a = np.einsum('pq,rs->pqrs',gamma_ref,gamma_ref)
                         gamma_b = np.einsum('pq,rs->psrq',gamma_ref,gamma_ref)

                         self._gamma2c = self._gamma2 - .5*(2*gamma_a-gamma_b)
                         self._delta_gamma = self._gamma - self.mf.make_rdm1(ao_repr = ao_repr)
                         self._occ_dg = self.calc_occupations(self._delta_gamma)[0]
                         if eig:
                            self._eig_gamma2c = self.eigs_gamma2(self._gamma2c)[0]

                    self._occ = self.calc_occupations(self._gamma)[0]
                    self._etotal = mf2.e_tot
                    print('CISD energy: ', self._etotal)
                    self.mf2 = mf2

                elif method2 in ['casci', 'casscf']:
                    ncas = self.options.get('ncas', None)
                    nelecas = self.options.get('nelecas', None)
                    nroots = self.options.get('nroots', None)
                    ci_ref = self.options.get('ci_ref',None)
                    if ncas is None : ncas = norb
                    if nelecas is None : nelecas = self.mol.nelectron
                    if nroots is None : nroots = 1
                    mf2 = methods_pyscf[method2](self.mf, ncas, nelecas)
                    mf2.verbose = self.mf.verbose
                    mf2.verbose = 3
                    mf2.fcisolver = fci.solver(self.mol,singlet=singlet)
                    mf2.fcisolver.nroots = nroots
                    
                    #mf2.sorting_mo_energy = True
                    #print('Check: ', properties)
                    if ci_ref is not None:
                        print('Using CI REF! ')
                        mf2.kernel(ci0=ci_ref)
                    elif mom:
                        print('Using MOM method: ')
                        mol_ref = refqmmol.engine.mol
                        mo_ref = refqmmol.engine.mf.mo_coeff
                        
                        tracker = MOMOrbitalTracker(mol_ref, mo_ref)
                        mo_new, perm = tracker.reorder(self.mf.mo_coeff)
                        self.mf.mo_coeff = mo_new
                        self.mf.mo_occ = self.mf.mo_occ[perm]
                        self.mf.mo_energy = self.mf.mo_energy[perm]
                        mf2 = methods_pyscf[method2](self.mf, ncas, nelecas)
                        mf2.kernel()
                    else:
                        mf2.kernel()

                    if 'gamma_trans' in properties:
                        self.mf2 = mf2
                        self._gamma_00, self._gamma_01, self._gamma_10, self._gamma_11 = self.calc_gamma_trans(
                                                                                         properties=properties,
                                                                                         ao_repr = ao_repr, 
                                                                                         ncas=ncas,
                                                                                         nelecas=nelecas)
                    if nroots == 1:
                        self._gamma = mf2.make_rdm1() # AO basis
                    else:
                        mf2.fcisolver.nroots = 1
                        mf2.kernel()
                        self._gamma = mf2.make_rdm1()
                    self._occ = self.calc_occupations(self._gamma)[0]
                    self.mf2 = mf2

                    has_gamma2 = 'gamma2' in properties
                    has_gamma2c = 'gamma2c' in properties
                    has_gamma2cum = 'gamma2cum' in properties

                    if has_gamma2 and not (has_gamma2c or has_gamma2cum):
                        self._gamma2 = self.calc_gamma2_cas(properties=properties,
                                                            ao_repr=ao_repr, ncas=ncas, nelecas=nelecas)

                    elif has_gamma2 and (has_gamma2c or has_gamma2cum):
                        self._gamma2, self._gamma2c = self.calc_gamma2_cas(properties=properties,
                                                                           ao_repr = ao_repr,
                                                                           ncas=ncas, nelecas=nelecas)
                    elif (has_gamma2c or has_gamma2cum) and not has_gamma2:
                        self._gamma2c = self.calc_gamma2_cas(properties=properties, ao_repr = ao_repr,  
                                                             ncas=ncas, nelecas=nelecas)
                    if has_gamma2c or has_gamma2cum:
                       self._delta_gamma = self._gamma - self.mf.make_rdm1(ao_repr = ao_repr)
                       self._occ_dg = self.calc_occupations(self._delta_gamma)[0]

                    if eig:
                       if has_gamma2:
                           self._eig_gamma2 = self.eigs_gamma2(self._gamma2)[0]
                       if has_gamma2c:
                           self._eig_gamma2c = self.eigs_gamma2(self._gamma2c)[0]

                    self._orb = self.mf.mo_coeff
                    self._etotal = mf2.e_tot
                    
                    print('CASCI E: ', self._etotal)
                else :
                    mf2 = methods_pyscf[method2](self.mf)
                    mf2.verbose = self.mf.verbose
                    mf2.run()
                    self._orb = self.mf.mo_coeff
                    self._gamma = mf2.make_rdm1(ao_repr = ao_repr, **kwargs)
                    print('METHOD: ',method2)
                    self._gamma2 = mf2.make_rdm2(ao_repr = ao_repr, **kwargs)
                    self._etotal = mf2.e_tot
                    self._occ = self.calc_occupations(self._gamma)[0]

                    if 'gamma2c' in properties or 'gamma2cum' in properties:
                         use_cumulant = 'gamma2cum' in properties
                         print('Cumulant Gamma2' if use_cumulant else 'Correlated Gamma2')

                         gamma_ref = self._gamma if use_cumulant else self.mf.make_rdm1(ao_repr = ao_repr)
                         gamma_a = np.einsum('pq,rs->pqrs',gamma_ref,gamma_ref)
                         gamma_b = np.einsum('pq,rs->psrq',gamma_ref,gamma_ref)

                         self._gamma2c = self._gamma2 - .5*(2*gamma_a-gamma_b)
                         self._delta_gamma = self._gamma - self.mf.make_rdm1(ao_repr = ao_repr)
                         self._occ_dg = self.calc_occupations(self._delta_gamma)[0]
                         if eig: 
                            self._eig_gamma2c = self.eigs_gamma2(self._gamma2c)[0]

                    if method2 == 'ccsd(t)':
                        ccsd_t = mf2.ccsd_t()
                        # print('ccst(t)', self._etotal, ccsd_t)
                        self._etotal = self._etotal + ccsd_t
                self.mf2 = mf2
                self._etotal_c = self._etotal - self.mf.e_tot
            else :
                mf = self.mf
                self._orb = mf.mo_coeff
                self._gamma = mf.make_rdm1(ao_repr = ao_repr, **kwargs)
                self._etotal = mf.e_tot

        if 'forces' in properties :
            self._forces = self.run_forces()

    def calc_gamma_trans(self, properties, ao_repr=True, ncas=None,nelecas=None,*kwargs):

        """ Built Diabatic states \gamma_00, \gamma_01, \gamma_10, \gamma_00 """

        if ncas is None:
           ncas = self.mf2.ncas
        if nelecas is None:
           nelecas = self.mf2.nelecas

        ci = self.mf2.ci
        ncore = self.mf2.ncore
        norb=ncas
        nelec=np.sum(nelecas)

        psi_0 = (np.array(ci[0]) + np.array(ci[1])) / np.sqrt(2)
        psi_1 = (np.array(ci[0]) - np.array(ci[1])) / np.sqrt(2)
        
        gamma_00 = self.mf2.make_rdm1(ci=psi_0,ncas=ncas,nelecas=nelecas)
        gamma_11 = self.mf2.make_rdm1(ci=psi_1,ncas=ncas,nelecas=nelecas)
        rdm1_s0_s1 = self.mf2.fcisolver.trans_rdm1(psi_0,psi_1,norb,nelec)
        rdm1_s1_s0 = self.mf2.fcisolver.trans_rdm1(psi_1,psi_0,norb,nelec)

        mo_coeff = self.mf2.mo_coeff
         
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1_hf = np.dot(mocore, mocore.conj().T) * 2
        dm1_hf = np.zeros_like(dm1_hf)

        gamma_01 = dm1_hf + reduce(np.dot, (mocas, rdm1_s0_s1, mocas.conj().T))
        gamma_10 = dm1_hf + reduce(np.dot, (mocas, rdm1_s1_s0, mocas.conj().T))
        
        return gamma_00, gamma_01, gamma_10, gamma_11

    def calc_gamma2_cas(self, properties, ao_repr=True,ncas=None,nelecas=None,reorder=True,*kwargs):
        ncore = self.mf2.ncore
        ncas = self.mf2.ncas
        ci = self.mf2.ci
        mo_coeff = self.mf.mo_coeff
        # mocas = mo_coeff[:,ncore:ncore+ncas]
        # amo_s = mocas.shape[1]
        casdm2 = self.mf2.fcisolver.make_rdm2(ci, ncas, nelecas, reorder=reorder) # MO basis
        casdm1 = self.mf2.fcisolver.make_rdm1(ci, ncas, nelecas) # MO basis

        has_gamma2 = 'gamma2' in properties
        has_gamma2c = 'gamma2c' in properties
        has_gamma2cum = 'gamma2cum' in properties


        if has_gamma2:
            rdm2_hfc = np.zeros_like(mo_coeff)
            for i in range(ncore):
                rdm2_hfc[i,i] = 2
            rdm2_hfc[ncore:ncore+ncas,ncore:ncore+ncas] = casdm1

            gamma_a = np.einsum('pq,rs->pqrs',rdm2_hfc,rdm2_hfc,optimize=True)
            gamma_b = np.einsum('pq,rs->psrq',rdm2_hfc,rdm2_hfc,optimize=True)
            dm2_non_int = .5*(2*gamma_a-gamma_b)

            dm2 = dm2_non_int
            dm2[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas] = casdm2
            rdm2_hf_mo = dm2
            if ao_repr:
                mo_ = mo_coeff
                dm2 = np.einsum('ijkl,pi,qj,rk,sl->pqrs', rdm2_hf_mo,
                               mo_, mo_,
                               mo_, mo_,optimize=True) # AO basis


        if (has_gamma2c or has_gamma2cum):
            use_cumulant = 'gamma2cum' in properties
            print('Cumulant Gamma2' if use_cumulant else 'Correlated Gamma2')

            gamma_ref = self._gamma if use_cumulant else self.mf.make_rdm1(ao_repr=ao_repr)

            inv= np.linalg.inv(mo_coeff)
            gamma_ = np.einsum('pi,ij,qj->pq', inv, gamma_ref, inv.conj(),optimize=True) # MO basis
            gamma_a = np.einsum('pq,rs->pqrs',gamma_,gamma_,optimize=True)
            gamma_b = np.einsum('pq,rs->psrq',gamma_,gamma_,optimize=True)
            rdm2_hf_mo = .5*(2*gamma_a-gamma_b)

            dm1_mo = np.zeros_like(mo_coeff)
            for i in range(ncore):
                dm1_mo[i,i] = 2
            dm1_mo[ncore:ncore+ncas,ncore:ncore+ncas] = casdm1
            gamma_a = np.einsum('pq,rs->pqrs',dm1_mo,dm1_mo,optimize=True)
            gamma_b = np.einsum('pq,rs->psrq',dm1_mo,dm1_mo,optimize=True)
            rdm2_cas_mo = .5*(2*gamma_a-gamma_b)

            gamma_2c = rdm2_cas_mo - rdm2_hf_mo
            rdm2_hf_valance = rdm2_hf_mo[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas]
            gamma_2c[ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas,ncore:ncore+ncas] = casdm2 - rdm2_hf_valance
            dm2_cas = gamma_2c[:ncore+ncas,:ncore+ncas,:ncore+ncas,:ncore+ncas]
            dm2c = dm2_cas

            if ao_repr:
                mo_ = mo_coeff[:,:ncore+ncas]
                dm2c = np.einsum('ijkl,pi,qj,rk,sl->pqrs', dm2_cas,
                                    mo_, mo_,
                                    mo_, mo_,optimize=True) # AO basis
        if (has_gamma2c or has_gamma2cum) and has_gamma2:
            results = dm2,dm2c
        elif (has_gamma2c or has_gamma2cum) and not has_gamma2:
            results = dm2c
        elif has_gamma2 and not (has_gamma2c or has_gamma2cum):
            results = dm2

        return results

    @property
    def tgamma(self):
        if self._tgamma is None:
            self.run(properties = ('energy', 'tgamma'))
        return self._tgamma

    @property
    def gamma(self):
        if self._gamma is None:
            self.run(properties = ('energy'))
        return self._gamma

    @property
    def gamma2(self):
        if self._gamma2 is None:
            self.run(properties = ('energy','gamma2'))
        return self._gamma2

    @property
    def gamma2c(self):
        if self._gamma2c is None:
            self.run(properties = ('energy','gamma2c'))
        return self._gamma2c

    @property
    def gamma2cum(self):
        if self._gamma2c is None:
            self.run(properties = ('energy','gamma2cum'))
        return self._gamma2c

    @property
    def eig_gamma2c(self):
        if self._eig_gamma2c is None:
            self.run(properties = ('energy','gamma2c'))
        return self._eig_gamma2c

    @property
    def eig_gamma2(self):
        if self._eig_gamma2 is None:
            self.run(properties = ('energy','gamma2'))
        return self._eig_gamma2

    @property
    def delta_gamma(self):
        if self._delta_gamma is None:
            self.run(properties = ('energy','gamma2c'))
        return self._delta_gamma

    @property
    def all_gammas(self):
        if self._gamma2 is None:
            self.run(properties = ('energy','gamma2','gamma2c'), eig=False)
        return self._gamma,self._gamma2,self._gamma2c,self._delta_gamma

    @property
    def all_gammas_cum(self):
        if self._gamma2cum is None:
            self.run(properties = ('energy','gamma2','gamma2cum'), eig=False)
        return self._gamma,self._gamma2,self._gamma2c,self._delta_gamma

    @property
    def gamma_trans(self):
        if self._gamma_00 is None:
            self.run(properties = ('energy','gamma_trans'))
        return self._gamma_00, self._gamma_01, self._gamma_10, self._gamma_11

    @property
    def occ_dg(self):
        if self._occ_dg is None:
            self.run(properties = ('energy','gamma2c'))
        return self._occ_dg

    @property
    def etotal(self):
        if self._etotal is None:
            self.run(properties = ('energy'))
        return self._etotal

    @property
    def etotal_c(self):
        if self._etotal_c is None:
            self.run(properties = ('energy'))
        return self._etotal_c

    @property
    def forces(self):
        if self._forces is None:
            self.run(properties = ('forces'))
        return self._forces

    @property
    def vext(self):
        if self._vext is None:
            self._vext = self.mol.intor_symmetric('int1e_nuc')
        return self._vext

    @property
    def kop(self):
        if self._kop is None:
            self._kop = self.mol.intor_symmetric('int1e_kin')
        return self._kop

    @property
    def ovlp(self):
        if self._ovlp is None:
            self._ovlp = self.mol.intor_symmetric('int1e_ovlp')
        return self._ovlp

    @property
    def occ(self):
        if self._occ is None:
            self.run(properties = ('energy','gamma2c'))
        return self._occ

    @property
    def mo_energy(self):
        if self._mo_energy is None:
            self.run(properties = ('energy'))
        return self._mo_energy

    @property
    def nelectron(self):
        r""" Total number of electrons. """
        return self.mol.nelectron

    @property
    def nao(self):
        r""" Natural atomic orbitals. """
        return self.mol.nao

    def calc_gamma(self, orb=None, occs=None):
        if orb is None : orb = self.orb
        if occs is None : orb = self.occs
        nm = min(orb.shape[1], len(occs))
        gamma = self.mf.make_rdm1(orb[:, nm], occs[:nm])
        return gamma

    def calc_etotal(self, gamma, **kwargs):
        r""" Get the total electronic energy based on 1-RDM.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        etotal : float
            Total electronic energy. """
        etotal = self.mf.energy_tot(gamma, **kwargs)  # nuc + energy_elec(e1+coul)
        return etotal

    def calc_etotal2(self,gamma2=None,gamma=None,ao_repr=None,hf_core=False,
                     g_c=False, delta_g=None,cum=False,gamma2cum=None, **kwargs):
        r""" Get the total electronic energy based on 2-RDM.

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        gamma2 : ndarray
            2-RDM
        ao_repr : bool
            False : Both 1-RDM and 2-RDM in the MO basis
            True : Both 1-RDM and 2-RDM in the AO basis
            None : 1-RDM in AO basis and 2-RDM in the MO basis.

        Returns
        -------
        etotal : float
            Total electronic energy. """

        if not ao_repr: # MO representation
            self.mf.run() # HF to get orb and occ
            orb = self.mf.mo_coeff
            occs = self.mf.get_occ()
            nmo = len(occs)
            h1e = reduce(np.dot,(orb.T, h_core,orb))
            h2e = ao2mo.kernel(self.mol.intor('int2e'),orb)
            h2e = ao2mo.restore(1, h2e, nmo)
        else:
            h1e=self.h1e
            h2e=self.h2e
#            h_core=self.mf.get_hcore()
#            h1e = h_core
#            h2e = self.mol.intor('int2e')

        if hf_core:
            self.mf.run() # HF to get orb and occ
            ove=self.mf.get_ovlp()

            if delta_g is None:
                gamma_hf = self.mf.make_rdm1(ao_repr = ao_repr)
                delta_gamma = gamma-gamma_hf
            else:
                delta_gamma = delta_g
                print('Taking predicted delta_gamma')

            if self._gamma2 is not None and g_c is False:
                gamma_a = np.einsum('pq,rs->pqrs',gamma_hf,gamma_hf,optimize=True)
                gamma_b = np.einsum('pq,rs->psrq',gamma_hf,gamma_hf,optimize=True)
                gammac = gamma2 - .5*(2*gamma_a-gamma_b)
            else:
                print('Taking predicted Gamma_C!')
                gammac = gamma2

            if not np.allclose(np.einsum('mnst,mn,st',gammac,ove,ove,optimize=True), 0, atol=1e-10):
                print('2RDM-correlated trace is not ZERO -> N', np.einsum('mnst,mn,st',gammac,ove,ove,optimize=True),'!=', 0)
            else:
                print('2RDM-correlated trace is ZERO')

            h1_c=np.einsum('ij,ji', h1e, delta_gamma,optimize=True)
            h2_c=np.einsum('ijkl,ijkl', h2e, gammac,optimize=True) * .5
            etotal = self.mf.e_tot + h1_c + h2_c
            print('HF:', self.mf.e_tot,' ','Con_H1: ', h1_c, 'Con_H2: ', h2_c)

        if cum:
            gamma_a = np.einsum('pq,rs->pqrs',gamma,gamma,optimize=True)
            gamma_b = np.einsum('pq,rs->psrq',gamma,gamma,optimize=True)
            gammac = 0.5*(2*gamma_a-gamma_b)

            h1_c = np.einsum('ij,ji', h1e, gamma)
            h2_g1 = np.einsum('ijkl,ijkl', h2e, gammac,optimize=True) * .5
            h2_cum = np.einsum('ijkl,ijkl', h2e, gamma2cum,optimize=True) * .5
            etotal = h1_c + h2_g1 + h2_cum
            etotal += self.mol.energy_nuc()
            print('Con_H1: ', h1_c, 'Con_H2_g1: ', h2_g1, 'Con_H2_Cum: ', h2_cum)

        if hf_core is False and cum is False:
            h1_c=np.einsum('ij,ji', h1e, gamma,optimize=True)
            h2_c=np.einsum('ijkl,ijkl', h2e, gamma2,optimize=True) * .5
#            temp = np.tensordot(h2e, gamma2, axes=([2,3],[2,3]))  # shape (M,M,M,M) -> (M,M)
#            h2_c = 0.5 * np.tensordot(temp, np.eye(temp.shape[0]), axes=2) 

            etotal = h1_c + h2_c
            etotal += self.mol.energy_nuc()
            print('Con_H1: ', h1_c, 'Con_H2: ', h2_c)

        print('Total Energy: ',etotal)

        return etotal

    def calc_dipole(self, gamma, **kwargs):
        r""" Get the total dipole moment.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        dip : list
            The dipole moment on x, y and z component."""
        dip = self.mf.dip_moment(self.mol, gamma, unit = 'au', verbose=self.mf.verbose)
        return dip

    def run_forces(self, **kwargs):
        r""" Function to calculate Forces with calculated 1-RDM

        Returns
        -------
        forces : ndarray
           Total atomic forces. """
        if '+' in self.method :
            if self.method.split('+')[1] in ['fci']:  # With FCI method Forces can't be obtained from FCI object. I approximated using RHF object.
                mf = self.mf
            else:
                mf = self.mf2
        else :
            mf = self.mf

        gf = mf.nuc_grad_method()
        gf.verbose = mf.verbose
        if hasattr(gf, 'grid_response') :
            gf.grid_response = True
        smearing = self.options.get('smearing',False)
        if smearing:    
            # avoid use CPHF equations
            forces = self.get_forces_fci(gamma=self.gamma,gamma2=self.gamma2)
        else:
            forces = -1.0 * gf.kernel()
        return forces

    def calc_forces(self, gamma, **kwargs):
        r""" Function to calculate Forces with a given 1-RDM

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        forces : ndarray
           Total atomic forces for a given 1-RDM. """

        # only for rhf and rks
        if self.method.split('+')[0] not in ['rhf', 'rks'] or '+' in self.method :
            raise AttributeError(f"Sorry the calc_forces not support '{self.method}'")
        gf = self.mf.nuc_grad_method()
        gf.verbose = self.mf.verbose
        gf.grid_response = True
        # Just a trick to skip the make_rdm1 and make_rdm1e without change the kernel
        save_make_rdm1, self.mf.make_rdm1 = self.mf.make_rdm1, gamma_to_gamma
        save_make_rdm1e, gf.make_rdm1e = gf.make_rdm1e, gamma_to_rdm1e
        forces = -1.0 * gf.kernel(mo_energy=self.mf, mo_coeff=gamma, mo_occ=gamma)
        self.mf.make_rdm1 = save_make_rdm1
        gf.make_rdm1e = save_make_rdm1e
        return forces

    def _calc_quadrupole_r(self, gamma, component = 'zz'):
        if component != 'zz' :
            raise AttributeError("Sorry, only support 'zz' component now")
        r2 = self.mol.intor_symmetric('int1e_r2')
        z2 = self.mol.intor_symmetric('int1e_zz')
        q = 1/2*(3*z2 - r2)
        quad = np.einsum('mn, mn->', gamma, q)
        return quad

    def calc_quadrupole(self, gamma, traceless = True):
        r""" Function to calculate the total quadruple for XX, XY, XY, YY, YZ, ZZ components.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        quadrupole : ndarray
           An array containing a quadruple per component."""

        # XX, XY, XZ, YY, YZ, ZZ
        with self.mol.with_common_orig((0,0,0)):
            q = self.mol.intor('int1e_rr', comp=9).reshape((3,3,*gamma.shape))
        quadp = np.zeros(6)
        quadp[0] = np.einsum('ij,ij->', q[0, 0], gamma)
        quadp[1] = np.einsum('ij,ij->', q[0, 1], gamma)
        quadp[2] = np.einsum('ij,ij->', q[0, 2], gamma)
        quadp[3] = np.einsum('ij,ij->', q[1, 1], gamma)
        quadp[4] = np.einsum('ij,ij->', q[1, 2], gamma)
        quadp[5] = np.einsum('ij,ij->', q[2, 2], gamma)
        quadp *= -1.0
        quadp += self.calc_quadrupole_nuclear()
        if traceless :
            trace = (quadp[0] + quadp[3] + quadp[5])/3.0
            quadp[0] -= trace
            quadp[3] -= trace
            quadp[5] -= trace
        return quadp

    def calc_quadrupole_nuclear(self, mol = None):
        r""" Function to calculate the nuclear quadruple for XX, XY, XY, YY, YZ, ZZ components.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        quadrupol : ndarray
           An array containing the nuclear quadruple per component."""

        mol = mol or self.mol
        charges = mol.atom_charges()
        pos = mol.atom_coords()
        nucquad = np.zeros(6)
        nucquad[0] = np.sum(charges * pos[:, 0] * pos[:, 0])
        nucquad[1] = np.sum(charges * pos[:, 0] * pos[:, 1])
        nucquad[2] = np.sum(charges * pos[:, 0] * pos[:, 2])
        nucquad[3] = np.sum(charges * pos[:, 1] * pos[:, 1])
        nucquad[4] = np.sum(charges * pos[:, 1] * pos[:, 2])
        nucquad[5] = np.sum(charges * pos[:, 2] * pos[:, 2])
        return nucquad

    def _get_loc_orb(self, nocc=-1):
        r2 = self.mol.intor_symmetric('int1e_r2')
        r = self.mol.intor_symmetric('int1e_r', comp=3)
        M = np.einsum('mi,nj,mn->ij', self.mf.mo_coeff[:,:nocc], self.mf.mo_coeff[:,:nocc], r2)
        J = np.einsum('mi,nj,tmn->tij', self.mf.mo_coeff[:,:nocc], self.mf.mo_coeff[:,:nocc], r)**2
        M -= np.einsum('tij-> ij', J)
        w,vr = eig(M)
        mo = self.mf.mo_coeff[:,:nocc] @ vr.T
        return mo

    def rotation2rotmat(self, rotation, mol = None, factor=1.0, angle = 'zyz'):
        r""" Function to rotate the density matrix.

        Parameters
        ----------
        rotation : ndarray
            Rotation Matrix
        mol : :obj: PySCF mol object
            Molecluar structure

        Returns
        -------
        rotmat : ndarray
        Rotated density matrix """

        mol = mol or self.mol
        return rotation2rotmat(rotation, mol, factor=factor, angle=angle)

    def get_atom_naos(self, mol = None):
        mol = mol or self.mol
        return get_atom_naos(mol)

    def project_dm_nr2nr(self, gamma, mol2, mol = None):
        mol = mol or self.mol
        if hasattr(mol, 'mol'): mol = mol.mol
        if hasattr(mol2, 'mol'): mol2 = mol2.mol
        dm = addons.project_dm_nr2nr(mol, gamma, mol2)
        return dm

    def gamma_mo2ao(self, gamma, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        gamma_ao = np.einsum('pi,ij,qj->pq', mo_coeff, gamma, mo_coeff.conj(),optimize=True)
        return gamma_ao

    def gamma_ao2mo(self, gamma, mo_coeff=None, ovlp=None):
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        if ovlp is None: ovlp = self.ovlp
        gamma_mo = np.einsum('ji,jl,lm,mn,np->ip', mo_coeff.conj(), ovlp, gamma, ovlp, mo_coeff, optimize=True)
        return gamma_mo

    def gamma2_mo2ao(self, gamma2, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        mo = mo_coeff
        mo_c = mo.conj()
        gamma2_ao = np.einsum('ijkl,pi,qj,rk,sl->pqrs', gamma2,
            mo, mo_c, mo, mo_c, optimize=True)
        return gamma2_ao

    def gamma2_ao2mo(self, gamma2, mo_coeff=None, ovlp=None):
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        if ovlp is None: ovlp = self.ovlp
        invmo = np.einsum('ji, jl->il', mo_coeff, ovlp)
        invmo_c = invmo.conj()
        gamma2_ao = np.einsum('ijkl,pi,qj,rk,sl->pqrs', gamma2,
            invmo_c, invmo, invmo_c, invmo, optimize=True)
        return gamma2_ao

    @staticmethod
    def gamma2_to_2d(gamma2):
        r = gamma2.shape[0]
        gamma2_2d = np.transpose(gamma2,[0,2,1,3]).reshape((-1,r**2))
        return gamma2_2d

    @staticmethod
    def gamma2_to_4d(gamma2):
        r = int(np.sqrt(gamma2.shape[0]))
        gamma2_4d = np.transpose(gamma2.reshape((r,r,r,r)),[0,2,1,3])
        return gamma2_4d

    def grad_elec(self, gamma=None, gamma2=None, ncas=None, nelecs=None, 
              atmlst=None, fci=True, **kwargs):

        mc_grad = self.mf.nuc_grad_method()
        mo_coeff = self.mf.mo_coeff
        mol = self.mf.mol
        
        if fci:    
            ncas = len(self.mf.mo_coeff)
            neleca, nelecb = mol.nelec
            nelecas = (neleca, nelecb)
            ncore = 0
            nocc = ncas
        else:
            nelecb = (nelecs-self.mf.mol.spin)//2
            neleca = nelecs - nelecb
            nelecas = (neleca, nelecb)
            ncorelec = self.mf.mol.nelectron - sum(nelecas)
            ncore = ncorelec // 2
            nocc = ncore + ncas
            
        nao, nmo = mo_coeff.shape
        nao_pair = nao * (nao+1) // 2
        mo_energy = self.mf.mo_energy
     
        mo_occ = mo_coeff[:,:nocc]
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]
        neleca, nelecb = mol.nelec
        assert (neleca == nelecb)
        orbo = mo_coeff[:,:neleca]
        orbv = mo_coeff[:,neleca:]
        
        ####
        #convert AO to MO first... 
        inv_coeff = np.linalg.inv(mo_coeff)
        gamma_mo = np.einsum('pi,ij,qj->pq', inv_coeff, gamma, inv_coeff.conj(),
                             optimize=True)
        gamma2_mo = np.einsum('ijkl,pi,qj,rk,sl->pqrs', gamma2 ,
                           inv_coeff, inv_coeff, inv_coeff, inv_coeff, optimize=True)
     
        casdm1 = gamma_mo[ncore:ncore+ncas,ncore:ncore+ncas]
        casdm2 = gamma2_mo[ncore:ncore+ncas,ncore:ncore+ncas,
                           ncore:ncore+ncas,ncore:ncore+ncas]
        #####
        
        dm_core = np.dot(mo_core, mo_core.T) * 2
        dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
        aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False)
        aapa = aapa.reshape(ncas,ncas,nmo,ncas)
        vj, vk = self.mf.get_jk(mol, (dm_core, dm_cas))
        h1 = self.mf.get_hcore()
        vhf_c = vj[0] - vk[0] * .5
        vhf_a = vj[1] - vk[1] * .5
        # Imat = h1_{pi} gamma1_{iq} + h2_{pijk} gamma_{iqkj}
        Imat = np.zeros((nmo,nmo))
        Imat[:,:nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c + vhf_a, mo_occ)) * 2
        Imat[:,ncore:nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
        Imat[:,ncore:nocc] += lib.einsum('ijpk,jikq->pq', aapa, casdm2)
        aapa = vj = vk = vhf_c = vhf_a = h1 = None
     
        ee = mo_energy[:,None] - mo_energy
        zvec = np.zeros_like(Imat)
        zvec[:ncore,ncore:neleca] = Imat[:ncore,ncore:neleca] / -ee[:ncore,ncore:neleca]
        zvec[ncore:neleca,:ncore] = Imat[ncore:neleca,:ncore] / -ee[ncore:neleca,:ncore]
        zvec[nocc:,neleca:nocc] = Imat[nocc:,neleca:nocc] / -ee[nocc:,neleca:nocc]
        zvec[neleca:nocc,nocc:] = Imat[neleca:nocc,nocc:] / -ee[neleca:nocc,nocc:]
     
        zvec_ao = reduce(np.dot, (mo_coeff, zvec+zvec.T, mo_coeff.T))
        vhf = self.mf.get_veff(mol, zvec_ao) * 2
        xvo = reduce(np.dot, (orbv.T, vhf, orbo))
        xvo += Imat[neleca:,:neleca] - Imat[:neleca,neleca:].T
        def fvind(x):
            x = x.reshape(xvo.shape)
            dm = reduce(np.dot, (orbv, x, orbo.T))
            v = self.mf.get_veff(mol, dm + dm.T)
            v = reduce(np.dot, (orbv.T, v, orbo))
            return v * 2
        dm1resp = cphf.solve(fvind, mo_energy, self.mf.mo_occ, xvo, max_cycle=30)[0]
        zvec[neleca:,:neleca] = dm1resp
     
        zeta = np.einsum('ij,j->ij', zvec, mo_energy)
        zeta = reduce(np.dot, (mo_coeff, zeta, mo_coeff.T))
     
        zvec_ao = reduce(np.dot, (mo_coeff, zvec+zvec.T, mo_coeff.T))
        p1 = np.dot(mo_coeff[:,:neleca], mo_coeff[:,:neleca].T)
        vhf_s1occ = reduce(np.dot, (p1, self.mf.get_veff(mol, zvec_ao), p1))
     
        Imat[:ncore,ncore:neleca] = 0
        Imat[ncore:neleca,:ncore] = 0
        Imat[nocc:,neleca:nocc] = 0
        Imat[neleca:nocc,nocc:] = 0
        Imat[neleca:,:neleca] = Imat[:neleca,neleca:].T
        im1 = reduce(np.dot, (mo_coeff, Imat, mo_coeff.T))
     
        casci_dm1 = dm_core + dm_cas
        hf_dm1 = self.mf.make_rdm1(mo_coeff, self.mf.mo_occ)
        hcore_deriv = mc_grad.hcore_generator(mol)
        s1 = mc_grad.get_ovlp(mol)
     
        diag_idx = np.arange(nao)
        diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
        casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
        dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                    (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
        dm2buf = lib.pack_tril(dm2buf)
        dm2buf[:,diag_idx] *= .5
        dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
        casdm2 = casdm2_cc = None
     
        if atmlst is None:
            atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst),3))
     
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao = hcore_deriv(ia)
            de[k] += np.einsum('xij,ij->x', h1ao, casci_dm1,optimize=True)
            de[k] += np.einsum('xij,ij->x', h1ao, zvec_ao,optimize=True)
     
            q1 = 0
            for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, nao):
                q0, q1 = q1, q1 + nf
                dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
                shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
                eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                 shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
                de[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao,optimize=True) * 2
     
                for i in range(3):
                    eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf,-1))
                    eri1tmp = eri1tmp.reshape(p1-p0,nf,nao,nao)
                    de[k,i] -= np.einsum('ijkl,ij,kl', eri1tmp, hf_dm1[p0:p1,q0:q1],
                                         zvec_ao,optimize=True) * 2
                    de[k,i] -= np.einsum('ijkl,kl,ij', eri1tmp, hf_dm1,
                                         zvec_ao[p0:p1,q0:q1],optimize=True) * 2
                    de[k,i] += np.einsum('ijkl,il,kj', eri1tmp, hf_dm1[p0:p1],
                                         zvec_ao[q0:q1],optimize=True)
                    de[k,i] += np.einsum('ijkl,jk,il', eri1tmp, hf_dm1[q0:q1],
                                         zvec_ao[p0:p1],optimize=True)
                    
                    de[k,i] -= np.einsum('ijkl,lk,ij', eri1tmp, dm_core[q0:q1],
                                         casci_dm1[p0:p1],optimize=True) * 2
                    de[k,i] += np.einsum('ijkl,jk,il', eri1tmp, dm_core[q0:q1],
                                         casci_dm1[p0:p1],optimize=True)
                    de[k,i] -= np.einsum('ijkl,lk,ij', eri1tmp, dm_cas[q0:q1],
                                         dm_core[p0:p1],optimize=True) * 2
                    de[k,i] += np.einsum('ijkl,jk,il', eri1tmp, dm_cas[q0:q1],
                                         dm_core[p0:p1],optimize=True)
                eri1 = eri1tmp = None
     
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1],optimize=True)
            de[k] -= np.einsum('xij,ji->x', s1[:,p0:p1], im1[:,p0:p1],optimize=True)
     
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1],optimize=True) * 2
            de[k] -= np.einsum('xij,ji->x', s1[:,p0:p1], zeta[:,p0:p1],optimize=True) * 2
     
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1],optimize=True) * 2
            de[k] -= np.einsum('xij,ji->x', s1[:,p0:p1], vhf_s1occ[:,p0:p1],optimize=True) * 2
     
        return de

    def grad_elec_static(self, gamma=None, gamma2=None, ncas=None, nelecs=None,
                     atmlst=None, fci=True, **kwargs):
        mc_grad = self.mf.nuc_grad_method()
        mo_coeff = self.mf.mo_coeff
        mol = self.mf.mol
        
        if fci:    
            ncas = len(self.mf.mo_coeff)
            neleca, nelecb = mol.nelec
            nelecas = (neleca, nelecb)
            ncore = 0
            nocc = ncas
        else:
            nelecb = (nelecs-self.mf.mol.spin)//2
            neleca = nelecs - nelecb
            nelecas = (neleca, nelecb)
            ncorelec = self.mf.mol.nelectron - sum(nelecas)
            ncore = ncorelec // 2
            nocc = ncore + ncas
            
        nao, nmo = mo_coeff.shape
        nao_pair = nao * (nao+1) // 2
        mo_energy = self.mf.mo_energy
     
        mo_occ = mo_coeff[:,:nocc]
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]
        neleca, nelecb = mol.nelec
        assert (neleca == nelecb)
        orbo = mo_coeff[:,:neleca]
        orbv = mo_coeff[:,neleca:]
        
        ####
        #convert AO to MO first... 
        inv_coeff = np.linalg.inv(mo_coeff)
        gamma_mo = np.einsum('pi,ij,qj->pq', inv_coeff, gamma, inv_coeff.conj(),
                             optimize=True)
        gamma2_mo = np.einsum('ijkl,pi,qj,rk,sl->pqrs', gamma2 ,
                           inv_coeff, inv_coeff, inv_coeff, inv_coeff, optimize=True)
     
        casdm1 = gamma_mo[ncore:ncore+ncas,ncore:ncore+ncas]
        casdm2 = gamma2_mo[ncore:ncore+ncas,ncore:ncore+ncas,
                           ncore:ncore+ncas,ncore:ncore+ncas]
        #####
        
        dm_core = np.dot(mo_core, mo_core.T) * 2
        dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
        aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False)
        aapa = aapa.reshape(ncas,ncas,nmo,ncas)
        vj, vk = self.mf.get_jk(mol, (dm_core, dm_cas))
        h1 = self.mf.get_hcore()
        vhf_c = vj[0] - vk[0] * .5
        vhf_a = vj[1] - vk[1] * .5
        # Imat = h1_{pi} gamma1_{iq} + h2_{pijk} gamma_{iqkj}
        Imat = np.zeros((nmo,nmo))
        Imat[:,:nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c + vhf_a, mo_occ)) * 2
        Imat[:,ncore:nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
        Imat[:,ncore:nocc] += lib.einsum('ijpk,jikq->pq', aapa, casdm2)
        aapa = vj = vk = vhf_c = vhf_a = h1 = None
     
        ee = mo_energy[:,None] - mo_energy
     
        Imat[:ncore,ncore:neleca] = 0
        Imat[ncore:neleca,:ncore] = 0
        Imat[nocc:,neleca:nocc] = 0
        Imat[neleca:nocc,nocc:] = 0
        Imat[neleca:,:neleca] = Imat[:neleca,neleca:].T
        im1 = reduce(np.dot, (mo_coeff, Imat, mo_coeff.T))
     
        casci_dm1 = dm_core + dm_cas
        hf_dm1 = self.mf.make_rdm1(mo_coeff, self.mf.mo_occ)
        hcore_deriv = mc_grad.hcore_generator(mol)
        s1 = mc_grad.get_ovlp(mol)
     
        diag_idx = np.arange(nao)
        diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
        casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
        dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                    (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
        dm2buf = lib.pack_tril(dm2buf)
        dm2buf[:,diag_idx] *= .5
        dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
        casdm2 = casdm2_cc = None
     
        if atmlst is None:
            atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst),3))
     
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao = hcore_deriv(ia)
            de[k] += np.einsum('xij,ij->x', h1ao, casci_dm1,optimize=True)
     
            q1 = 0
            for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, nao):
                q0, q1 = q1, q1 + nf
                dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
                shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
                eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                 shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
                de[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao,optimize=True) * 2
     
                for i in range(3):
                    eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf,-1))
                    eri1tmp = eri1tmp.reshape(p1-p0,nf,nao,nao)
                    
                    de[k,i] -= np.einsum('ijkl,lk,ij', eri1tmp, dm_core[q0:q1],
                                         casci_dm1[p0:p1],optimize=True) * 2
                    de[k,i] += np.einsum('ijkl,jk,il', eri1tmp, dm_core[q0:q1],
                                         casci_dm1[p0:p1],optimize=True)
                    de[k,i] -= np.einsum('ijkl,lk,ij', eri1tmp, dm_cas[q0:q1],
                                         dm_core[p0:p1],optimize=True) * 2
                    de[k,i] += np.einsum('ijkl,jk,il', eri1tmp, dm_cas[q0:q1],
                                         dm_core[p0:p1],optimize=True)
                eri1 = eri1tmp = None
    
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1],optimize=True)
            de[k] -= np.einsum('xij,ji->x', s1[:,p0:p1], im1[:,p0:p1],optimize=True)
     
        return de
     
    def get_forces_fci(self,gamma=None,gamma2=None,ncas=None,nelecas=None,fci=True,**kwargs):
        atmlst = range(self.mf.mol.natm)
#        de = self.grad_elec(gamma=gamma, gamma2=gamma2,
        de = self.grad_elec_static(gamma=gamma, gamma2=gamma2,
           ncas=ncas, nelecs=nelecas, atmlst = atmlst, fci=fci)
        forc_t=-1.0*(de+self.mf.nuc_grad_method().grad_nuc(self.mf.mol,atmlst=atmlst)) 
        #Adding gradient of the Nuclei-nuclei repulsion!
        return forc_t


def gamma_to_gamma(*args, **kwargs):
    r""" Function two assure 1-RDM to be the predicted one.

    Returns
    -------
    gamma : ndarray
        1-RDM """
    gamma = None
    for v in args :
        if isinstance(v, np.ndarray):
            gamma = v
            break
    if gamma is None :
        for k, v in kwargs :
            if isinstance(v, np.ndarray):
                gamma = v
                break
    if gamma is None :
        raise AttributeError("Please give one numpy.ndarray for gamma.")
    return gamma

def gamma_to_rdm1e(mf, *args, **kwargs):
    r""" Function to calculate the energy density matrix (1-RDMe).

    .. math::
        \hat{D} = \sum_{i=1}^{occ} \epsilon_{i} \left|i\right> \left< i \right| \\
        \left< \mu |\hat{D}| \nu \right> = \sum_{\sigma \tau} F_{\mu \sigma} S_{\sigma \tau}^{-1} P_{\tau \mu}

    Parameters
    ----------
    mf : :obj: PySCF object
        SCF class of PySCF

    Returns
    -------
    dm1e : ndarray
        Energy density matrix """
    gamma = gamma_to_gamma(*args, **kwargs)
    s = mf.get_ovlp()
    sinv = np.linalg.inv(s)
    f = mf.get_fock(dm = gamma)
    dm1e = sinv@f@gamma
    return dm1e

def rotation2rotmat(rotation, mol, factor=-1.0, angle='zyz'):
    r""" Function to rotate the density matrix.

    Parameters
    ----------
    rotation : ndarray
        Rotation Matrix

    Returns
    -------
    rotmat : ndarray
        Rotated density matrix """
    angl = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        for n in range(nc): angl.append(l)
    angl=np.asarray(angl)
    dims = angl*2+1
    aol = np.empty(len(dims)+1, dtype=np.int32)
    aol[0] = 0
    dims.cumsum(dtype=np.int32, out=aol[1:])
    rotmat = np.eye(mol.nao)
    if np.allclose(rotation, np.eye(len(rotation))) :
        pass
    else :
        alpha, beta, gamma = Rotation.from_matrix(rotation).as_euler(angle)*factor
        for i in range(len(angl)):
            r = Dmatrix.Dmatrix(angl[i], alpha, beta, gamma, reorder_p=True)
            rotmat[aol[i]:aol[i+1],aol[i]:aol[i+1]]=r
    return rotmat

def get_atom_naos(mol):
    r""" Function to get the number of atomic orbitals for a given angular momentum.

    Parameters
    ----------
    mol : :obj: PySCF mol object
        Molecluar structure.

    Returns
    -------
    naos : ndarray
        Number of atomic orbitals. """
    angl = []
    atomids = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        ia = mol.bas_atom(ib)
        for n in range(nc):
            angl.append(l)
            atomids.append(ia)
    atomids = np.asarray(atomids)
    angl = np.asarray(angl)
    dims = angl*2+1
    naos = []
    for ia in range(mol.natm) :
        n = np.sum(dims[atomids == ia])
        naos.append(n)
    naos = np.asarray(naos)
    return naos

