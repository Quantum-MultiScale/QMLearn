import numpy as np
from scipy.interpolate import interp1d
import ase.units as units
from qmlearn.drivers.core import atoms_rmsd, atoms2bestplane
from qmlearn.io import read_images
from qmlearn.utils import tenumerate
from qmlearn.drivers.rotate import get_eckart_rotate
from sklearn.decomposition import PCA

def get_train_atoms(traj=None, nsamples=None, skip = 0, tol=0.01, direction = None, transform = True, refatoms = None):
    images = read_images(traj)
    nsteps = len(images)
    if nsamples is None : nsamples = nsteps
    # print('nsteps', nsteps, traj)
    j = skip - 1
    data=[]
    if direction is not None :
        # Because the reference atoms already changed
        transform = True
    for i in range(nsamples):
        k=j+1
        for j in range(k, nsteps):
            atoms = images[j]
            new = True
            if not transform :
                atoms.positions[:] = atoms.positions - np.mean(atoms.positions, axis = 0)
                # atoms.positions[:] = atoms.positions - atoms.get_center_of_mass()
            if len(data) == 0 :
                if direction is not None :
                    atoms = atoms2bestplane(atoms, direction = direction)
                elif refatoms is not None :
                        _, atoms = atoms_rmsd(refatoms, atoms, transform=transform)
            else :
                for ia, a in enumerate(data):
                    convert = transform if ia == 0 else False
                    rmsd, atoms = atoms_rmsd(a, atoms, transform=convert)
                    if rmsd < tol :
                        new=False
                        break
            if new:
                data.append(atoms)
                break
    j += 1
    i = len(data)
    if i < nsamples :
        print(f"WARN : Only get {len(data)} samples at {j} step. Maybe you can reduce the 'tol'.", flush = True)
    else :
        print(f'Get {len(data)} samples at {j} step.', flush = True)
    return data


class AtomsCreater(object):
    def __init__(self, modes= None, frequencies=None, energies=None, atoms=None, temperature = 300, random_seed = None,
            maximum = 1E8, types = None):
        self.temperature = temperature*units.kB
        self.modes = modes
        self.maximum = maximum
        self.atoms = atoms
        self.random_seed = random_seed
        if energies is None:
            if frequencies is None:
                raise ValueError("Please provide the vibrational frequencies or energies")
            energies = np.asarray(frequencies)*units.invcm
        self.energies = energies
        self.types = types
        #
        self._initial()

    def _initial(self):
        atoms = self.atoms
        #
        pos = PCA().fit_transform(atoms.positions)
        #print('POS: ',pos)
        if np.all(pos[:, 1] < 1E-8): # linear molecule
            nstart = 5
        else : # nonlinear molecule
            nstart = 6
        #
        indices = np.arange(nstart, 3*len(atoms))
        d = len(indices)
        if len(self.modes) == 3*len(atoms):
            self.modes = self.modes[indices]
            self.energies = self.energies[indices]
        elif len(self.modes) != d:
            if np.all(pos[:, 1] < 1E-8) and len(self.modes) == d - 1:
                print("WARN: This is a linear molecule and missing one mode.")
            # else :
                # raise ValueError("The wrong number of normal modes.")
        d = len(self.modes)
        #
        if self.types is None :
            self.types = np.zeros(d)
        else :
            self.types = self.types[-d:]
        #
        uc = units._hbar * units.m / np.sqrt(units._e * units._amu)
        emode_ev = uc**2
        mid = self.temperature*len(atoms)*2*emode_ev
        chi_mid = d*(1-2/(9*d))**3
        sigma = np.sqrt(mid/chi_mid)
        #
        self.sigma = sigma
        self.random = np.random.default_rng(self.random_seed)
        #
        for i, t in enumerate(self.types):
            if t == 2 :
                self.modes[i] = interp1d(np.asarray(self.modes[i][0]), np.asarray(self.modes[i][1]), axis = 0)
            else :
                self.modes[i] = np.asarray(self.modes[i])

    def get_new_atoms(self, sigma = None):
        sigma = sigma or self.sigma
        d = len(self.modes)
        atoms = self.atoms.copy()
        coef = self.random.normal(0, sigma, size = (d, 1))[:,0]
        for i, c in enumerate(coef):
            if self.types[i] == 0 :
                c = c/self.energies[i]
                m = c*self.modes[i]
            elif self.types[i] == 1 :
                c = np.sign(c)*abs(c)**0.5/self.energies[i]**0.25
                m = c*self.modes[i]
            elif self.types[i] == 2 :
                # For this mode, the range should [-1, 1], so set the sigma=1.0/2.0.
                c = c/sigma*1.0/2.0
                if abs(c) > 1.0 : c = 1.0*np.sign(c)
                m = self.modes[i](c)
            else :
                raise ValueError(f'{self.types[i]} type of mode not support yet')
            atoms.positions += m
        return atoms

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter <= self.maximum:
            self._iter += 1
            return self.get_new_atoms()
        else:
            raise IndexError(f'Reaches the maximum : {self.maximum}')

def build_train_atoms(images, nsamples = 30, tol = 0.01, use_reflection = False, rotate_method = 'kabsch', reorder_method='none', data = None, refatoms = None, **kwargs):
    images = iter(images)
    if data is None :
        if refatoms is not None :
            data = [refatoms.copy()]
        else :
            data=[next(images)]

    print("Start build")
    for istep, atoms in enumerate(images):
        print('istep', istep, len(data), end = '\r')
        for ia, a in enumerate(data):
            if ia == 0 :
                rmsd, atoms = atoms_rmsd(a, atoms, rmsd_cut = tol, use_reflection = use_reflection, rotate_method = rotate_method, reorder_method = reorder_method, **kwargs)
            else :
                rmsd, _ = atoms_rmsd(a, atoms, transform = False)
            if rmsd < tol : break
        else :
            data.append(atoms)
            if len(data) == nsamples: break
    print(f'Get {len(data)} samples at {istep + 1} step.', flush = True)
    return data

def build_properties(images, properties = None, refqmmol = None, qmmol_options = {}, **kwargs):
    data = {k: [] for k in properties}
    for i, atoms in tenumerate(images):
        data = append_properties(atoms, data = data, properties = properties,
                refqmmol = refqmmol, qmmol_options = qmmol_options, **kwargs)
    return data

def append_properties(atoms, data = None, properties = None, refqmmol = None, ref_qmmol=None, qmmol_options = {}, inv = False, mom = False, smearing = False, **kwargs):
    from qmlearn.drivers.mol import QMMol
    #
    if data is None :
        if properties is None : properties = ['vext', 'gamma', 'energy', 'forces', 'dipole']
        data= {k: [] for k in properties}
    properties = list(data.keys())
    #
    if isinstance(atoms, QMMol):
        qmmol = atoms
        if ref_qmmol is not None and mom:
            ref_qmmol.engine.mf.run()
            qmmol.run(properties = properties, ref_qmmol = ref_qmmol, smearing = smearing, mom=mom, **kwargs)
    else :
        if refqmmol is not None and mom is False:
            qmmol = refqmmol.duplicate(atoms, refatoms=atoms)
            qmmol.run(**kwargs)
        elif ref_qmmol is not None and mom:
            qmmol = QMMol(atoms = atoms, **qmmol_options)
            ref_qmmol.engine.mf.run()
            qmmol.run(ref_qmmol = ref_qmmol, mom = mom, smearing = smearing, **kwargs)
        else :
            qmmol = QMMol(atoms = atoms, **qmmol_options)
            qmmol.run(**kwargs)
#        qmmol.run(**kwargs)
    #
    for key in properties :
        if key == 'vext' :
            data[key].append(qmmol.engine.vext)
        elif key == 'gamma' :
            if inv:
                print("Running Vinv")
                vinv = np.linalg.inv(qmmol.engine.vext)
                data[key].append(vinv @ qmmol.engine.gamma)
            else:
                data[key].append(qmmol.engine.gamma)
        elif key == 'energy' :
            data[key].append(qmmol.engine.etotal)
        elif key == 'forces' :
            data[key].append(qmmol.engine.forces)
        elif key == 'dipole' :
            data[key].append(qmmol.calc_dipole(qmmol.engine.gamma))
        elif key == 'ke' :
            data[key].append(qmmol.calc_ke(qmmol.engine.gamma))
        elif key == 'ovlp' :
            data[key].append(qmmol.engine.ovlp)
        elif key == 'gamma2':
            if smearing:
                qmmol.engine.mf = None
            if inv:
                print("Running Vinv")
                vinv = np.linalg.inv(qmmol.engine.vext)
                gamma2_inv = np.einsum('pa,qb,abcd->pqcd',
                                        vinv, vinv, qmmol.engine.gamma2
                                        ,optimize=True)
                data[key].append(gamma2_inv)
            else:
                data[key].append(qmmol.engine.gamma2)
        elif key == 'gamma2c':
            if smearing:
                qmmol.engine.mf = None
            if inv:
                print("Running Vinv")
                vinv = np.linalg.inv(qmmol.engine.vext)
                gamma2c_inv = np.einsum('pa,qb,abcd->pqcd',
                                        vinv, vinv, qmmol.engine.gamma2c
                                        ,optimize=True)
                data[key].append(gamma2c_inv)
            else:
                data[key].append(qmmol.engine.gamma2c)
        elif key == 'gamma2cum':
            if smearing:
               qmmol.engine.mf = None
            if inv: 
               print("Running Vinv")
               vinv = np.linalg.inv(qmmol.engine.vext)
               gamma2cum_inv = np.einsum('pa,qb,abcd->pqcd',
                                       vinv, vinv, qmmol.engine.gamma2cum
                                       ,optimize=True)
               data[key].append(gamma2cum_inv)
            else:
               data[key].append(qmmol.engine.gamma2cum)
        elif key == 'delta_gamma':
            if inv:
                vinv = np.linalg.inv(qmmol.engine.vext)
                data[key].append(vinv @ qmmol.engine.delta_gamma)
            else: 
                data[key].append(qmmol.engine.delta_gamma)
        elif key == 'occ_dg':
            data[key].append(qmmol.engine.occ_dg)
        elif key == 'occ':
            data[key].append(qmmol.engine.occ)
        elif key == 'energy_c':
            data[key].append(qmmol.engine.etotal_c)
        else :
            raise ValueError(f'Sorry, not support the property {key} now')
    return data


def get_frame_images(images, refatoms = None):
    if refatoms is None :
        refatoms = images[0]
        images = images[1:]
        new_images = [refatoms.copy()]
    else :
        new_images = []
    #
    for a in images :
        rotmat = get_eckart_rotate(refatoms, a)
        a = a.copy()
        a.positions[:] = np.dot(a.positions, rotmat)
        #
        new_images.append(a)
    return new_images
