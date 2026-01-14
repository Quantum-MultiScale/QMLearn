import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import ase
import ase.build

from qmlearn.drivers.mol import QMMol
from qmlearn.drivers.core import minimize_rmsd_operation, atoms_rmsd

class Water(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energy = -76.09381930372805
        atoms = ase.build.molecule('H2O')
        basis = 'cc-pvTZ'
        xc = 'lda,vwn_rpa'
        method = 'rks'
        charge = 0
        qmmol = QMMol(atoms = atoms, method = method, basis=basis, xc = xc, charge=charge)
        cls.atoms = atoms
        cls.qmmol = qmmol

    def test_0_qmmol(self):
        qmmol = self.qmmol
        qmmol.run()
        etotal = qmmol.engine.etotal
        self.assertAlmostEqual(etotal, self.energy, 12)

    def test_1_rmsd_rotation(self):
        atoms = ase.build.molecule('H2O')
        angle = [0.2, -0.3, 0.4]
        rotation = Rotation.from_euler('zyz', angle).as_matrix()
        atoms2 = atoms.copy()
        atoms2.positions[:] = atoms2.positions @ rotation
        rotation2, _, _= minimize_rmsd_operation(atoms2, atoms)
        self.assertTrue(np.allclose(np.abs(rotation), np.abs(rotation2)))
        rmsd, _ = atoms_rmsd(atoms, atoms2)
        self.assertAlmostEqual(rmsd, 0.0, 12)

    def test_rotmat(self):
        from scipy.spatial.transform import Rotation
        atoms = ase.build.molecule('H2O')
        angle = [0.2, -0.3, 0.4]
        rotation = Rotation.from_euler('zyz', angle).as_matrix()
        atoms2 = atoms.copy()
        atoms2.positions[:] = atoms2.positions @ rotation
        rotation2, _, _ = minimize_rmsd_operation(atoms, atoms2)
        #
        qmmol = self.qmmol
        s = qmmol.ovlp
        #
        qmmol2_ori = qmmol.duplicate(atoms2, refatoms = atoms2)
        s2_ori = qmmol2_ori.ovlp
        #
        qmmol2 = qmmol.duplicate(atoms2)
        s2 = qmmol2.ovlp
        self.assertTrue(np.allclose(s, s2))
        #
        rotmat = qmmol2.rotation2rotmat(qmmol2.op_rotate, factor=-1.0)
        s2_back = rotmat.T @ s2 @ rotmat
        self.assertTrue(np.allclose(s2_back, s2_ori))


if __name__ == "__main__":
    print("Tests for rotate")
    unittest.main()
