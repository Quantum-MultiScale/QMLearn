import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import ase
import ase.build

from qmlearn.drivers.mol import QMMol

class Reorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atoms = ase.build.molecule('H2O')
        atoms.positions[0] += 0.05
        basis = 'cc-pvTZ'
        xc = 'lda,vwn_rpa'
        method = 'rks'
        charge = 0
        qmmol = QMMol(atoms = atoms, method = method, basis=basis, xc = xc, charge=charge, reorder_method='inertia-hungarian')
        cls.qmmol = qmmol
        cls.qmmol.run()

    def test_0_reorder(self):
        indices = [2, 0, 1]
        self.check_operate(indices=indices)

    def test_1_rotate(self):
        angle = [0.2, -0.3, 0.4]
        rotation = Rotation.from_euler('zyz', angle).as_matrix()
        self.check_operate(rotation=rotation)

    def check_operate(self, indices = None, rotation = None):
        qmmol = self.qmmol
        gamma = qmmol.engine.gamma
        if indices is None : indices = slice(None)
        if rotation is None : rotation = np.eye(3)
        atoms2 = qmmol.atoms[indices]
        atoms2.positions[:] = atoms2.positions[:] @ rotation
        #
        qmmol2_ori = qmmol.duplicate(atoms2, refatoms=atoms2)
        qmmol2_ori.run()
        gamma2_ori = qmmol2_ori.engine.gamma
        #
        qmmol2 = qmmol.duplicate(atoms2)
        qmmol2.run()
        gamma2 = qmmol2.engine.gamma
        #
        self.assertTrue(np.allclose(gamma, gamma2))
        #
        gamma2_back = qmmol2.convert_back(gamma2, prop='gamma')
        self.assertTrue(np.allclose(gamma2_back, gamma2_ori, atol = 1E-5))
        #
        e0 = qmmol2_ori.calc_etotal(gamma2_ori)
        e1 = qmmol2.calc_etotal(gamma2)
        e2 = qmmol2_ori.calc_etotal(gamma2_back)
        print(e0, e1, e2)
        #
        self.assertAlmostEqual(e1, e0, 5)
        self.assertAlmostEqual(e2, e0, 5)
        f0 = qmmol.engine.forces
        f1 = qmmol2_ori.engine.forces
        f2 = qmmol2.engine.forces
        f2_back = qmmol2.convert_back(f2, prop='forces')
        #
        print(f2 - f0)
        print(f2_back - f1)
        self.assertTrue(np.allclose(f2, f0, atol = 1E-3))
        self.assertTrue(np.allclose(f2_back, f1, atol = 1E-3))


if __name__ == "__main__":
    print("Tests for Reorder")
    unittest.main()
