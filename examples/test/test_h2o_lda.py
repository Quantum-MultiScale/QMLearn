"""
Test suite for H2O LDA database (based on examples/examples/h2o/h2o_b3lyp.ipynb workflow)

This test suite:
1. Tests reading the H2O LDA database
2. Validates database structure and properties
3. Tests QMMol consistency
4. Tests QMModel creation from database
5. Tests QMLCalculator usage

Note: The database file and 'ir' directory are automatically generated during setUpClass
and deleted after testing in tearDownClass. Using LDA instead of B3LYP for faster execution.
"""
import unittest
import numpy as np
import os
import shutil
import ase.build
from pathlib import Path

from qmlearn.drivers.mol import QMMol
from qmlearn.io import read_db, write_db
from qmlearn.preprocessing import AtomsCreater, build_train_atoms, build_properties
from qmlearn.model import QMModel
from qmlearn.io.model import db2qmmodel


class H2OLDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test class with H2O molecule and database generation"""
        # Test parameters - using LDA for faster execution
        cls.basis = '6-31g*'
        cls.xc = 'lda,vwn_rpa'  # Using LDA instead of B3LYP for speed
        cls.method = 'rks'
        cls.temperature = 300
        cls.nsamples = 27
        cls.tol = 0.04
        cls.random_seed = 8888
        
        # Create test directory if it doesn't exist
        # Test file is in examples/test/, notebook is in examples/examples/h2o/
        cls.test_dir = Path(__file__).parent.parent / 'examples' / 'h2o'
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Database filename - using LDA in name
        cls.dbfile = cls.test_dir / 'h2o_300_0.040_27_lda_test_qmldb.hdf5'
        
        # Create reference molecule
        atoms = ase.build.molecule('H2O')
        cls.refqmmol = QMMol(atoms=atoms, method=cls.method, basis=cls.basis, xc=cls.xc)
        cls.atoms = atoms
        
        # Generate database if it doesn't exist
        # Note: Using LDA for faster execution (much faster than B3LYP)
        if not cls.dbfile.exists():
            print(f"Database not found. Generating database: {cls.dbfile}")
            print("Using LDA for faster execution...")
            cls._generate_database()
        else:
            print(f"Using existing database: {cls.dbfile}")

    @classmethod
    def tearDownClass(cls):
        """Clean up: delete the database file and ir directory after testing"""
        if cls.dbfile.exists():
            try:
                cls.dbfile.unlink()
                print(f"Deleted test database: {cls.dbfile}")
            except Exception as e:
                print(f"Warning: Could not delete database file {cls.dbfile}: {e}")

        # Clean up the 'ir' directory created by ASE Infrared
        ir_dir = Path('ir')
        if ir_dir.exists():
            try:
                shutil.rmtree(ir_dir)
                print(f"Deleted ir directory: {ir_dir}")
            except Exception as e:
                print(f"Warning: Could not delete ir directory {ir_dir}: {e}")

    @classmethod
    def _generate_database(cls):
        """Generate the database following the notebook workflow"""
        print(f"Generating database: {cls.dbfile}")
        
        try:
            # Optimize geometry
            from ase.optimize import LBFGS
            from qmlearn.api.api4ase import QMLCalculator
            
            atoms = cls.atoms.copy()
            refqmmol = QMMol(atoms=atoms, method=cls.method, basis=cls.basis, xc=cls.xc)
            atoms.calc = QMLCalculator(qmmodel=refqmmol, method='engine')
            optimizer = LBFGS(atoms)
            optimizer.run(fmax=1E-5)
            
            # Get vibrational modes
            from ase.vibrations import Infrared
            ir = Infrared(atoms, nfree=4)
            ir.run()
            vib = ir.get_vibrations()
            modes = vib.get_modes()
            frequencies = vib.get_frequencies().real
            
            # Create training atoms
            creater = AtomsCreater(
                modes=modes, 
                frequencies=frequencies,
                atoms=atoms, 
                temperature=cls.temperature,
                random_seed=cls.random_seed
            )
            images = build_train_atoms(creater, nsamples=cls.nsamples, tol=cls.tol, refatoms=atoms)
            
            # Build properties
            prop = ['vext', 'gamma', 'energy', 'forces', 'dipole', 'ke']
            properties = build_properties(images, refqmmol=refqmmol, properties=prop)
            
            # Write database
            write_db(str(cls.dbfile), refqmmol, images, properties)
            print(f"Database generated successfully: {cls.dbfile}")
        except Exception as e:
            print(f"Error generating database: {e}")
            raise

    def test_0_database_exists(self):
        """Test that the database file exists"""
        self.assertTrue(self.dbfile.exists(), f"Database file {self.dbfile} does not exist")

    def test_1_read_database(self):
        """Test reading the database"""
        data = read_db(str(self.dbfile), names='rks')
        
        # Check that all required keys are present
        self.assertIn('qmmol', data)
        self.assertIn('atoms', data)
        self.assertIn('properties', data)
        
        # Check qmmol
        refqmmol = data['qmmol']
        self.assertIsNotNone(refqmmol)
        self.assertEqual(refqmmol.method, self.method)
        self.assertEqual(refqmmol.basis, self.basis)
        self.assertEqual(refqmmol.xc, self.xc)
        
        # Check atoms
        train_atoms = data['atoms']
        self.assertGreater(len(train_atoms), 0)
        self.assertEqual(len(train_atoms), self.nsamples)
        
        # Check properties
        props = data['properties']
        expected_props = ['vext', 'gamma', 'energy', 'forces', 'dipole', 'ke']
        for prop in expected_props:
            self.assertIn(prop, props, f"Property {prop} not found in database")
            self.assertEqual(len(props[prop]), self.nsamples, 
                           f"Property {prop} has wrong number of samples")

    def test_2_properties_shapes(self):
        """Test that property arrays have correct shapes"""
        data = read_db(str(self.dbfile), names='rks')
        props = data['properties']
        train_atoms = data['atoms']
        refqmmol = data['qmmol']
        
        nao = refqmmol.nao
        natoms = len(train_atoms[0])
        
        # Check vext shape
        self.assertEqual(props['vext'][0].shape, (nao, nao))
        
        # Check gamma shape
        self.assertEqual(props['gamma'][0].shape, (nao, nao))
        
        # Check energy shape (should be scalar)
        self.assertTrue(np.isscalar(props['energy'][0]) or props['energy'][0].shape == ())
        
        # Check forces shape
        self.assertEqual(props['forces'][0].shape, (natoms, 3))
        
        # Check dipole shape
        self.assertEqual(props['dipole'][0].shape, (3,))
        
        # Check ke shape (kinetic energy is a scalar)
        self.assertTrue(np.isscalar(props['ke'][0]) or props['ke'][0].shape == ())

    def test_3_properties_values(self):
        """Test that property values are reasonable"""
        data = read_db(str(self.dbfile), names='rks')
        props = data['properties']
        
        # Check energy values are finite and reasonable
        energies = props['energy']
        self.assertTrue(all(np.isfinite(e) for e in energies))
        # LDA/6-31g* H2O energy should be around -76 Hartree
        self.assertTrue(all(e < -70 for e in energies))
        
        # Check forces are finite
        for forces in props['forces']:
            self.assertTrue(np.all(np.isfinite(forces)))
        
        # Check dipole moments are finite
        for dipole in props['dipole']:
            self.assertTrue(np.all(np.isfinite(dipole)))

    def test_4_qmmol_consistency(self):
        """Test that QMMol from database is consistent"""
        data = read_db(str(self.dbfile), names='rks')
        refqmmol = data['qmmol']
        train_atoms = data['atoms']
        
        # Test that we can duplicate with a training atom
        test_atom = train_atoms[0]
        qmmol_dup = refqmmol.duplicate(test_atom)
        self.assertIsNotNone(qmmol_dup)
        self.assertEqual(qmmol_dup.nao, refqmmol.nao)
        
        # Test that overlap matrix can be computed
        ovlp = refqmmol.ovlp
        self.assertEqual(ovlp.shape, (refqmmol.nao, refqmmol.nao))
        self.assertTrue(np.allclose(ovlp, ovlp.T))  # Should be symmetric

    def test_5_model_creation(self):
        """Test creating a QMModel from the database"""
        # This test may take longer, so we'll use a subset of data
        try:
            model = db2qmmodel(
                str(self.dbfile), 
                names='rks',
                index=slice(0, min(10, self.nsamples))  # Use first 10 samples for speed
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(model.refqmmol)
            self.assertIn('gamma', model.mmodels)
        except Exception as e:
            self.fail(f"Failed to create QMModel from database: {e}")

    def test_6_qml_calculator(self):
        """Test using QMLCalculator with the reference QMMol"""
        from qmlearn.api.api4ase import QMLCalculator
        
        data = read_db(str(self.dbfile), names='rks')
        refqmmol = data['qmmol']
        train_atoms = data['atoms']
        
        # Create calculator
        atoms = train_atoms[0].copy()
        calc = QMLCalculator(qmmodel=refqmmol, method='engine')
        atoms.calc = calc
        
        # Test that we can get energy
        energy = atoms.get_potential_energy()
        self.assertTrue(np.isfinite(energy))
        self.assertLess(energy, 0)  # Should be negative
        
        # Test that we can get forces
        forces = atoms.get_forces()
        self.assertEqual(forces.shape, (len(atoms), 3))
        self.assertTrue(np.all(np.isfinite(forces)))


if __name__ == "__main__":
    print("Tests for H2O LDA database")
    unittest.main()
