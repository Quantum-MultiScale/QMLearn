import numpy as np
from pyscf import gto, scf, mcscf
from scipy.spatial.transform import Rotation

# Water geometry
geom = [
    ('H', (0.0,  0.795, -0.454)),
    ('H', (0.0, -0.795, -0.454)),
    ('O', (0.0,  0.0,    0.113)),
]

def make_mol(atom_list):
    return gto.M(
        atom=atom_list,
        basis='cc-pVDZ',
        charge=0, spin=0,
        unit='Angstrom',
        verbose=0,
    )

def run_casscf(mol, mo_coeff=None):
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcscf.CASSCF(mf, 8, 4)
    mc.kernel(mo_coeff)
    return mc.e_tot, mc.mo_coeff, mf

# --- Original geometry ---
mol1 = make_mol(geom)
e1, mo1, mf1 = run_casscf(mol1)
print(f"Original energy:  {e1:.10f} Ha")

# --- Rotate geometry by a random rotation ---
R = Rotation.random(random_state=42).as_matrix()  # random 3x3 rotation

geom_rot = [(sym, tuple(R @ np.array(pos))) for sym, pos in geom]
mol2 = make_mol(geom_rot)

# Rotate MO coefficients: C' = C @ R_ao where R_ao is the AO rotation matrix
# For a quick test, just use the default MO guess (no MO rotation) —
# CASSCF should still converge to the same energy
e2_noguess, _, _ = run_casscf(mol2)
print(f"Rotated (no MO guess): {e2_noguess:.10f} Ha")

# --- Now use rotated MOs as initial guess ---
# The AO rotation matrix in the Wigner D sense is what rotation2rotmat computes.
# For a simple sanity check, use the overlap-projected MOs instead:
from pyscf import addons
mo2_proj = addons.project_mo_nr2nr(mol1, mo1, mol2)
e2_withguess, _, _ = run_casscf(mol2, mo_coeff=mo2_proj)
print(f"Rotated (projected MO guess): {e2_withguess:.10f} Ha")

print(f"\nMax energy difference: {max(abs(e1-e2_noguess), abs(e1-e2_withguess)):.2e} Ha")
