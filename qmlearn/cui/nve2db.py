#!/usr/bin/env python
# coding: utf-8
"""
nve2db: Convert NVE CASSCF MD trajectory data to a QMLearn HDF5 database.

Reads md.traj (ASE trajectory) and the .npy dump files produced by the NVE
MD script (energies, gradients, dm_ao), subsamples diverse frames by RMSD,
applies Kabsch alignment, rotates dm_ao into the aligned frame, and writes
the result to a QMLearn HDF5 database.

The output is directly compatible with the standard QMLearn training pipeline.
"""
import os
import argparse
import numpy as np

from ase.io.trajectory import Trajectory

from qmlearn.drivers.mol import QMMol
from qmlearn.drivers.core import atoms_rmsd, minimize_rmsd_operation
from qmlearn.io import write_db


# ---------------------------------------------------------------------------
# Load npy data
# ---------------------------------------------------------------------------

def load_npy_data(filehead):
    """Load the .npy dump files produced by the NVE MD script.

    Returns
    -------
    energies  : dict{step: float}   (Hartree)
    gradients : dict{step: ndarray} (Hartree/Bohr, raw nuclear gradient dE/dR)
    dm_ao     : dict{step: ndarray} (AO-basis 1-RDM)
    """
    energies_raw  = np.load(f"{filehead}_energies.npy",  allow_pickle=True).item()
    gradients_raw = np.load(f"{filehead}_gradients.npy", allow_pickle=True).item()
    dm_raw        = np.load(f"{filehead}_dm.npy",        allow_pickle=True).item()

    energies  = {int(s): float(v["casscf"])       for s, v in energies_raw.items()}
    gradients = {int(s): np.array(v["gradients"]) for s, v in gradients_raw.items()}
    dm_ao     = {int(s): np.array(v["dm_ao"])     for s, v in dm_raw.items()}

    return energies, gradients, dm_ao


# ---------------------------------------------------------------------------
# Geometry subsampling
# ---------------------------------------------------------------------------

def nve_build_train_atoms(traj_path, nsamples=30, tol=0.01, refatoms=None,
                          rotate_method='kabsch', reorder_method='none',
                          use_reflection=False, **kwargs):
    """Read md.traj, deduplicate by RMSD, and return Kabsch-aligned images.

    Mirrors build_train_atoms but also records the traj step index and the
    3x3 Kabsch rotation matrix for each selected frame (needed to rotate dm_ao).

    Returns
    -------
    images         : list of ASE Atoms  (Kabsch-aligned; refatoms is images[0])
    selected_steps : list of int        (-1 for the refatoms entry)
    op_rotates     : list of ndarray    (3x3 rotation per frame; identity for refatoms)
    """
    traj_list = list(Trajectory(traj_path))

    if refatoms is not None:
        data = [refatoms.copy()]
        selected_steps = [-1]
        op_rotates = [np.eye(3)]
        iter_start = 0
    else:
        data = [traj_list[0].copy()]
        selected_steps = [0]
        op_rotates = [np.eye(3)]
        iter_start = 1

    rmsd_kwargs = dict(
        rotate_method=rotate_method,
        reorder_method=reorder_method,
        use_reflection=use_reflection,
        **kwargs,
    )

    for traj_step, atoms in enumerate(traj_list[iter_start:], start=iter_start):
        atoms_orig = atoms.copy()

        for ia, a in enumerate(data):
            if ia == 0:
                op_rotate, _, _ = minimize_rmsd_operation(a, atoms_orig, **rmsd_kwargs)
                rmsd, atoms = atoms_rmsd(a, atoms, rmsd_cut=tol, **rmsd_kwargs)
            else:
                rmsd, _ = atoms_rmsd(a, atoms, transform=False)
            if rmsd < tol:
                break
        else:
            data.append(atoms)
            selected_steps.append(traj_step)
            op_rotates.append(op_rotate)
            if len(data) == nsamples:
                break

    print(f"Selected {len(data)} frames from trajectory.", flush=True)
    return data, selected_steps, op_rotates


# ---------------------------------------------------------------------------
# Property collection from npy
# ---------------------------------------------------------------------------

def nve_build_properties_from_npy(images, selected_steps, op_rotates,
                                   energies, gradients, dm_ao,
                                   refqmmol):
    """Build the QMLearn data dict from precomputed npy values.

    dm_ao for each frame is rotated into the Kabsch-aligned AO frame via
    Wigner D-matrix rotation so it is consistent with the stored geometry.

    The refatoms frame (step == -1) is handled by running refqmmol once.
    """
    from pyscf import gto
    from pyscf.pbc.tools.pyscf_ase import atoms_from_ase

    properties = ['vext', 'gamma', 'energy', 'forces']
    data = {k: [] for k in properties}

    refqmmol.run(properties=properties)

    for atoms, step, op_rotate in zip(images, selected_steps, op_rotates):
        if step == -1:
            data['vext'].append(refqmmol.engine.vext)
            data['gamma'].append(refqmmol.engine.gamma)
            data['energy'].append(refqmmol.engine.etotal)
            data['forces'].append(refqmmol.engine.forces)
            continue

        # vext: nuclear attraction integrals for the aligned geometry (no SCF)
        mol_frame = gto.M(
            atom=atoms_from_ase(atoms),
            basis=refqmmol.basis,
            charge=refqmmol.init_kwargs.get('charge', 0),
            spin=refqmmol.init_kwargs.get('spin', 0),
            unit='Angstrom',
            verbose=0,
        )
        data['vext'].append(mol_frame.intor_symmetric('int1e_nuc'))

        # gamma: rotate dm_ao into the Kabsch-aligned AO frame
        dm = dm_ao[step]
        if not np.allclose(op_rotate, np.eye(3)):
            rotmat = refqmmol.engine.rotation2rotmat(op_rotate, factor=-1.0, angle='zyz')
            dm = rotmat.T @ dm @ rotmat
        data['gamma'].append(dm)

        # energy (Hartree) and forces = -gradient (Hartree/Bohr)
        data['energy'].append(energies[step])
        data['forces'].append(-np.array(gradients[step]))

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='nve2db: NVE CASSCF MD trajectory -> QMLearn HDF5 database',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('traj', help='ASE trajectory file (md.traj)')
    parser.add_argument('filehead', help='Prefix for .npy dump files (e.g. H2O_CAS_NVE)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HDF5 file (default: <traj_stem>_qmldb.hdf5)')
    parser.add_argument('--refatoms', default=None,
                        help='Reference geometry xyz file. If omitted, first traj frame is used.')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--tol', type=float, default=0.01,
                        help='RMSD deduplication tolerance in Angstrom')
    parser.add_argument('--basis', default='6-31g')
    parser.add_argument('--method', default='casscf')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--spin', type=int, default=0)
    parser.add_argument('--ncas', type=int, default=2)
    parser.add_argument('--nelecas', type=int, default=2)
    parser.add_argument('--nroots', type=int, default=1)
    parser.add_argument('--rotate_method', default='kabsch')
    parser.add_argument('--reorder_method', default='none')
    parser.add_argument('--use_reflection', action='store_true', default=False)
    return parser.parse_args()


def run(args):
    print(args, flush=True)

    output = args.output or os.path.splitext(args.traj)[0] + '_qmldb.hdf5'
    if os.path.isfile(output):
        raise ValueError(f"Output file already exists: {output}")

    if args.refatoms is not None:
        from ase.io import read as ase_read
        refatoms = ase_read(args.refatoms)
    else:
        refatoms = list(Trajectory(args.traj))[0].copy()

    refqmmol = QMMol(
        atoms=refatoms,
        basis=args.basis,
        method=args.method,
        charge=args.charge,
        spin=args.spin,
        ncas=args.ncas,
        nelecas=args.nelecas,
        nroots=args.nroots,
        rotate_method=args.rotate_method,
        reorder_method=args.reorder_method,
        use_reflection=args.use_reflection,
    )

    images, selected_steps, op_rotates = nve_build_train_atoms(
        args.traj,
        nsamples=args.nsamples,
        tol=args.tol,
        refatoms=refatoms,
        rotate_method=args.rotate_method,
        reorder_method=args.reorder_method,
        use_reflection=args.use_reflection,
    )

    energies, gradients, dm_ao = load_npy_data(args.filehead)

    data = nve_build_properties_from_npy(
        images, selected_steps, op_rotates,
        energies, gradients, dm_ao,
        refqmmol,
    )

    write_db(output, refqmmol, images, data)
    print(f"Wrote {len(images)} frames to {output}", flush=True)


def main():
    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
