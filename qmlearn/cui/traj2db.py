#!/usr/bin/env python
# coding: utf-8
import os
import argparse
from collections import OrderedDict
from qmlearn.drivers.mol import QMMol
from qmlearn.io import read_images, write_db, merge_db
from qmlearn.preprocessing import append_properties
from qmlearn.utils import tenumerate
from qmlearn.io.hdf5 import DBHDF5

def get_args():
    parser = argparse.ArgumentParser(
            description='QMLearn traj2db:\n Create the QMLearn database from trajectory file',
            usage='use "%(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--traj2db', dest='traj2db', action='store_true', default=False,
            help='Create the QMLearn database from trajectory file')

    parser.add_argument('trajs', nargs = '+', help = 'Structure file contains train atoms. Supported format:\n'
            'traj, xyz, extxyz, hdf5')
    parser.add_argument('-o', '--output', dest='output', type=str, action='store',
            default=None, help='The output file')
    parser.add_argument('--xc', dest='xc', type=str, action='store',
            default='lda,vwn_rpa', help='exchange-correlation functional')
    parser.add_argument('--basis', dest='basis', type=str, action='store',
            default='6-31g', help='basis set for engine')
    parser.add_argument('--charge', dest='charge', type=int, action='store',
            default=0, help='Total number of electrons in the system')
    parser.add_argument('--spin', dest='spin', type=int, action='store',
            default=0, help='Spin = 2S = Nalpha - Nbeta, not 2S+1')
    parser.add_argument('--method', dest='method', type=str, action='store',
            default='rks', help='The method for the engine')
    parser.add_argument('--istart', dest='istart', type=int, action='store',
            default=0, help='The first structure in the file')
    parser.add_argument('--iend', dest='iend', type=int, action='store',
            default=None, help='The end structure in the file')
    parser.add_argument('--properties', dest='properties', nargs = '+',
            default=[], help='Other properties besides "vext", "gamma", "energy", "forces" and "dipole". Supported :\n'
            '"ke",  "ovlp"')
    parser.add_argument('--merge', dest='merge', action='store_true',
            help='If the input files are database, can merge them to one output file.')
    parser.add_argument('--ncas', dest='ncas', type=int, action='store',
                        default=2, help='N Active Space')
    parser.add_argument('--nelecas', dest='nelecas', type=int, action='store',
                        default=2, help='N Electrons')
    parser.add_argument('--nroots', dest='nroots', type=int, action='store',
                        default=1, help='Roots to be calculated')
    parser.add_argument('--rotate_method', dest='rotate_method', type=str, action='store',
                        default='kabsch', help='Rotate Method')
    parser.add_argument('--reorder_method', dest='reorder_method', type=str, action='store',
                        default='inertia-hungarian', help='Reorder Method')
    parser.add_argument('--ignore_hydrogen', action='store_true',
                        default=False, help='Ignore hydrogens (reorder) if flag is given')
    parser.add_argument('--no_reflection', dest='use_reflection',  action='store_false',
                        default=True, help='Disable reflection (default: enabled)')
    parser.add_argument('--no_stereo', dest='stereo' , action='store_false',
                        default=True, help='Disable stereo (default: enabled)')
    parser.add_argument('--ci_ref', dest='ci_ref', type=str, action='store',
                        default=None, help='Use a CI vector as Ref')
    parser.add_argument('--dm0', dest='dm0', type=str, action='store',
                        default=None, help='Use a initial guess for RDM')
    parser.add_argument('--masses', dest='masses', type=str, action='store',
                        default=None, help='List of desired atomic masses')
    parser.add_argument('--inv', dest='inv', type=bool, action='store',
                        default=False, help='If "Yes" will multiply gamma and Gamma by np.linalg.inv(vext)')
    parser.add_argument('--mom', dest='mom', type=bool, action='store',
                        default=False, help='If "Yes" MOM will be apply taking as Reference: refqmmol') 
    parser.add_argument('--smearing', dest='smearing', type=bool, action='store',
                        default=False, help='If "Yes" will smear HF')

    args = parser.parse_args()
    return args

def run(args):
    print(args)
    #-----------------------------------------------------------------------
    basis = args.basis
    xc = args.xc
    method = args.method
    charge = args.charge
    spin = args.spin
    output = args.output
    properties = args.properties
    trajs = args.trajs
    istart = args.istart
    iend = args.iend
    merge = args.merge
    ncas = args.ncas
    nelecas = args.nelecas
    nroots = args.nroots
    use_reflection = args.use_reflection
    ignore_hydrogen = args.ignore_hydrogen
    stereo = args.stereo
    rotate_method = args.rotate_method
    reorder_method = args.reorder_method
    ci_ref = args.ci_ref
    dm0 = args.dm0
    masses = args.masses
    inv = args.inv
    mom = args.mom
    smearing =  args.smearing
    #-----------------------------------------------------------------------
    trajs = list(OrderedDict.fromkeys(trajs))
    print(f'Input files are : {trajs}')
    if not output : output = os.path.splitext(trajs[0])[0]+'_qmldb.hdf5'
    if os.path.isfile(output):
        raise ValueError(f"The {output} already exist.")

    if merge :
        return merge_db(trajs, output=output)

    properties.extend(['vext', 'gamma', 'energy', 'forces', 'dipole'])
    index = slice(istart, iend)
    qmmol_options = {
            'basis' : basis,
            'xc' : xc,
            'method' : method,
            'charge' : charge,
            'spin' : spin,
            'ncas' : ncas,
            'nelecas' : nelecas,
            'nroots' : nroots,
            'use_reflection' : use_reflection,
            'ignore_hydrogen' : ignore_hydrogen,
            'stereo' : stereo,
            'rotate_method' : rotate_method,
            'reorder_method' : reorder_method,
            'ci_ref' : ci_ref,
            'dm0' : dm0,
            'masses' : masses,
            'smearing' : smearing,
            }

    #print(qmmol_options)

    format = os.path.splitext(trajs[0])[-1][1:]
    if format == 'traj' :
       train_atoms=read_images(trajs[0], index=index)
    elif format == 'hdf5' :
       db = DBHDF5(trajs[0])
       refqmmol_ao = db.read_qmmol(name=db.get_names('*/qmmol')[0])
       atoms_positions = db.read_images(name=db.get_names('*/train_atoms_*')[0])
       train_atoms=atoms_positions[istart:iend]
    data= {k: [] for k in properties}
    for i, atoms in tenumerate(train_atoms):
        if masses is not None: atoms.masses = masses
        qmmol = QMMol(atoms = atoms, **qmmol_options)
#        if reorder_method is not None: qmmol = qmmol.duplicate(atoms)
        if i == 0 : refqmmol = qmmol
        if mom:
#          print("Using MOM: ", mom)
          data = append_properties(qmmol, data = data, inv = inv, refqmmol = refqmmol, mom = mom) 
        else:
          data = append_properties(qmmol, data = data, inv = inv)

    write_db(output, refqmmol, train_atoms, data)

def main():
    args = get_args()
    return run(args)


if __name__ == "__main__":
    main()
