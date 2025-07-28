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
    args = parser.parse_args()
    return args

def run(args):
    print(args)
    #-----------------------------------------------------------------------
    basis = args.basis
    xc = args.xc
    method = args.method
    charge = args.charge
    output = args.output
    properties = args.properties
    trajs = args.trajs
    istart = args.istart
    iend = args.iend
    merge = args.merge
    ncas = args.ncas
    nelecas = args.nelecas
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
            'ncas' : ncas,
            'nelecas' : nelecas,
            }

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
        qmmol = QMMol(atoms = atoms, **qmmol_options)
        if i == 0 : refqmmol = qmmol
        data = append_properties(qmmol, data = data)

    write_db(output, refqmmol, train_atoms, data)

def main():
    args = get_args()
    return run(args)


if __name__ == "__main__":
    main()
