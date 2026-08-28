import os
import numpy as np
from ase import Atoms, io
from qmlearn.io.hdf5 import DBHDF5
from qmlearn.utils import tenumerate

def read_images(traj, format=None, index=None):
    if index is None :
        inds = slice(None)
    else :
        inds = index
    #
    if isinstance(traj[0], Atoms):
        images = traj[inds]
    else :
        if not format :
            format = os.path.splitext(traj)[-1][1:]
        if format : format = format.lower()
        if format == 'traj' :
            from ase.io.trajectory import Trajectory
            images = Trajectory(traj)
            if index: images = images[inds]
        elif format in ['xyz', 'exyz', 'extxyz'] :
            images = io.read(traj, index = inds)[inds]
        else :
            raise AttributeError(f"Sorry, not support '{format}' format.")
    return images

def read_db(filename, names = '*'):
    db = DBHDF5(filename, 'r')
    if isinstance(names, str):
        method = names
        names = {}
        l = db.get_names(method + '/qmmol*')
        if len(l)>0 : names['qmmol'] = l[0]
        l = db.get_names(method + '/train_atoms*')
        if len(l)>0 : names['atoms'] = l[0]
        l = db.get_names(method + '/train_prop*')
        if len(l)>0 : names['properties'] = l[0]
        l = db.get_names(method + '/model*')
        if len(l)>0 : names['model'] = l[0]
        print(f'Guess DB names : {names}', flush = True)
    data = {}
    for key, v in names.items() :
        if key == 'qmmol' :
            data[key] = db.read_qmmol(name = v)
        elif key == 'atoms' :
            data[key] = db.read_images(name = v)
        elif key == 'properties' :
            data[key] = db.read_properties(name = v)
        elif key == 'model' :
            data[key] = db.read_model(name = v)
        elif not key.startswith('_') :
            raise ValueError(f'The key {key} can not read from the db')
    data['_names'] = names.copy()
    db.close()
    return data

def write_db(output, qmmol=None, images=None, properties=None, model=None, prefix = 'train', names = None, mode='w', **kwargs):
    default_names = dict.fromkeys(['qmmol', 'atoms', 'properties', 'model'])
    if names is None:
        names = default_names
    else:
        default_names.update(names)
    db = DBHDF5(output, mode, qmmol=qmmol, **kwargs)
    if qmmol is None and model is not None: qmmol = model.refqmmol
    if qmmol :
        db.write_qmmol(qmmol, name = names['qmmol'], **kwargs)
    if images:
        db.write_images(images, prefix=prefix, name = names['atoms'], **kwargs)
    if properties:
        db.write_properties(properties, prefix=prefix, name = names['properties'], **kwargs)
    if model:
        db.write_model(model, prefix=prefix, name = names['model'], **kwargs)
    print('Names in database: ', db.names, flush = True)
    db.close()

def merge_db(filenames, names = '*', output = None):
    for i, filename in tenumerate(filenames):
        data = read_db(filename, names=names)
        if i == 0 :
            qmmol = data['qmmol']
            images = data['atoms']
            properties = data['properties']
            for k in properties :
                properties[k] = [properties[k]]
            if 'dipole' in properties:
                properties['dipole'][0][0, 0] = 1000.0
        else :
            images.extend(data['atoms'])
            for k in properties :
                if k not in data['properties'] :
                    raise AttributeError(f"The {filename} missing {k} property.")
                properties[k].append(data['properties'][k])
    #
    for k in properties :
        properties[k] = np.concatenate(properties[k])
    #
    data = {
            'qmmol' : qmmol,
            'atoms' : images,
            'properties' : properties,
            }
    if output is not None :
        #write_db(output, qmmol, images, properties, names = names)
        write_db(output, qmmol, images, properties)
    return data

def read_model(filename, name='*/model*'):
    data = read_db(filename, names={'model':name})
    return data['model']

def write_model(output, model=None, mode='w', **kwargs):
    write_db(output, model=model, mode=mode, **kwargs)
