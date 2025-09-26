from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from qmlearn.model.model import QMModel
from qmlearn.io import read_db
from qmlearn.utils import tenumerate
from sklearn.preprocessing import StandardScaler

def db2qmmodel(filename, names = '*', mmodels = None, qmmol_options = None, purify_gamma = True,
        predicted_gamma = True, index =None, target='gamma', method='gamma', purify_method = 'smearing'):
    r"""Train QMModel to learn :math:`{\gamma}` in terms of :math:`V_{ext}` from training data
    then an additional layer of training learn :math:`{\delta}E` and :math:`{\delta}{\gamma}`
    based on previously learned :math:`{\gamma}`.

    Parameters
    ----------
    filename : str
        Name of database file
    names : str, optional
        Name of database, by default '*'
    mmodels : dict, optional
        Set of machine learning models used for training. If not provided
        by default KKR will be used to learn gamma and linear regression for
        delta learning
    index : float, optional
        Allows user to define how many training points to use for a given molecule. Preferred: index = slice(0, n_samples, selection)
    predicted_gamma : bool, optional
        If True, will use gamma values stored in QMLearn database. Else, will recalculate gamma.
    purify_gamma : bool, optional
        If True, will purify gamma by rotating and reordering

    Returns
    -------
    model : obj
        trained model
    """
    if index is None : index = slice(None)
    if isinstance(filename, dict):
        data = filename
    else :
        data = read_db(filename, names=names)
    refqmmol = data['qmmol']
    if qmmol_options :
        refqmmol.init_kwargs.update(qmmol_options)
        refqmmol = refqmmol.__class__(**refqmmol.init_kwargs)
        print("New options of QMMOL :\n", refqmmol.init_kwargs)
    train_atoms = data['atoms'][index]
    properties = data['properties']
    if len(train_atoms) < len(data['atoms']):
        print(f"Only use {len(train_atoms)}/{len(data['atoms'])} of training set.")
    #
    X = properties['vext'][index]
    y = properties[target][index]
    #
    if mmodels is None :
        mmodels={
            target: KernelRidge(alpha=0.1,kernel='linear'),
            'd_gamma': LinearRegression(),
            'd_energy': LinearRegression(),
            'd_forces': LinearRegression(),
        }
        print(f'Guess mmodels: {mmodels}', flush = True)
    model = QMModel(mmodels=mmodels, refqmmol = refqmmol, purify_gamma = purify_gamma, method=method)
    model.fit(X, y)
    #
    for k in mmodels :
        if k.startswith('d_'):
            delta_learn = True
            break
    else :
        delta_learn = False
    #
    if delta_learn :
        print('Start predicting...', flush = True)
        shape = y[0].shape
        if 'gamma_pp' in properties and predicted_gamma:
            gammas = properties['gamma_pp'][index]
        elif 'delta_gamma' in model.mmodels :
            gammas = []
            gammas_c = []
            for i, a in tenumerate(train_atoms):
                gamma_d_ = model.predict(a, refatoms=a, model=model.mmodels['delta_gamma']).reshape(shape)
                gamma, gamma_d = model.qmmol.engine.purify_d_gamma(gamma_d=gamma_d_,method=purify_method,pure=purify_gamma)
                gammas.append(gamma)
                gammas_c.append(gamma_d)
            properties['gamma_pp'] = gammas
        else :
            gammas = []
            for i, a in tenumerate(train_atoms):
                gamma = model.predict(a, refatoms=a).reshape(shape)
                if model.purify_gamma :
                    gamma = model.qmmol.purify_gamma(gamma,method=purify_method)
                gammas.append(gamma)
            properties['gamma_pp'] = gammas
#        y = gammas
        print('Start delta learning...', flush = True)
        for k in mmodels :
            if not k.startswith('d_') : continue
            key = k[2:]
            if key not in properties :
                print(f"!WARN : '{key}' not in the database", flush = True)
            if key[-2:] == '_c':
                print(f'Delta learning $\gamma_c$ {key}')
                y = gammas_c
                model.fit(y, properties[key][index], method = k)
            else:
                print(f'Delta learning $\gamma$ {key}')
                y = gammas
                model.fit(y, properties[key][index], method = k)
    print('Finish the reading.', flush = True)
    return model
