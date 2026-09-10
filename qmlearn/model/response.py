import numpy as np


def vext_generator(mol):
    r""" Return a function ``vext_deriv(atm_id)`` giving the analytic AO-basis
    derivative of the external-potential (nuclear-attraction) matrix,
    :math:`d(v_{ext})_{\mu\nu}/dR_A`, shape ``(3, nao, nao)``.

    This mirrors ``pyscf.grad.rhf.hcore_generator`` exactly, but isolates the
    nuclear-attraction piece alone (dropping the kinetic-energy piece), since
    ``vext`` -- not ``h1 = kin + vext`` -- is the feature QMLearn's KRR models
    are trained on (``QMModel.translate_input`` returns ``qmmol.vext``).

    Parameters
    ----------
    mol : pyscf.gto.Mole

    Returns
    -------
    vext_deriv : callable
        ``vext_deriv(atm_id) -> ndarray, shape (3, nao, nao)``
    """
    aoslices = mol.aoslice_by_atom()
    ipnuc = -mol.intor('int1e_ipnuc', comp=3)

    def vext_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:, p0:p1] += ipnuc[:, p0:p1]
        return vrinv + vrinv.transpose(0, 2, 1)

    return vext_deriv


def kernel_directional_derivative(model, x, v):
    r""" Analytic directional derivative of a fitted
    :class:`sklearn.kernel_ridge.KernelRidge` prediction:

    .. math::
        \frac{d}{dt}\Big|_{t=0}\ \text{model.predict}(x + t v)

    Supports the ``'linear'`` and ``'rbf'`` kernels only.

    Parameters
    ----------
    model : sklearn.kernel_ridge.KernelRidge
        Already-fitted regressor.
    x : ndarray, shape (n_features,)
        Feature vector (e.g. the flattened ``vext``) at which to evaluate.
    v : ndarray, shape (n_features,)
        Direction in feature space to differentiate along (e.g. the
        flattened ``dvext/dX_A``).

    Returns
    -------
    dy : ndarray, shape (n_targets,)
        ``d(y_pred)/dt`` along direction ``v``.
    """
    x = np.asarray(x).ravel()
    v = np.asarray(v).ravel()
    X_train = model.X_fit_
    dual_coef = model.dual_coef_
    if dual_coef.ndim == 1:
        dual_coef = dual_coef[:, None]

    if model.kernel == 'linear':
        # K(x,x_i) = x.x_i  =>  dK/dx = x_i  =>  dK/dt = x_i.v
        coeff = X_train.dot(v)
    elif model.kernel == 'rbf':
        # K(x,x_i) = exp(-gamma |x-x_i|^2)
        # dK/dx = -2 gamma (x-x_i) K(x,x_i)  =>  dK/dt = -2 gamma K(x,x_i) (x-x_i).v
        gamma = model.gamma
        if gamma is None:
            gamma = 1.0 / X_train.shape[1]
        diff = x - X_train
        K = np.exp(-gamma * np.einsum('ij,ij->i', diff, diff))
        coeff = -2.0 * gamma * K * diff.dot(v)
    else:
        raise NotImplementedError(
            f"kernel_directional_derivative only supports the 'linear' and "
            f"'rbf' kernels for now, got '{model.kernel}'."
        )

    dy = coeff.dot(dual_coef)
    return dy


def grad_rdm_response(model_gamma, model_gamma2, qmmol, atmlst=None):
    r""" RDM-response term of the analytic gradient for ML-predicted RDMs:

    .. math::
        h \cdot \frac{d\gamma}{dX_A} + (2e) \cdot \frac{d\Gamma}{dX_A}

    This is what should be ADDED to the Term-1-only output of
    ``EnginePyscf.grad_elec_full_static(gamma, gamma2, pulay=False)`` (or
    ``grad_elec_static(..., pulay=False)``) -- not to their default
    ``pulay=True`` output, which already accounts for the same physics
    through the (approximate, stationarity-assuming) -W.S^(1) term.

    Parameters
    ----------
    model_gamma, model_gamma2 : qmlearn.model.QMModel
        Already-trained models predicting ``gamma``/``gamma2`` from ``vext``
        with a ``'linear'`` or ``'rbf'``-kernel ``KernelRidge`` regressor
        (``model_gamma.model``/``model_gamma2.model``).
    qmmol : qmlearn.drivers.mol.QMMol
        The molecule (geometry) at which to evaluate the response.
    atmlst : list of int, optional
        Atoms to compute the term for. Defaults to all atoms.

    Returns
    -------
    de : ndarray, shape (len(atmlst), 3)
    """
    eng = qmmol.engine
    mol = eng.mol
    nao = mol.nao_nr()
    h1 = eng.mf.get_hcore()
    eri = mol.intor('int2e', aosym='s1').reshape(nao, nao, nao, nao)

    x_gamma = np.asarray(model_gamma.translate_input(qmmol)).ravel()
    x_gamma2 = np.asarray(model_gamma2.translate_input(qmmol)).ravel()

    deriv = vext_generator(mol)

    if atmlst is None:
        atmlst = range(mol.natm)
    de = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        dv = deriv(ia)  # (3, nao, nao)
        for xi in range(3):
            v = dv[xi].ravel()
            dgamma = kernel_directional_derivative(model_gamma.model, x_gamma, v).reshape(nao, nao)
            dgamma2 = kernel_directional_derivative(model_gamma2.model, x_gamma2, v).reshape(nao, nao, nao, nao)
            de[k, xi] = (np.einsum('pq,pq->', h1, dgamma, optimize=True)
                         + 0.5 * np.einsum('pqrs,pqrs->', eri, dgamma2, optimize=True))
    return de


def grad_rdm_response_fd(func, qmmol, atmlst=None, step=1e-4):
    r""" Finite-difference version of :func:`grad_rdm_response` for an
    arbitrary ``vext -> (gamma, Gamma)`` pipeline that isn't a plain
    ``KernelRidge.predict`` call -- e.g. one that also purifies gamma
    (an iterative Fermi-level solve, not closed-form) and/or builds Gamma
    from a *different* model's cumulant prediction via the cumulant
    expansion :math:`\Gamma = \Gamma_{cum} + \tfrac12(2\gamma\otimes\gamma -
    \text{exchange})`, as :meth:`QMLCalculator.calc_with_gamma2cum2` does.

    Uses the analytic ``d(vext)/dX_A`` (:func:`vext_generator`) for the
    geometry part, and a plain central finite difference of ``func`` (cheap:
    no SCF is involved) for the model/post-processing part.

    Parameters
    ----------
    func : callable
        ``vext (ndarray, shape (nao,nao)) -> (gamma, Gamma)`` where ``gamma``
        has shape ``(nao,nao)`` and ``Gamma`` has shape ``(nao,nao,nao,nao)``.
    qmmol : qmlearn.drivers.mol.QMMol
        The molecule (geometry) at which to evaluate the response.
    atmlst : list of int, optional
        Atoms to compute the term for. Defaults to all atoms.
    step : float
        Finite-difference step in the ``vext`` feature.

    Returns
    -------
    de : ndarray, shape (len(atmlst), 3)
    """
    eng = qmmol.engine
    mol = eng.mol
    nao = mol.nao_nr()
    h1 = eng.mf.get_hcore()
    eri = mol.intor('int2e', aosym='s1').reshape(nao, nao, nao, nao)
    vext0 = np.asarray(qmmol.vext)

    deriv = vext_generator(mol)

    if atmlst is None:
        atmlst = range(mol.natm)
    de = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        dv = deriv(ia)  # (3, nao, nao)
        for xi in range(3):
            v = dv[xi]
            gp, g2p = func(vext0 + step * v)
            gm, g2m = func(vext0 - step * v)
            dgamma = (gp - gm) / (2 * step)
            dgamma2 = (g2p - g2m) / (2 * step)
            de[k, xi] = (np.einsum('pq,pq->', h1, dgamma, optimize=True)
                         + 0.5 * np.einsum('pqrs,pqrs->', eri, dgamma2, optimize=True))
    return de


def wedge(A, B):
    r""" :math:`(A \wedge B)_{pqrs} = A_{pq} B_{rs} - \tfrac12 A_{ps} B_{rq}`,
    the same bilinear form used throughout this project's cumulant expansion
    (``EnginePyscf``/``Engine.purify_gamma2cum``'s ``gamma2cum + wedge(gamma,gamma)``,
    with ``wedge(gamma,gamma) = a - 0.5*b`` there). Kept as a standalone,
    non-symmetric-in-its-arguments building block so the product rule below
    can be written directly as ``wedge(A, B) + wedge(B, A)``.
    """
    a = np.einsum('pq,rs->pqrs', A, B, optimize=True)
    b = np.einsum('ps,rq->pqrs', A, B, optimize=True)
    return a - 0.5 * b


def grad_rdm_response_cumulant(model_gamma, model_cumulant, gamma, qmmol, atmlst=None):
    r""" Analytic RDM-response term (Term 2) for the gamma+cumulant composite
    model used throughout this project's "maps[6]"-style pipeline,

    .. math::
        \Gamma = \gamma \wedge \gamma + \Gamma_{cum},
        \qquad \gamma = \gamma(v_{ext}), \quad \Gamma_{cum} = \Gamma_{cum}(v_{ext})

    predicted by two *independent* KRR models. Unlike :func:`grad_rdm_response`
    (built for a single model predicting :math:`\Gamma` directly), :math:`\Gamma`
    here depends on :math:`\gamma` **twice**: once through :math:`\Gamma_{cum}`'s
    own fit, and once through the wedge product itself, which is quadratic in
    :math:`\gamma`. The missing chain-rule piece is, by the product rule,

    .. math::
        \frac{d(\gamma\wedge\gamma)}{dX_A} =
            \frac{d\gamma}{dX_A}\wedge\gamma + \gamma\wedge\frac{d\gamma}{dX_A}

    so that

    .. math::
        \frac{d\Gamma}{dX_A} = \frac{d\Gamma_{cum}}{dX_A}
            + \frac{d\gamma}{dX_A}\wedge\gamma + \gamma\wedge\frac{d\gamma}{dX_A}

    with both :math:`d\gamma/dX_A` and :math:`d\Gamma_{cum}/dX_A` given by the
    same analytic kernel-derivative chain rule as :func:`grad_rdm_response`
    (:func:`kernel_directional_derivative` chained with :func:`vext_generator`)
    -- no finite difference anywhere, unlike :func:`grad_rdm_response_fd`.

    Parameters
    ----------
    model_gamma, model_cumulant : qmlearn.model.QMModel
        Already-trained models predicting ``gamma``/the cumulant piece (e.g.
        ``gamma2cum`` or ``gamma2c``) from ``vext``, with a ``'linear'`` or
        ``'rbf'``-kernel ``KernelRidge`` regressor.
    gamma : ndarray, shape (nao, nao)
        The (already predicted, unperturbed) ``gamma`` at ``qmmol``'s
        geometry -- reused here rather than re-predicted.
    qmmol : qmlearn.drivers.mol.QMMol
        The molecule (geometry) at which to evaluate the response.
    atmlst : list of int, optional
        Atoms to compute the term for. Defaults to all atoms.

    Returns
    -------
    de : ndarray, shape (len(atmlst), 3)
    """
    eng = qmmol.engine
    mol = eng.mol
    nao = mol.nao_nr()
    h1 = eng.mf.get_hcore()
    eri = mol.intor('int2e', aosym='s1').reshape(nao, nao, nao, nao)

    x_gamma = np.asarray(model_gamma.translate_input(qmmol)).ravel()
    x_cum = np.asarray(model_cumulant.translate_input(qmmol)).ravel()

    deriv = vext_generator(mol)

    if atmlst is None:
        atmlst = range(mol.natm)
    de = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        dv = deriv(ia)  # (3, nao, nao)
        for xi in range(3):
            v = dv[xi].ravel()
            dgamma = kernel_directional_derivative(model_gamma.model, x_gamma, v).reshape(nao, nao)
            dcum = kernel_directional_derivative(model_cumulant.model, x_cum, v).reshape(nao, nao, nao, nao)
            dgamma2 = dcum + wedge(dgamma, gamma) + wedge(gamma, dgamma)
            de[k, xi] = (np.einsum('pq,pq->', h1, dgamma, optimize=True)
                         + 0.5 * np.einsum('pqrs,pqrs->', eri, dgamma2, optimize=True))
    return de
