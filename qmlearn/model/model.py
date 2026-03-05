import numpy as np
from ase import Atoms
import scipy.linalg as sla
# from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from qmlearn.drivers.mol import QMMol


class QMModel(object):
    r""" QMModel class is wrapper around sklearn regression model classes. It provide method to
    fit and predict based on provided training and testing data

    Attributes
    ----------
    mmodels : dict, optional
        Set of machine learning algorithms used for training , If not provided
        by default Kernel Ridge Rigression (KRR) will be used to learn :math:`{\gamma}` from :math:`V_{ext}` and
        linear regression to learn :math:`{\delta}E`, and :math:`{\delta}{\gamma}`

    method : {'gamma'}, str
        Determine which property to learn from external potentials :math:`V_{ext}`. As of now only
        :math:`{\gamma}` can be learned from :math:`V_{ext}`. And then all other properties
        calculated from :math:`{\gamma}`

    ncharge: int, optional
        Total number of electrons.

    nspin: int, optional
        Total spin as defined in PySCF (num. alpha electrons - num. beta electrons).

    refqmmol: QMMol object
        Reference QMMol object
    """
    def __init__(self, mmodels = None, method='gamma', ncharge=None, nspin = 1, occs = None,
            refqmmol = None, purify_gamma = True, purify_Gamma = True, purify_method=None, **kwargs):
        self.init_kwargs = locals()
        self.init_kwargs.pop('self', None)
        self.init_kwargs.pop('kwargs', None)
        self.init_kwargs.update(kwargs)

        self._method = method
        self.mmodels = mmodels or {}
        self.purify_gamma = purify_gamma
        self.purify_Gamma = purify_Gamma
        self.purify_method = purify_method
        #
        if self.method not in self.mmodels :
            self.mmodels[self.method] = KernelRidge(alpha=0.1,kernel='linear')
        # for orbital
        self.nspin = nspin
        self.ncharge = ncharge
        self.occs = occs
        if self.occs is None and self.ncharge :
            if self.nspin == 1 :
                self.occs = np.ones(self.ncharge//2)*2
            elif self.nspin == 2 :
                self.occs = np.ones(self.ncharge)
        #
        self._refqmmol = refqmmol
        self.qmmol = None

    @property
    def model(self):
        return self.mmodels[self.method]

    @model.setter
    def model(self, value):
        self.mmodels[self.method] = value

    @property
    def refqmmol(self):
        if self._refqmmol is None :
            raise AttributeError("Please set the 'refqmmol' before the predict.")
        return self._refqmmol

    @refqmmol.setter
    def refqmmol(self, value):
        self._refqmmol = value

    @property
    def method(self):
        if self._method.lower() not in {'gamma','gamma2c','gamma2','gamma2cum',
                                        'occ','eigs_gammat','eigs_gammatc',
                                        'occ_dg','delta_gamma','gamma_00','gamma_01','gamma_10','gamma_11'} :
            raise AttributeError("""Only support 'gamma', 'gamma2c', 'gamma2', 'gamma2cum', 'occ',
                                  'eigs_gammat', 'eigs_gammatc', 'occ_dg' or 'delta_gamma'
                                  'gamma_00','gamma_01','gamma_10','gamma_11', method now.""")
        return self._method

    def fit(self, X, y, model = None, method = None):
        r""" Fit model

        Parameters
        ----------
        X : array
            Training data
        y : array
            Target values

        model : QMModel obj, optional
            Regression model (i.e., KernelRidge, LinearRegression), If not provided will take from
            mmodels dictionary.
        method : {'gamma', 'd_gamma', 'd_energy', 'd_forces'}

            | 'gamma' -> learn :math:`{\delta\gamma}` using :math:`V_{ext}`
            | 'd_gamma' -> learn :math:`{\delta\gamma}` based on predicted :math:`{\gamma}`.
            | 'd_energy' -> learn :math:`{\delta}E` based on predicted :math:`{\gamma}`
            | 'd_forces' -> learn :math:`{\delta}F` based on predicted :math:`{\gamma}`

        Returns
        -------
        QMModel
            return trained QMModel object
        """
        if model is None :
            method = method or self.method
            model = self.mmodels[method]
        #
        if isinstance(X[0], np.ndarray):
            if X[0].ndim > 1 :
                X = [item.ravel() for item in X]
        if isinstance(y[0], np.ndarray):
            if y[0].ndim > 1 :
                y = [item.ravel() for item in y]
        
        #
        model.fit(X, y)
        return model

    def predict(self, x, model = None, method = None, **kwargs):
        r""" Predict using trained QMModel

        Parameters
        ----------
        x : array
            Training data

       method : {'gamma', 'd_gamma', 'd_energy', 'd_forces'}
            | 'gamma' -> predict :math:`{\delta\gamma}` using :math:`V_{ext}`
            | 'd_gamma' -> predict :math:`{\gamma}+{\delta\gamma}` based on predicted :math:`{\gamma}`.
            | 'd_energy' -> predict :math:`E+{\delta}E` based on predicted :math:`{\gamma}`
            | 'd_forces' -> predict :math:`F+{\delta}F` based on predicted :math:`{\gamma}`

        Returns
        -------
        y : array
            Predicted target values
        """
        #x = [self.translate_input(x, **kwargs).ravel()]
        x = self.translate_input(x, **kwargs).reshape((1,-1))
        if model is None :
            method = method or self.method
            model = self.mmodels[method]
        #
        y = model.predict(x)[0]
        #
        return y

    def convert_back(self, y, prop = 'gamma', qmmol = None, **kwargs):
        """Convert back the predicted properties to the original reference frame of the molecule

        Parameters
        ----------
        prop : str
            Define property to be converted back to original reference frame
        qmmol : None
            Default is to refer to qmmol object previously defined

        Returns
        -------
        y : array
            Predicted target values in original refrence frame of the molecule
        """
        qmmol = qmmol or self.qmmol
        y = qmmol.convert_back(y, prop = prop, **kwargs)
        return y

    def translate_input(self, x, **kwargs):
        """Return external potential :math:`V_{ext}` from x. x could be numpy ndarray, ASE Atoms object,
         or QMMol object
        """
        if isinstance(x, np.ndarray) :
            out = x
        elif isinstance(x, Atoms) :
            x = self.refqmmol.duplicate(x, **kwargs)
        elif isinstance(x, QMMol):
            pass
        else :
            raise AttributeError("Please check the input.")

        if isinstance(x, QMMol):
            self.qmmol = x
            out = x.vext
        return out

    def orth_orb(self, s, orb):
        s_new = np.einsum('mi,nj,mn->ij', orb, orb, s)
        w,vr=sla.eig(s_new)
        w=1./np.real(np.sqrt(w))
        s_moh=vr@np.diag(w)@vr.T
        oorb = np.einsum('ij,mj->mi',s_moh,orb)
        s_new = np.einsum('mi,nj,mn->ij', oorb, oorb, s)
        return oorb

    def write(self, output, mode='w', **kwargs):
        from qmlearn.io import write_model
        write_model(output, model=self, mode=mode, **kwargs)


class MQMModel(QMModel):
    def __init__(self, mmodels = None, method='gamma', ncharge=None, nspin = 1, occs = None, refqmmol = None,
            fragments = None, qmmodels = None, dft_fragments = False, **kwargs):
        super().__init__(mmodels=mmodels, method=method, ncharge=ncharge, nspin=nspin, occs=occs, refqmmol=refqmmol, **kwargs)
        self._fragments = fragments
        self._qmmodels = qmmodels
        self._dft_fragments = dft_fragments

    @property
    def qmmodels(self):
        if self._qmmodels is None :
            raise AttributeError("Please set the 'qmmodels' before the predict.")
        return self._qmmodels

    @qmmodels.setter
    def qmmodels(self, value):
        self._qmmodels = value

    @property
    def fragments(self):
        if self._fragments is None :
            raise AttributeError("Please set the 'fragments' before the predict.")
        return self._fragments

    @fragments.setter
    def fragments(self, value):
        self._fragments = value

    @property
    def dft_fragments(self):
        return self._dft_fragments

    @dft_fragments.setter
    def dft_fragments(self, value):
        self._dft_fragments = value

    def translate_input(self, x, convert = True, offdiag = False, **kwargs):
        if isinstance(x, np.ndarray) :
            out = x
        elif isinstance(x, Atoms) :
            x = self.refqmmol.duplicate(x, **kwargs)
        elif isinstance(x, QMMol):
            pass
        else :
            raise AttributeError("Please check the input.")

        if isinstance(x, QMMol):
            self.qmmol = x
            out = self.qmmol.vext
            block = self.get_block_vext(x, convert = convert)
            if offdiag :
                out = out - sla.block_diag(*block)
        return out

    def get_block_vext(self, qmmol, convert = True, **kwargs):
        if isinstance(qmmol, Atoms) :
            qmmol = self.refqmmol.duplicate(qmmol, **kwargs)
        self.sub_qmmols = []
        block = []
        for i, index in enumerate(self.fragments):
            frag = qmmol.atoms[index]
            if self.dft_fragments :
                if hasattr(self.qmmodels[i], 'refqmmol'):
                    refqmmol = self.qmmodels[i].refqmmol
                else :
                    refqmmol = self.qmmodels[i]
                a = refqmmol.duplicate(frag, refatoms=frag, **kwargs)
                v = a.vext
            else :
                a = self.qmmodels[i].refqmmol.duplicate(frag, **kwargs)
                if convert :
                    v = a.convert_back(a.vext, prop = 'gamma')
                else :
                    v = a.vext
            self.sub_qmmols.append(a)
            block.append(v)
        return block

    def predict(self, x, method = None, convert=True, split=False, **kwargs):
        x = self.translate_input(x, convert = convert, **kwargs).ravel()
        method = method or self.method
        model = self.mmodels[method]
        y = model.predict([x])[0]
        y = np.asarray(y)
        if 'gamma' in method :
            block = self.predict_block_gamma(method=method, convert=convert)
            y_frags = sla.block_diag(*block)
            if y.size > 1 : y = y.reshape(y_frags.shape)
            if split:
                y = (y_frags, y)
            else :
                y = y_frags + y
        return y

    def predict_block_gamma(self, x=None, method=None, convert =True, **kwargs):
        if x is not None :
            # This one just to make sure the 'sub_qmmols' were calculated.
            x = self.translate_input(x, convert = convert, **kwargs).ravel()
        method = method or self.method
        block = []
        for i, mol in enumerate(self.sub_qmmols) :
            if self.dft_fragments :
                y0 = mol.engine.gamma
            else :
                x0 = mol.vext
                y0 = self.qmmodels[i].predict(x0, method=method)
                y0 = y0.reshape(x0.shape)
                mol.gamma = y0
                if convert :
                    y0 = mol.convert_back(y0, prop = method)
            block.append(y0)
        return block

    def predict_diff_v1(self, x, method = None, convert=True, split=False, **kwargs):
        x = self.translate_input(x, convert = convert, **kwargs).ravel()
        method = method or self.method
        model = self.mmodels[method]
        y = model.predict([x])[0]
        y = np.asarray(y)
        block = self.predict_block_v1(method=method, convert=convert)
        if 'gamma' in method :
            y_frags = sla.block_diag(*block)
        elif 'force' in method :
            y_frags = np.vstack(block)
        elif y.size == 1 :
            y_frags = np.sum(block)
        else :
            raise AttributeError("'MQMModel' only for 'energy', 'force' and 'gamma' now.")
        if y.size > 1 : y = y.reshape(y_frags.shape)

        if split:
            return y_frags, y
        else:
            return y + y_frags

    def predict_block_v1(self, x=None, method=None, convert =True, **kwargs):
        if x is not None :
            # This one just to make sure the 'sub_qmmols' were calculated.
            x = self.translate_input(x, convert = convert, **kwargs).ravel()
        method = method or self.method
        block = []
        for i, mol in enumerate(self.sub_qmmols) :
            if method == 'gamma' :
                x0 = mol.vext
            else :
                x0 = mol.gamma
            y0 = self.qmmodels[i].predict(x0, method=method)
            if np.size(y0) == np.size(x0) :
                y0 = y0.reshape(x0.shape)
                if method == 'gamma' : # save the gamma
                    mol.gamma = y0
            if convert :
                y0 = mol.convert_back(y0, prop = method)
            block.append(y0)
        return block
