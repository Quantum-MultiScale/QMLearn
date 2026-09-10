import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from sklearn.preprocessing import StandardScaler
from qmlearn.model.response import grad_rdm_response_fd, grad_rdm_response_cumulant

class QMLCalculator(Calculator):
    r""" Mean QML calculator

    Attributes
    ----------
    qmmodel : QMMol object
        Reference QMMol object

    method : str
        Options

        | 'gamma' : Use QMLearn learning proccess to predict the desire property.
        | 'engine' : Use PySCF engine to predict the desire property.

    properties : list:str
        Options

        | 'energy' : Energy
        | 'forces' : Forces
        | 'dipole' : Dipole
        | 'stress' : Stress
        | 'gamma' : 1-RDM

    """
    implemented_properties = ['energy', 'forces', 'dipole', 'stress', 'gamma', 'gamma2']

    def __init__(self, qmmodel = None, qmmodel2=None, second_learn = {}, method = 'gamma',
            label='QMLearn', atoms=None, directory='.', refqmmol = None, properties = ('energy', ),
            force_method = 'cphf', **kwargs):
        r"""
        force_method : str
            Only used by :meth:`calc_with_gamma2cum2`. Options

            | 'cphf' (A) : full CPHF/Z-vector response (``get_forces_fci(fci=False)``
              for 'casci', or the ``fci=True`` default for 'fci'/'ccsd') --
              the previous, default behaviour.
            | 'static' (B) : ``get_forces_fci(fci=True)``, i.e. ``grad_elec_static``.
            | 'naive_pulay' (C/E) : ``get_forces_fci(fci=True, full_static=True)``,
              i.e. ``grad_elec_full_static`` with its default ``pulay=True``.
            | 'no_response' (D) : Term 1 only, ``grad_elec_full_static(pulay=False)``
              -- Hellmann-Feynman + the two-electron term, no correction at all.
            | 'ml_response' (F) : Term 1 (``pulay=False``) + Term 2, the RDM-response
              term ``h.(dgamma/dX_A) + (2e).(dGamma/dX_A)`` computed by
              finite-differencing the predict+purify+cumulant-expansion
              pipeline itself (:func:`qmlearn.model.response.grad_rdm_response_fd`).

            All five route through :meth:`EnginePyscf.get_forces_fci`'s
            ``pulay``/``response_de`` kwargs, so the nuclear-repulsion term is
            added exactly once, the same way, for every option.
        """
        Calculator.__init__(self, label = label, atoms = atoms, directory = directory, **kwargs)
        self.qmmodel = qmmodel
        self.qmmodel2 = qmmodel2
        self.second_learn = second_learn
        self.method = method
        self._refqmmol = refqmmol
        self._properties = properties
        self.force_method = force_method

    @property
    def refqmmol(self):
        r"""QMMol object. """
        if self._refqmmol is None :
            if hasattr(self.qmmodel, 'refqmmol'):
                return self.qmmodel.refqmmol
            else :
                # qmmodel is refqmmol
                return self.qmmodel
        return self._refqmmol

    @refqmmol.setter
    def refqmmol(self, value):
        self._refqmmol = value

    @property
    def properties(self):
        if not isinstance(self._properties, set):
            if isinstance(self._properties, str):
                self._properties = {self._properties,}
            else :
                self._properties = set(self._properties)
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value

    def calculate(self, atoms=None, properties=('energy', ), system_changes=all_changes):
        r""" Function to calculate the desire properties.

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | Dipole : 'dipole'
            | Stress : 'stress'
            | 1-RDM : 'gamma'
        """
        properties = set(properties) | self.properties
        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)
        atoms = atoms or self.atoms
        self.results['stress'] = np.zeros(6)
        if self.method == 'engine' :
            qmmol = self.refqmmol.duplicate(atoms, refatoms=atoms)
            self.calc_with_engine(qmmol, properties=properties)
        elif self.method == 'engine2' :
            qmmol = self.refqmmol.duplicate(atoms, refatoms=atoms)
            self.calc_with_engine2(qmmol,properties=properties)
        else :
            if self.method == 'gamma' :
                qmmol = self.refqmmol.duplicate(atoms.copy())
                self.calc_with_gamma(qmmol, properties=properties)
            elif self.method == 'gamma2':
                qmmol = self.refqmmol.duplicate(atoms.copy())
                self.calc_with_gamma2(qmmol, properties=properties)
            elif self.method == 'gamma2cum':
                qmmol =  self.refqmmol.duplicate(atoms.copy())
                self.calc_with_gamma2cum(qmmol, properties=properties)
            elif self.method == 'gamma2cum2':
                qmmol =  self.refqmmol.duplicate(atoms.copy())
                self.calc_with_gamma2cum2(qmmol, properties=properties)
            else :
                raise AttributeError(f"Sorry, not support '{self.method}' now.")

    def calc_with_gamma(self, qmmol, properties = ['energy'],purify_method='smearing'):
        r""" Function to calculate the desire properties using QMLearn learning process.

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | Dipole : 'dipole'
            | Stress : 'stress'
            | 1-RDM : 'gamma'
        """
        shape = self.qmmodel.refqmmol.vext.shape
        if self.qmmodel.method == 'delta_gamma':
            gamma_d_ = self.qmmodel.predict(qmmol,model=self.qmmodel.mmodels['delta_gamma']).reshape(shape)
            if self.qmmodel.purify_gamma :
              gamma, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=gamma_d_)
            else:
              gamma, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=gamma_d_,pure=False)
        else:
            gamma = self.qmmodel.predict(qmmol).reshape(shape)

        if self.qmmodel.purify_gamma :
            gamma = qmmol.purify_gamma(gamma,method=purify_method)

        m2 = self.second_learn.get('gamma', None)
        if m2 and not self.qmmodel.method == 'delta_gamma':
            gamma2 = self.qmmodel.predict(gamma, method = m2).reshape(shape)
            if self.qmmodel.purify_gamma :
                gamma2 = qmmol.purify_gamma(gamma2,method=purify_method)
        else :
            gamma2 = gamma

        if 'energy_c' in properties :
            m2 = self.second_learn.get('energy_c', None)
            if m2 :
                energy = self.qmmodel.predict(gamma_d,method=m2)
                hf_energy = qmmol.engine.mf.e_tot
                print('Running HF energies: \n',  'Predicted E C: \n' ,energy, '\n HF Energy: \n', hf_energy, '\n')
                self.results['energy'] = (energy+hf_energy) * Ha
  
        if 'forces_c' in properties:
            m2 = self.second_learn.get('forces_c', None)
            if m2 :
                forces = self.qmmodel.predict(gamma_d,method=m2)
#                forces /= 1000
            forces = self.qmmodel.convert_back(forces, prop='forces')
            hf_forces = -1.0 * qmmol.engine.mf.nuc_grad_method().kernel()
            print('Running HF forces: \n', 'Predicted Forces C: \n' ,forces, '\n HF Forces: \n', hf_forces, '\n')
            forces += hf_forces
            print('Total Forces: C+HF', forces)
            self.results['forces'] = forces * Ha/Bohr

        if 'energy' in properties and not 'energy_c' in properties:
            m2 = self.second_learn.get('energy', None)
            if m2 :
                energy = self.qmmodel.predict(gamma, method=m2)
                self.results['energy'] = energy * Ha
            else :
                energy = qmmol.calc_etotal(gamma2)
                self.results['energy'] = energy * Ha

        if 'forces' in properties and not 'forces_c' in properties:
            m2 = self.second_learn.get('forces', None)
            if m2 :
                forces = self.qmmodel.predict(gamma, method=m2)
                print('Running Predicted Forces: \n')
            else :
                print('Getting Energies from Predicted gamma \n')
                forces = qmmol.calc_forces(gamma2)
            forces = self.qmmodel.convert_back(forces, prop='forces')
            print(forces, '\n')
            self.results['forces'] = forces * Ha/Bohr
            # forces_shift = np.mean(self.results['forces'], axis = 0)
            # print('Forces shift :', forces_shift, flush = True)
            # self.results['forces'] -= forces_shift

        if 'dipole' in properties :
            m2 = self.second_learn.get('dipole', None)
            if m2 :
                dipole = self.qmmodel.predict(gamma, method=m2)
            else :
                dipole = qmmol.calc_dipole(gamma2)
            dipole = np.dot(dipole, qmmol.op_rotate_inv)
            self.results['dipole'] = dipole * Bohr

        if 'stress' in properties:
            self.results['stress'] = np.zeros(6)

        if 'gamma' in properties :
            if self.qmmodel.method == 'delta_gamma':
                gamma = gamma2 
            else:
                gamma = self.qmmodel.convert_back(gamma2, prop='gamma')
            self.results['gamma'] = gamma

    def calc_with_gamma2(self,qmmol, properties = ('energy')):
        r""" Function to calculate the desire properties using QMLearn learning process with gamma2.
                                                                                 
        Parameters                                                                       
        ----------                                                                       
        properties : list:str                                                            
            Options                                                                      
                                                                                 
            | Energy : 'energy'                                                          
            | Forces : 'forces'                                                          
            | 1-RDM : 'gamma'                                                            
            | 2-RDM : 'gamma2'
            | \delta_1RDM : 'delta_gamma'
            | Correlated 2-RDM: 'gamma2c'

        """                  
        shape = self.qmmodel.refqmmol.vext.shape
        shape2 = (shape[0],) * 4
        
        gamma2c_ = self.qmmodel.predict(qmmol,
                                 model=self.qmmodel.mmodels['gamma2c']).reshape(shape2)

        delta_gamma = qmmol.engine.gamma1_f_gamma2(gamma2c_)

        gamma_fp, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=delta_gamma,pure=True)
        
        gamma2 , gamma2c = qmmol.engine.purify_gamma2c(gamma=gamma_fp,gamma2c=gamma2c_,pure=True)

        if 'gamma' in properties:
            self.results['gamma'] = gamma_fp
        if 'gamma2c' in properties:
            self.results['gamma2c'] = gamma2c
        if 'gamma2' in properties:
            self.results['gamma2'] = gamma2

        if 'forces' in properties:
            m2 = self.second_learn.get('forces', None)
            print(m2)
            if m2 :
                forces = self.qmmodel.predict(gamma_d,method=m2,
                                              model=self.qmmodel.mmodels['d_forces'])
                forces = self.qmmodel.convert_back(forces, prop='forces')
            else:
                if qmmol.method == 'fci':
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2)
                elif qmmol.method == 'ccsd':
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2)
                elif qmmol.method == 'casci':
                    ncas = self.qmmodel.refqmmol.engine_options['ncas']
                    nelecas = self.qmmodel.refqmmol.engine_options['nelecas']
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2,
                                                    ncas=ncas,nelecas=nelecas,fci=False)
                forces = self.qmmodel.convert_back(forces, prop='forces')
            self.results['forces'] = forces* Ha/Bohr
            print('Forces: ',forces* Ha/Bohr)

        if 'energy' in properties:  
            energy = qmmol.engine.calc_etotal2(gamma=gamma_fp, gamma2=gamma2,
                                       ao_repr=True)
            self.results['energy'] = energy * Ha


    def calc_with_gamma2cum(self,qmmol, properties = ('energy')):
        r""" Function to calculate the desire properties using QMLearn learning process with gamma2 cum.
                                                                                 
        Parameters                                                                       
        ----------                                                                       
        properties : list:str                                                            
            Options                                                                      
                                                                                 
            | Energy : 'energy'                                                          
            | Forces : 'forces'                                                          
            | 1-RDM : 'gamma'                                                            
            | 2-RDM : 'gamma2'
            | \delta_1RDM : 'delta_gamma'
            | Cumulant 2-RDM: 'gamma2cum'

        """
        qmmol.engine.mf.run()  # need mo_coeff/mo_energy for get_forces_fci below
        shape = self.qmmodel.refqmmol.vext.shape
        shape2 = (shape[0],) * 4
        gamma_d_ = self.qmmodel.predict(qmmol,
                       model=self.qmmodel.mmodels['delta_gamma']).reshape(shape)
        if self.qmmodel.purify_gamma :
             gamma_fp, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=gamma_d_)
        else:
             gamma_fp, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=gamma_d_,pure=False)

        gamma2cum_ = self.qmmodel2.predict(qmmol,
                                 model=self.qmmodel2.mmodels['gamma2cum']).reshape(shape2)

        if self.qmmodel.purify_Gamma:
            gamma2, gamma2cum = qmmol.engine.purify_gamma2c(gamma=gamma_fp,gamma2c=gamma2cum_,
                                                                gamma_hf=gamma_fp)
        else:
            gamma2, gamma2cum = qmmol.engine.purify_gamma2c(gamma=gamma_fp,gamma2c=gamma2cum_,
                                                                gamma_hf=gamma_fp, pure=False)
        if 'gamma' in properties:
            self.results['gamma'] = gamma_fp
        if 'gamma2cum' in properties:
            self.results['gamma2cum'] = gamma2cum
        if 'delta_gamma' in properties:
            self.results['delta_gamma'] = gamma_d
        if 'gamma2' in properties:
            self.results['gamma2'] = gamma2

        if 'forces' in properties:
            m2 = self.second_learn.get('forces', None)
            print(m2)
            if m2 :
                forces = self.qmmodel.predict(gamma_d,method=m2,
                                              model=self.qmmodel.mmodels['d_forces'])
                forces = self.qmmodel.convert_back(forces, prop='forces')
            else:
                if qmmol.method == 'fci':
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2)
                elif qmmol.method == 'ccsd':
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2)
                elif qmmol.method == 'casci':
                    ncas = self.qmmodel.refqmmol.engine_options['ncas']
                    nelecas = self.qmmodel.refqmmol.engine_options['nelecas']
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2,
                                                    ncas=ncas,nelecas=nelecas,fci=False)
                forces = self.qmmodel.convert_back(forces, prop='forces')
            self.results['forces'] = forces* Ha/Bohr
            print('Forces: ',forces* Ha/Bohr)

        if 'energy' in properties:
            energy = qmmol.engine.calc_etotal2(gamma=gamma_fp, gamma2cum=gamma2cum,
                                       ao_repr=True, cum=True)
            self.results['energy'] = energy * Ha


    def calc_with_gamma2cum2(self,qmmol, properties = ('energy')):
        r""" Function to calculate the desire properties using QMLearn learning process with gamma2 cum (Second type).

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | 1-RDM : 'gamma'
            | 2-RDM : 'gamma2'
            | \delta_1RDM : 'delta_gamma'
            | Cumulant 2-RDM: 'gamma2cum'

        """
        if self.force_method in ('cphf', 'static'):
            # grad_elec/grad_elec_static (used by these two) need converged
            # mo_coeff/mo_energy. The other force_method options all route
            # through grad_elec_full_static, which only needs mf.get_hcore()/
            # get_ovlp() and AO integrals -- no SCF has to be run for those.
            qmmol.engine.mf.run()
        shape = self.qmmodel.refqmmol.vext.shape
        shape2 = (shape[0],) * 4

        def predict_gamma_gamma2(x):
            r""" x = qmmol (the real geometry, for the actual prediction below)
            or a raw AO vext array (only used by grad_rdm_response_fd to
            finite-difference the response term) -> (gamma, Gamma, gamma2cum).
            self.qmmodel.predict/self.qmmodel2.predict already handle either
            input via QMModel.translate_input -- exactly like
            calc_with_gamma2cum's gamma_d_ = self.qmmodel.predict(qmmol, ...). """
            gamma_ = self.qmmodel.predict(x,
                           model=self.qmmodel.mmodels['gamma']).reshape(shape)
            if self.qmmodel.purify_gamma :
                gamma_fp_ = qmmol.engine.purify_gamma(gamma_, method='smearing')
            else:
                gamma_fp_ = gamma_
            gamma2cum_ = self.qmmodel2.predict(x,
                                     model=self.qmmodel2.mmodels['gamma2cum']).reshape(shape2)
            gamma2_, _ = qmmol.engine.purify_gamma2cum(gamma=gamma_fp_, gamma2cum=gamma2cum_, pure=False)
            return gamma_fp_, gamma2_, gamma2cum_

        gamma_fp, gamma2, gamma2cum = predict_gamma_gamma2(qmmol)

        if 'gamma' in properties:
            self.results['gamma'] = gamma_fp
        if 'gamma2cum' in properties:
            self.results['gamma2cum'] = gamma2cum
        if 'gamma2' in properties:
            self.results['gamma2'] = gamma2

        if 'forces' in properties:

            m2 = self.second_learn.get('forces', None)
            if m2 :
                raise AttributeError("gamma2cum2 does not support second_learn forces / d_forces.")

            # All five options below go through get_forces_fci, which adds
            # grad_nuc and flips the sign exactly once, the same way, for
            # every method (see EnginePyscf.get_forces_fci's pulay/response_de
            # kwargs) -- avoids hand-rolling that arithmetic per branch.
            if self.force_method == 'static':
                # B: grad_elec_static
                forces = qmmol.engine.get_forces_fci(gamma=gamma_fp, gamma2=gamma2, fci=True)
            elif self.force_method == 'no_response':
                # D: Term 1 only (Hellmann-Feynman), grad_elec_full_static(pulay=False)
                forces = qmmol.engine.get_forces_fci(gamma=gamma_fp, gamma2=gamma2,
                                                fci=True, full_static=True, pulay=False)
            elif self.force_method in ('naive_pulay', 'full_static'):
                # C/E: Term 1 + naive Pulay, grad_elec_full_static(pulay=True)
                forces = qmmol.engine.get_forces_fci(gamma=gamma_fp, gamma2=gamma2,
                                                fci=True, full_static=True, pulay=True)
            elif self.force_method == 'ml_response':
                # F: Term 1 + the RDM-response term. Analytic (via the
                # gamma+cumulant wedge-product chain rule) when gamma isn't
                # purified; falls back to finite difference when it is,
                # since purify_gamma's Fermi-level solve has no simple
                # closed-form Jacobian.
                if self.qmmodel.purify_gamma:
                    term2 = grad_rdm_response_fd(lambda v: predict_gamma_gamma2(v)[:2], qmmol)
                else:
                    term2 = grad_rdm_response_cumulant(self.qmmodel, self.qmmodel2, gamma_fp, qmmol)
                forces = qmmol.engine.get_forces_fci(gamma=gamma_fp, gamma2=gamma2, fci=True,
                                                full_static=True, pulay=False, response_de=term2)
            elif self.force_method == 'cphf':
                # A: full CPHF/Z-vector, grad_elec. 'fci' is already exactly
                # orbital-invariant (fci=True/grad_elec_static suffices), but
                # 'ccsd' is not -- it genuinely needs the Z-vector solve
                # (fci=False; ncas/nelecas default to the full space when not
                # given, since CCSD's RDMs already span the whole basis).
                if qmmol.method == 'fci':
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2)
                elif qmmol.method == 'ccsd':
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2,fci=False)
                elif qmmol.method == 'casci':
                    ncas = self.qmmodel.refqmmol.engine_options['ncas']
                    nelecas = self.qmmodel.refqmmol.engine_options['nelecas']
                    forces = qmmol.engine.get_forces_fci(gamma=gamma_fp,gamma2=gamma2,
                                                    ncas=ncas,nelecas=nelecas,fci=False)
                else :
                    raise AttributeError(f"Sorry, not support the '{qmmol.method}' method now")
            else:
                raise AttributeError(f"Sorry, not support force_method='{self.force_method}' now"
                                      f" (options: 'cphf', 'static', 'naive_pulay', 'no_response', "
                                      f"'ml_response').")
            forces = self.qmmodel.convert_back(forces, prop='forces')
            self.results['forces'] = forces* Ha/Bohr
            print('Forces: ',forces* Ha/Bohr)

        if 'energy' in properties:
            energy = qmmol.engine.calc_etotal2(gamma=gamma_fp, gamma2cum=gamma2cum,
                                       ao_repr=True, cum=True)
            self.results['energy'] = energy * Ha


    def calc_with_engine(self, qmmol, properties = ('energy')):
        r""" Function to calculate the desire properties using PySCF engine.

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | Dipole : 'dipole'
            | Stress : 'stress'
            | 1-RDM : 'gamma'
        """
        qmmol.engine.run(properties = properties)
        if 'energy' in properties :
            energy = qmmol.engine.etotal
            self.results['energy'] = energy * Ha
        if 'forces' in properties:
            forces = qmmol.engine.forces
            self.results['forces'] = forces * Ha/Bohr
        if 'stress' in properties:
            self.results['stress'] = np.zeros(6)
        if 'dipole' in properties :
            dipole = qmmol.calc_dipole(qmmol.engine.gamma)
            self.results['dipole'] = dipole * Bohr
        if 'gamma' in properties :
            self.results['gamma'] = qmmol.engine.gamma

    def calc_with_engine2(self, qmmol, properties = ('energy')):
        r""" Function to calculate the desire properties using PySCF engine from 1RDM and 2RDM.
                                                                   
        Parameters
        ----------
        properties : list:str   
            Options         

            | Energy : 'energy'                          
            | Forces : 'forces'                                                           
            | 1-RDM : 'gamma'                                                             
            | 2-RDM : 'gamma2'                                                            
            | \delta_1RDM : 'delta_gamma'                                                 
            | Correlated 2-RDM: 'gamma2c'                                                 
        """                                     
        print('RUNNING ENGINE', properties)
        qmmol.engine.run(properties = properties)
        if 'delta_gamma' in properties:
            delta_gamma = qmmol.engine.delta_gamma
            self.results['delta_gamma'] = delta_gamma
        if 'gamma2' in properties:
            gamma2 = qmmol.engine.gamma2
            self.results['gamma2'] = gamma2
        if 'gamma2c' in properties:
            gamma2c = qmmol.engine.gamma2c
            self.results['gamma2c'] = gamma2c
        if 'gamma' in properties: 
            gamma  = qmmol.engine.gamma
            self.results['gamma'] = gamma
        if 'gamma2cum' in properties: 
            gamma2cum  = qmmol.engine.gamma2cum
            self.results['gamma2cum'] = gamma2cum

        if 'energy' in properties:
            energy = qmmol.engine.etotal
            self.results['energy'] = energy * Ha

        if 'forces' in properties:
            forces = qmmol.engine.run_forces()
            self.results['forces'] = forces * Ha/Bohr
