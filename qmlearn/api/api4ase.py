import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from sklearn.preprocessing import StandardScaler

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
            **kwargs):
        Calculator.__init__(self, label = label, atoms = atoms, directory = directory, **kwargs)
        self.qmmodel = qmmodel
        self.qmmodel2 = qmmodel2
        self.second_learn = second_learn
        self.method = method
        self._refqmmol = refqmmol
        self._properties = properties

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
              print('False Delta Purification')
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
            forces = self.qmmodel.convert_back(forces, prop='forces_c')
            hf_forces = -1.0 * qmmol.engine.mf.nuc_grad_method().kernel()
            print('Running HF forces: \n', 'Predicted Forces C: \n' ,forces, '\n HF Forces: \n', hf_forces, '\n')
            forces += hf_forces
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
        gamma_d_ = self.qmmodel.predict(qmmol,               
                       model=self.qmmodel.mmodels['delta_gamma']).reshape(shape) 
        if self.qmmodel.purify_gamma :
             gamma_fp, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=gamma_d_)
        else:
             gamma_fp, gamma_d = qmmol.engine.purify_d_gamma(gamma_d=gamma_d_,pure=False)
#        gamma_fp = self.qmmodel.convert_back(gamma_fp_, prop='gamma')
#        gamma_d = self.qmmodel.convert_back(gamma_d__, prop='gamma')
                                                                                         
        gamma2c_ = self.qmmodel2.predict(qmmol,
                                 model=self.qmmodel2.mmodels['gamma2c']).reshape(shape2)
        if self.qmmodel.purify_Gamma :
            gamma2 , gamma2c = qmmol.engine.purify_gamma2c(gamma=gamma_fp,gamma2c=gamma2c_) 
        else:
            gamma2 , gamma2c = qmmol.engine.purify_gamma2c(gamma=gamma_fp,gamma2c=gamma2c_,pure=False)

        if 'gamma' in properties:
            self.results['gamma'] = gamma_fp
        if 'gamma2c' in properties:
            self.results['gamma2c'] = gamma2c
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

        if 'delta_gamma' in properties or 'gamma2' in properties or 'gamma2c' in properties or 'gamma' in properties: 
            gamma, gamma2, gamma2c, delta_gamma = qmmol.engine.all_gammas
            if 'delta_gamma' in properties :
                self.results['delta_gamma'] = delta_gamma
            if 'gamma2' in properties :
                self.results['gamma2'] = gamma2
            if 'gamma2c' in properties :
                self.results['gamma2c'] = gamma2c
            if 'gamma' in properties:
                self.results['gamma'] = gamma

        if 'energy' in properties:
            energy = qmmol.engine.etotal
            self.results['energy'] = energy * Ha

        if 'forces' in properties:
            forces = qmmol.engine.run_forces()
            self.results['forces'] = forces * Ha/Bohr
