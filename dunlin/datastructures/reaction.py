from numbers import Number
from typing import Union

import dunlin.utils as ut
from dunlin.datastructures.bases import DataDict, DataValue

class Reaction(DataValue):
    ###########################################################################
    #Preprocessing
    ###########################################################################
    @classmethod
    def equation2stoich(cls, equation):
        #An example equation is a + 2*b -> c
        #Split the reaction
        try:
            rcts, prds = equation.split('->')
            rcts       = rcts.strip()
            prds       = prds.strip()
        except:
            msg = 'Invalid reaction. The expected format is <reactants> -> <products>.'
            msg = f'{msg} Received: {equation}'
            raise ValueError(msg)
        
        #Get the stoichiometry
        if rcts:
            rcts_stoich = [cls.get_stoich(chunk, invert_sign=True) for chunk in rcts.split('+')]
        else:
            rcts_stoich = []
        
        if prds:
            prds_stoich = [cls.get_stoich(chunk, invert_sign=False) for chunk in prds.split('+')]
        else:
            prds_stoich = []
            
        rcts = []
        prds = []
        stoich = {}
        for species, coeff in rcts_stoich:
            rcts.append(species)
            stoich[species] = coeff
        
        for species, coeff in prds_stoich:
            prds.append(species)
            stoich[species] = coeff
            
        return stoich, rcts, prds
        
    @staticmethod
    def get_stoich(rct: str, invert_sign: bool =False) -> tuple[str, str]:
        rct_ = rct.strip().split('*')

        if len(rct_) == 1:
            n, x = 1, rct_[0].strip()
        else:
            try:
                n, x = rct_[0].strip(), rct_[1].strip()
                    
            except Exception as e:
                raise e
                
            if ut.str2num(n) < 0:
                msg = 'Invalid stoichiometry. Coefficient must be positive.'
                msg = f'{msg} Received: {n}'
                raise ValueError(msg)
                    
        if invert_sign:
            return x, f'-{n}'
        else:
            return x, f'+{n}'
    
    @staticmethod
    def get_rxn_rate(rate: Union[str, Number]) -> tuple[str, set]:
        rate = str(rate).strip()
        
        if not rate:
            msg = 'Invalid reaction rate. .'
            msg = f'{msg} Received: {rate}'
            raise ValueError(msg)
        
        namespace = ut.get_namespace(rate)
        
        return rate, namespace
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 all_names : set, 
                 name      : str, 
                 equation  : str, 
                 rate      : str, 
                 bounds    : list[Number, Number]=None
                 ) -> None:
        
        #An example equation is a + 2*b -> c
        stoich, rcts, prds = self.equation2stoich(equation)
        
        #Parse the reaction rates
        rate, rxn_namespace = self.get_rxn_rate(rate)
        
        #Check namespaces
        undefined = rxn_namespace.difference(all_names)
        if undefined:
            raise NameError(f'Undefined namespace: {undefined}.')
        
        #Parse the bounds
        bounds_ = None if bounds is None else tuple(bounds)
        
        #It is now safe to call the parent's init
        super().__init__(all_names, 
                         name, 
                         equation      = equation,
                         stoichiometry = stoich,
                         rate          = str(rate),
                         reactants     = frozenset(rcts),
                         products      = frozenset(prds),
                         states        = frozenset(rcts+prds),
                         bounds        = bounds_
                         )
        
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        dct = {'equation': self.equation,
               'rate' : ut.try2num(self.rate)
               }
        
        if self.bounds:
            dct['bounds'] = list(self.bounds)
            
        dct = {self.name: dct}
        return dct
    
class ReactionDict(DataDict):  
    itype = Reaction
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, all_names: set, reactions: dict) -> None:
        #Make the dict
        super().__init__(all_names, reactions)
        
        states = set()
        
        for rxn in self.values():
            states.update(list(rxn.stoichiometry))
        
        #Save attributes
        self.states    = frozenset(states)
        