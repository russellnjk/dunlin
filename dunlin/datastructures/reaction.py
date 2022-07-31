from typing import Optional

import dunlin.utils                    as ut
import dunlin.datastructures.exception as exc
import dunlin.standardfile.dunl.writedunl as wd
from dunlin.utils.typing         import Bnd, OStr
from dunlin.datastructures.bases import _ADict, _AItem

class Reaction(_AItem):
    ###########################################################################
    #Preprocessing
    ###########################################################################
    @classmethod
    def eqn2stoich(cls, eqn):
        #An example eqn is a + 2*b -> c
        #Split the reaction
        try:
            rcts, prds = eqn.split('->')
            rcts = rcts.strip()
            prds = prds.strip()
        except:
            raise exc.InvalidDefinition('reaction equation', '<reactants> -> <products>', eqn)
        
        #Get the stoichiometry
        if rcts:
            rcts_stoich = [cls.get_stoich(chunk, invert_sign=True) for chunk in rcts.split('+')]
        else:
            rcts_stoich = []
        
        if prds:
            prds_stoich = [cls.get_stoich(chunk, invert_sign=False) for chunk in prds.split('+')]
        else:
            prds_stoich = []
            
        stoich      = dict(rcts_stoich + prds_stoich)
        
        return stoich
        
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
                    raise exc.InvalidDefinition('stoichiometry', received=n)
                    
        if invert_sign:
            return x, f'-{n}'
        else:
            return x, f'+{n}'
    
    @staticmethod
    def get_rxn_rate(fwd: str, rev: OStr) -> tuple[str, set]:
        fwd = str(fwd)
        rev = None if rev is None else str(rev)
        try:
            rate   = fwd.strip() + ' - ' + rev.strip() if rev else fwd.strip()
            
            if not rate:
                raise Exception()
                
        except:
            raise exc.InvalidDefinition('reaction rate', received=rate)
        
        variables = ut.get_eqn_variables(rate)
        
        return rate, variables
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, name: str, eqn: str, 
                 fwd: str, rev: OStr = None, 
                 bounds: Optional[Bnd]=None,
                 compartment: OStr = None
                 ):
        # #An example eqn is a + 2*b -> c
        # #Split the reaction
        # try:
        #     rcts, prds = eqn.split('->')
        #     rcts = rcts.strip()
        #     prds = prds.strip()
        # except:
        #     raise exc.InvalidDefinition('reaction equation', '<reactants> -> <products>', eqn)
        
        # #Get the stoichiometry
        # if rcts:
        #     rcts_stoich = [self.get_stoich(chunk, invert_sign=True) for chunk in rcts.split('+')]
        # else:
        #     rcts_stoich = []
        
        # if prds:
        #     prds_stoich = [self.get_stoich(chunk, invert_sign=False) for chunk in prds.split('+')]
        # else:
        #     prds_stoich = []
            
        # stoich      = dict(rcts_stoich + prds_stoich)
        
        stoich = self.eqn2stoich(eqn)
        
        #Parse the reaction rates
        rate, rxn_variables = self.get_rxn_rate(fwd, rev)
        rxn_variables.update(stoich)
        
        #Get namespace
        fwd_namespace = ut.get_namespace(fwd)
        rev_namespace = ut.get_namespace(rev)
        eqn_namespace = set(stoich)
        
        namespace = set.union(fwd_namespace, rev_namespace, eqn_namespace)
        
        if ext_namespace is not None:
            undefined = namespace.difference(ext_namespace)
            if undefined:
                raise NameError(f'Undefined namespace: {undefined}.')

        #Check bounds
        if bounds is not None:
            lb, ub = bounds
            if lb > ub:
                raise ValueError(f'Lower bound is more than upper bound: {bounds}')
            
            bounds = lb, ub
        
        #It is now safe to call the parent's init
        super().__init__(ext_namespace, name)
        
        #Store attributes
        self.eqn            = eqn
        self._stoichiometry = stoich
        self.rate           = rate
        self.namespace      = tuple(namespace)
        self.fwd_namespace  = tuple(fwd_namespace)
        self.rev_namespace  = tuple(rev_namespace)
        self.eqn_namespace  = tuple(eqn_namespace)
        self.bounds         = bounds
        self.fwd            = str(fwd)
        self.rev            = None if rev is None else str(rev)
        
        #Freeze
        self.freeze()
        
    ###########################################################################
    #Access
    ###########################################################################
    @property
    def stoichiometry(self) -> dict[str, str]:
        return self._stoichiometry
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_dict(self) -> dict:
        #Needs to be changed for export
        dct = {}
        for attr in ['eqn', 'fwd', 'rev', 'bounds']:
            value = getattr(self, attr)
            
            if value is not None:
                value = ut.try2num(value)
                
                if type(value) == tuple:
                    value = list(value)
                    
                dct[attr] = value
                
        return dct

class ReactionDict(_ADict):  
    itype = Reaction
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, reactions: dict, ext_namespace: set):
        namespace     = set()
        
        def callback(name, value):
            namespace.update(value.namespace)

        #Make the dict
        super().__init__(reactions, ext_namespace, callback)
        
        #Save attributes
        self.namespace = tuple(namespace)
        
        #Freeze
        self.freeze()
    