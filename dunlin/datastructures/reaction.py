import dunlin.utils                    as ut
import dunlin.datastructures.exception as exc
from dunlin.datastructures.bases import NamespaceDict, GenericItem

class Reaction(GenericItem):
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
                    raise exc.InvalidDefinition('stoichiometry', received=n)
                    
        if invert_sign:
            return x, f'-{n}'
        else:
            return x, f'+{n}'
    
    @staticmethod
    def get_rxn_rate(fwd: str, rev: str=None) -> tuple[str, set]:
        fwd = str(fwd)
        rev = None if rev is None else str(rev)
        try:
            rate   = fwd.strip() + ' - ' + rev.strip() if rev else fwd.strip()
            
            if not rate:
                raise Exception()
                
        except:
            raise exc.InvalidDefinition('reaction rate', received=rate)
        
        variables = ut.get_namespace(rate)
        
        return rate, variables
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, 
                 ext_namespace: set, 
                 name         : str, 
                 eqn          : str, 
                 fwd          : str, 
                 rev          : str=None 
                 ) -> None:
        
        #An example eqn is a + 2*b -> c
        stoich, rcts, prds = self.eqn2stoich(eqn)
        
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
        
        #It is now safe to call the parent's init
        super().__init__(ext_namespace, 
                         name, 
                         eqn=eqn,
                         _stoichiometry = stoich,
                         rate           = rate,
                         namespace      = tuple(namespace),
                         fwd_namespace  = tuple(fwd_namespace),
                         rev_namespace  = tuple(rev_namespace),
                         eqn_namespace  = tuple(eqn_namespace),
                         fwd            = str(fwd),
                         rev            = None if rev is None else str(rev),  
                         reactants      = tuple(rcts),
                         products       = tuple(prds)
                         )
        
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
    def to_data(self) -> dict:
        #Needs to be changed for export
        dct = {}
        for attr in ['eqn', 'fwd', 'rev']:
            value = getattr(self, attr)
            
            if value is not None:
                value = ut.try2num(value)
                
                if type(value) == tuple:
                    value = list(value)
                    
                dct[attr] = value
                
        return dct

class ReactionDict(NamespaceDict):  
    itype = Reaction
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, reactions: dict) -> None:
        namespace     = set()
        
        #Make the dict
        super().__init__(ext_namespace, reactions)
        
        states = set()
        
        for rxn_name, rxn in self.items():
            namespace.update(rxn.namespace)
            states.update(list(rxn.stoichiometry))
        
        #Save attributes
        self.namespace = tuple(namespace)
        self.states    = tuple(states)
        
        #Freeze
        self.freeze()
    