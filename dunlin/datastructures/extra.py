from typing import Sequence

import dunlin.utils                       as ut
import dunlin.standardfile.dunl.writedunl as wd
from dunlin.datastructures.bases import NamespaceDict, GenericItem
from dunlin.utils.typing         import ODict

class ExtraVariable(GenericItem):
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, name: str, *items: Sequence[str]):
        #Extract args and expr
        func_name, *args_ = items
        
        #Convert arguments to strings
        args = []
        for a_ in args_:
            a = str(a_)
            
            #Check
            if not ut.strisnum(a):
                ut.check_valid_name(a, allow_reserved=True)
            
            #Add the string version to args
            args.append(a)
            
        #Check namespace
        args_namespace = ut.get_namespace(args, allow_reserved=True)
        undefined      = args_namespace.difference(ext_namespace)
        if undefined:
            msg = f'Undefined namespace in extra variable {name}: {undefined}'
            raise NameError(msg)
        
        #It is now safe to call the parent's init
        super().__init__(ext_namespace, name)
        
        #Save attributes
        self.name      = name
        self.func_name = func_name
        self.args_ori  = tuple(args_)
        self.args      = tuple(args)
        self.namespace = tuple(args_namespace)
        self.signature = ', '.join(self.args)
        
        #Check name and freeze
        self.freeze()
    
    ###########################################################################
    #Export
    ###########################################################################
    def to_data(self) -> list:
        return [self.func_name, *self.args_ori]

class ExtraDict(NamespaceDict):
    itype = ExtraVariable
    
    ###########################################################################
    #Constructor
    ###########################################################################
    def __init__(self, ext_namespace: set, extras: dict) -> None:
        #Make the dict
        super().__init__(ext_namespace, extras)
        
        #Cache the extra variables
        self.names = () if extras is None else tuple(extras.keys())
        
        #Freeze
        self.freeze()
    
    


