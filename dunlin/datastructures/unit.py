from .bases import TabularDict

class UnitsDict(TabularDict):
    is_numeric = False
    
    def __init__(self, ext_namespace: set, mapping: dict) -> None:
        if mapping:
            missing = ext_namespace.difference(mapping)
            
            if missing:
                msg = f'Missing units for {missing}.'
                raise ValueError(msg)
            
            unexpected = set(mapping).difference(ext_namespace)
            
            if unexpected:
                msg = f'Unexpected namespaces {unexpected}.'
                raise ValueError(msg)
            
        super().__init__(set(), 'units', mapping)
    
    def to_data(self) -> dict:
        return self.df.to_dict('list')