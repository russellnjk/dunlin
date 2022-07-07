import numpy  as np
import pandas as pd
from datetime import datetime
from numbers  import Number
from typing   import Optional, Sequence, TypeVar, Union

'''
Contains type aliases for common type hints.
'''

#.dunl 
Primitive = Union[str, Number, datetime, bool]

#Duck-types
Listlike = Union[list, tuple]
Dflike   = Union[dict, pd.DataFrame, pd.Series]

#Containers
Dflst     = Sequence[pd.DataFrame]
Dfdct     = dict[pd.DataFrame]

#Numeric
Num = Number
Arr = Union[np.ndarray, Number]  
Bnd = tuple[Num, Num]

#Modelling terms
Model     = TypeVar('Model')
Index     = Union[str, Num]
Scenario  = Union[str, Num, tuple[Num]]
VScenario = tuple[Union[str, Listlike], Scenario]
VData     = Union[Num, pd.Series, dict]

#Optionals
ODict     = Optional[dict]
OStr      = Optional[str]
OScenario = Optional[Scenario]

