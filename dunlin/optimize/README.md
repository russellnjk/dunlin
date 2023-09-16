# Description
The `optimize` submodule handles optimization (including curve-fitting) and sensitivity analysis.


# Motivation
Simulation/numerical integration alone is insufficient for modelers; data needs to be fitted and parameters need to be tuned. 

# Implementation
The `Optimizer` class contains all methods for optimization and sensitivity analysis for a given objective function. To make testing easier, the main constructor `__init__` is designed to be used without a model. The alternative constructor `from_model` is to be used in all other cases. 

The settings for optimization and sensitivity analysis are taken from 
the model's `opt_args` attribute. This attribute contain details of the free parameters and settings for optimization/sensitivity analysis. However, the `opt_args` argument in the constructor of `Optimizer` only needs to contain the latter. Meanwhile, the free parameters are accepted as a separate argument. This is to make explicit that free parameters are a required input as opposed to algorithm settings.

The `Trace` class is instantiated after optimization is complete and contains methods for visualizing and checking the trace plot and parameter distributions from the optimization process. The settings for trace analysis are taken from the model's `trace_args` attribute. Because `Trace` instances are created every time an optimization process is complete, the `Optimizer` class also stores `trace_args` during instantiation.

The `Optimizer` class also contains methods for sensitivity analysis. These are wrap over SALib's functions and the return values of those methods are SALib data structures. Because the methods for sensitivity analysis can be coded separately from the optimization algorithms, they are compartmentalized in the `SensitivityMixin` class to keep the length of each file reasonable.

Because curve-fitting is such a common goal of optimization, the `Curvefitter` class is designed specifically to quickly set up the 
objective function (Sum-Squared Error). It is implemented as a subclass of `Optimizer`.

Finally, the user is not expected to know which classes to instantiate so the submodule comes with front-end functions for simplicity.