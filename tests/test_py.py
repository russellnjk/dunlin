# import pymodel as M

import pydream.parameters as ppm
import pydream.core       as pcr
import pydream.Dream      as pdm

from scipy.stats import norm

p0_norm = norm(1, 0.1)
p1_norm = norm(2, 0.1)


def likelihood(p):
    e = abs(p[0]-1.2)+abs(p[1]-2.2)
    e = e/0.1
    return e

parameters = [ppm.SampledParam(norm, 1, 0.1), ppm.SampledParam(norm, 1, 0.1)]

sampled_params, log_ps = pcr.run_dream(parameters, 
                                       likelihood, 
                                       niterations=10000, 
                                       verbose=True
                                       )

print(sampled_params)
print(log_ps)

