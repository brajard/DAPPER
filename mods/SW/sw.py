# Create a setup for a shallow-water experiment
from common import TwinSetup
from tools.chronos import Chronology
from tools.randvars import  RV, GaussRV
from mods.SW.core import SWSetup,step,m,dt, month_s,sample_filename,field_index,ordered_varname_state
from tools.math import ens_compatible
import numpy as np
t = Chronology(dt=float(dt),dkObs=1,T=dt*200,BurnIn=-1)

#Forward model
f = {
	'm' :    m, #size of the state vector
	'noise': 0, #noise to be added to the model output ?
}


############################
# Observation settings
############################
p  = 2000

obs_field = {'hphy','uphy','vphy'}
ptotal = p*len(obs_field)

def obs_inds(t):
	""" for each observed field, draw p random indices in stack vector"""
	I = np.array(range(m))
	tinds = tuple(np.random.choice(I[field_index(n,ordered_varname_state)],(p)) for n in obs_field)
	return np.concatenate(tinds)

@ens_compatible
def hmod(E,t):
	return E[obs_inds(t)]
	#It's here that the parametrization term could be added

#
std_o = 4

# Localization has to be added here
h = {
	'm': ptotal,
	'model': hmod,
	'noise': GaussRV(C=4*np.eye(ptotal))
}

setup = SWSetup(
	sample_filename = sample_filename,
	t = t,
	h = h,
	fdict = f)

