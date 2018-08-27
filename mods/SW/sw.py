# Create a setup for a shallow-water experiment
from common import Chronology, TwinSetup
from tools.randvars import  RV, GaussRV
from mods.SW.core import step,m,dt, month_s,sample_filename,field_index
from tools.math import ens_compatible
import numpy as np
t = Chronology(dt=float(dt),dkObs=1,T=dt*200,BurnIn=-1)

#Forward model
f = {
	'm' :    m, #size of the state vector
	'model': step, #model
	'noise': 0, #noise to be added to the model output ?
}

X0 = RV (m=m, file=sample_filename)

############################
# Observation settings
############################
p  = 2000

obs_field = {'hphy','uphy','vphy'}
ptotal = p*len(obs_field)

def obs_inds(t):
	""" for each observed field, draw p random indices in stack vector"""
	I = np.array(range(m))
	tinds = tuple(np.random.choice(I[field_index(n)],(p)) for n in obs_field)
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

setup = TwinSetup(f,h,t,X0)

