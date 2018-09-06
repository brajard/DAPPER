# Create a setup for a shallow-water experiment
from common import TwinSetup
from tools.chronos import Chronology
from tools.randvars import  RV, GaussRV
from mods.SW.core import SWSetup,step,m,dt, month_s,sample_filename,field_index,ordered_varname_state
from tools.math import ens_compatible
import numpy as np
t = Chronology(dt=float(dt),dkObs=100,T=2*month_s,BurnIn=-1)

#Forward model
f = {
	'm' :    m, #size of the state vector
	'noise': 0, #noise to be added to the model output ?
}


############################
# Observation settings
############################
p  = 1000

obs_field = {'hphy'}

ptotal = p*len(obs_field)
tinds =  np.zeros((int(t.T/t.dt),p*len(obs_field)),dtype=int)
I = np.array(range(m))
for k,KObs,t_,dt in t.forecast_range:
	tmp = tuple(np.random.choice(I[field_index(n,ordered_varname_state)],(p)) for n in obs_field)
	tinds[int(t_/dt)-1,:] = np.concatenate(tmp)



@ens_compatible
def hmod(E,t):
	return E[tinds[int(t/dt)-1]]
	#It's here that the parametrization term could be added

#
std_o = 4

# Localization has to be added here
h = {
	'm': ptotal,
	'model': hmod,
	'noise': GaussRV(C=std_o*np.eye(ptotal))
}

setup = SWSetup(
	sample_filename = sample_filename,
	t = t,
	h = h,
	fdict = f)

