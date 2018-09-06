from neuralsw.model.shalw import SWda as SWmodel
import numpy as np
import os
from common import progbar
from tools.utils import MLR_Print
from tools.admin import Operator
from tools.chronos import Chronology
from tools.randvars import RV

datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../shallownn/data')
# datadir = os.path.realpath('/Users/brajard/Documents/recherche/collaboration/bigdata/shallownn/data')

rstfile = os.path.join(datadir, 'restart_10years_dafull.nc')
sample_filename = os.path.join(datadir, 'SW_samples_dafull.npz')

# to regenerate the sample
rewrite = True

# parameters
prms = { 'nx': 80, 'ny': 80 }

warg = dict()
swarg = dict()
swarg['nu'] = 0.72*4

SW = SWmodel(nx=prms['nx'], ny=prms['ny'], warg=warg,**swarg)
if os.path.isfile(rstfile):
	SW.inistate_rst(rstfile)

dt = SW.dt
month_s = 3600 * 24 * 30
month_int = month_s // dt  # number of time step for a month

# Variables have to be ordered to be stacked and unstacked
# restVar are variables needed for a restart and therefore needed as state for a forward

#ordered_varname_state = ['hphy', 'uphy', 'vphy']
#ordered_varname_ext = ordered_varname_state + \
#                      ['hfil', 'ufil', 'vfil','hpre','upre','vpre']

ordered_varname_state = ['hphy', 'uphy', 'vphy']+['hfil', 'ufil', 'vfil','hpre','upre','vpre']
ordered_varname_ext = ordered_varname_state
assert (not bool(SW._restVar.symmetric_difference(set(ordered_varname_ext))))

# Total size of the domain:
nxy = prms['nx'] * prms['ny']
# Total size of the state variable
m = nxy * len(ordered_varname_state)


#########################
# Domain Management
#########################

def field_index ( name, ordered_varname=ordered_varname_ext ):
	"""Return in the stacked vector the index corresponding to a field"""
	i = ordered_varname.index(name)
	return slice(i * nxy, (i + 1) * nxy)


# stack/unstack function
def unstack ( x, sw, ordered_varname ):
	""" unstack x (vector) in the state variables of the SW
    unstack function modifies SW fields"""
	x = x.ravel()
	for i, name in enumerate(ordered_varname):
		# print(i, name)
		sw.set_state(name, x[i * nxy:(i + 1) * nxy].reshape((prms['ny'], prms['nx'])))


def stack ( sw, ordered_varname ):
	x = np.empty(nxy*len(ordered_varname))
	for i, name in enumerate(ordered_varname):
		x[i * nxy:(i + 1) * nxy] = np.ravel(sw.get_state(name))
	return x


def step_1 ( x0, t, dt_ ):
	"""Step a single state vector."""
	assert dt_ == SW.dt
	assert np.isfinite(t)
	# t is supposed to be a float.
	# SW is time independent (in absence of forcing) except for the saving utilities
	# So not sure the following is useful
	assert isinstance(t, float)
	SW.set_time(t)
	unstack(x0, SW,ordered_varname_ext)
	SW.next()
	x = stack(SW,ordered_varname_ext)
	return x


from tools.utils import multiproc_map


def step ( E, t, dt_ ):
	"""Vector and 2D-array (ens) input, with multiproc for ens case."""
	if E.ndim == 1:
		return step_1(E, t, dt_)
	if E.ndim == 2:
		# Parallelized:
		E = np.array(multiproc_map(step_1, E, t=t, dt_=dt_))
		# Non-parallelized:
		#for n,x in enumerate(E): E[n] = step_1(x,t,dt_)
		return E


#########################
# Free run
#########################
# The free run will generate samples to produce some initial states for the simulation
def gen_sample ( Len, SpinUp, Spacing ):
	sample = np.zeros((Len, nxy*len(ordered_varname_ext)))
	x = stack(SW,ordered_varname_ext)
	n = 0
	for k in progbar(range(Len * Spacing + SpinUp), desc='Simulating'):
		x = step_1(x, 0.0, dt)
		if k >= SpinUp and k % Spacing == 0:
			sample[n] = x
			n += 1
	return sample


#####################
# Setup class       #
#####################
# The experiment setup contains a
# -  RV for initstate
# - a f operator for model forward
# - a t chronology (independant from the model)
# - a observation operator (h independant from the model)

class SWRV(RV):
	def __init__ ( self, setup, *args, **kwargs ):
		super().__init__(*args, **kwargs)
		self._setup = setup

	def sample ( self, N ):
		# Extract the full state
		Eintern = super().sample(N)
		self._setup.clean()  # To be sure the model states will be cleaned
		self._setup.Eintern = Eintern
		return self._setup.E



class SWSetup(MLR_Print):
	def __init__ ( self, sample_filename, t, h ,fdict):
		self._mext = nxy * len(ordered_varname_ext)
		self._mstate = nxy * len(ordered_varname_state)
		self._h = h if isinstance(h, Operator) else Operator(**h)
		self._t = t if isinstance(t, Chronology) else Chronology(**t)
		self._f = Operator(model=self.model,**fdict)
		self._m = m  # m is the extended size or the state size ?
		self._Eintern = None  # Current state
		self._tsave = None  # Current time step
		self._X0 = SWRV(self, self._mext, file = sample_filename)
		self._firstupdate = False

	def clean ( self ):
		self._Eintern = None
		self._tsave = None


	@property
	def f (self):
		return self._f
	@property
	def t ( self ):
		return self._t

	@property
	def h ( self ):
		return self._h

	@property
	def Eintern ( self ):
		return self._Eintern

	@property
	def E (self):
		return self._Eintern[:,:self._mstate]

	@property
	def X0( self ):
		return self._X0

	@E.setter
	def E (self, value):
		if self._firstupdate :
			mm = self._mstate
			self._firstupdate = False
			inc = value - self.E
			if 2*mm <= self._mext:
				self._Eintern[:,mm:2*mm] += inc
			if 3*mm <= self._mext:
				self._Eintern[:,2*mm:] += inc
		self._Eintern[:,:self._mstate] = value



	@Eintern.setter
	def Eintern ( self, value ):
		assert self._Eintern is None, 'changing model state is not allowed'
		self._Eintern = value
		self._firstupdate = True

	def model( self, E, t,dt ):
		if not self._tsave is None:
			assert np.isclose(t,self._tsave), 'model not set at the current time step'
		self.E = E
		self._Eintern = step(self._Eintern,t,dt)
		self._tsave = t + dt
		return self.E



if __name__ == "__main__":
	import matplotlib.pyplot as plt

	if not os.path.isfile(sample_filename) or rewrite:
		print('Generating a "random" sample with which to start simulations')
		sample = gen_sample(100, 0, 2 * month_int)
		# spacing : 2 months
		np.savez(sample_filename, sample=sample)
