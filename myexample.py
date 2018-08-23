from neuralsw.model.shalw import SWmodel
import numpy as np
import os
from common import progbar

datadir = os.path.realpath('/Users/brajard/Documents/recherche/collaboration/bigdata/shallownn/data')

rstfile = os.path.join(datadir, 'restart_10years.nc')
sample_filename = os.path.join(datadir, 'SW_samples.npz')

#to regenerate the sample
rewrite = True

# parameters
prms = {
    'nx': 80,
    'ny': 80
}




warg = dict()

SW = SWmodel(nx=prms['nx'], ny=prms['ny'], warg=warg)
SW.inistate_rst(rstfile)

dt = SW.dt
# Variables have to be ordered to be stacked and unstacked
# restVar are variables needed for a restart and therefore needed as state for a forward
ordered_varname = tuple(v for v in SW._restVar)

# Total size of the domain:
nxy = prms['nx']*prms['ny']
#Total size of the state variable
m = nxy * len(ordered_varname)


# stack/unstack function
def unstack(x, sw):
    """ unstack x (vector) in the state variables of the SW
    unstack function modifies SW fields"""
    for i, name in enumerate(ordered_varname):
        #print(i, name)
        sw.set_state(name, x[i * nxy:(i + 1) * nxy].reshape((prms['ny'], prms['nx'])))


def stack(sw):
    x = np.empty(m)
    for i, name in enumerate(ordered_varname):
        #print(i, name)
        x[i * nxy:(i + 1) * nxy] = np.ravel(sw.get_state(name))
    return x
def step_1(x0, t, dt_):
    """Step a single state vector."""
    assert dt_ == SW.dt
    assert np.isfinite(t)
    #t is supposed to be a float.
    #SW is time independent (in absence of forcing) except for the saving utilities
    #So not sure the following is useful
    assert isinstance(t, float)
    SW.set_time(t)
    unstack(x0,SW)
    SW.next()
    x = stack(SW)
    return x



from tools.utils import multiproc_map
def step(E, t, dt_):
    """Vector and 2D-array (ens) input, with multiproc for ens case."""
    if E.ndim==1:
        return step_1(E,t,dt_)
    if E.ndim==2:
    # Parallelized:
        E = np.array(multiproc_map(step_1, E, t=t, dt_=dt_))
        # Non-parallelized:
        #for n,x in enumerate(E): E[n] = step_1(x,t,dt_)
        return E

#########################
# Free run
#########################
# The free run will generate samples to produce some initial states for the simulation
def gen_sample(Len,SpinUp,Spacing):
    sample = np.zeros((Len,m))
    x = stack(SW)
    n = 0
    for k in progbar(range(Len*Spacing + SpinUp),desc='Simulating'):
        x = step_1(x,0.0,dt)
        if k>=SpinUp and k%Spacing==0:
            sample[n] = x
            n += 1
    return sample



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if not os.path.isfile(sample_filename) or rewrite:
        print('Generating a "random" sample with which to start simulations')
        month = 3600*24*30 // dt #number of time step for a month
        sample = gen_sample(100, 0, 2*month)
        #spacing : 2 months
        np.savez(sample_filename, sample=sample)
