# Reproduce results from
# table1 of sakov et al "iEnKF" (2012)

from common import *
import numpy as np
import mods.MAOOAM.params2
import mods.MAOOAM.aotensor
import mods.MAOOAM.integrator
import mods.MAOOAM.ic_def as ic_def

#initial condition already after the transient time (10 years here)
ic_def.load_IC()
import mods.MAOOAM.ic as ic
mu0= ic.X1

print (" Model initialized")
m = mods.MAOOAM.params2.ndim
p = m

#3 years simulation - time step 0.1
#obs time step 1 day
#model transient time 10 years
#DA transient time 1

T=32444
BurnIn=32444
t = Chronology(0.1,dtObs=9,T=T,BurnIn=BurnIn)

f = {
    'm': m,
    'model': lambda x,t,dt: mods.MAOOAM.integrator.step(x,t,dt),
    'TLM'  : 0,
    'noise': 0
    }

#the var on 100 years effective dt=0.1
var = array([   2.25705441e-05,   2.70829072e-04,   3.00434304e-04,
         2.24907615e-04,   1.48750056e-04,   2.30601477e-04,
         6.24018781e-05,   6.07697222e-05,   3.35717671e-05,
         3.32832185e-05,   2.80775530e-05,   5.68928169e-05,
         7.48198724e-05,   2.98037439e-05,   2.41227820e-05,
         8.89876165e-05,   7.04450249e-06,   6.81838195e-06,
         2.59002057e-06,   2.57755616e-06,   1.23963736e-09,
         4.46789964e-08,   1.76000387e-09,   5.35801312e-12,
         1.43288311e-09,   6.22013831e-08,   1.27243907e-09,
         5.07635211e-12,   1.12155637e-04,   2.71467456e-03,
         1.57806939e-04,   2.76227153e-03,   3.06848875e-04,
         3.23858331e-03,   4.04752311e-05,   1.61287593e-06])
#GRL var after 1000 years - 200 years - 0.1 - fortran
var2=array([  5.50178812e-06,   1.16978285e-03,   1.17177680e-03,
         7.21962886e-04,   3.83099335e-04,   3.90241255e-04,
         2.28479504e-04,   2.29940349e-04,   1.34958968e-04,
         1.36127019e-04,   3.68486787e-05,   1.65499476e-04,
         1.74648794e-04,   6.60638117e-05,   5.55218680e-05,
         5.36613113e-05,   2.34540876e-05,   2.35861194e-05,
         1.35298128e-05,   1.35433822e-05,   6.82109881e-10,
         9.61053046e-11,   1.87162920e-10,   8.67905811e-15,
         5.22157944e-10,   7.27624452e-11,   1.82442890e-10,
         1.20562714e-14,   2.96399161e-04,   3.60052729e-05,
         8.05671226e-04,   1.79436977e-04,   4.95553796e-04,
         3.45518933e-05,   5.79866355e-06,   5.70396595e-06])

#ensemble VARIANCE is 1% of var 200 years effective dt=0.01
C0 = 0.01*0.01*diag(var2)
X0 = GaussRV(C=C0,mu=mu0)


#observation noise variance is 1% of the var on 200 years effective dt=0.01
R= 0.01*0.01*diag(var2)
hnoise = GaussRV(C=CovMat(R),mu=0)

h = {
    'm': p,
    'model': lambda x,t: x,
    'TLM'  : lambda x,t: eye(p),
    'noise': hnoise,
    }

other = {'name': os.path.basename(__file__)}

setup = OSSE(f,h,t,X0,**other)
