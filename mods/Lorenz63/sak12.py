# Reproduce results from
# table1 of sakov et al "iEnKF" (2012)

from common import *

from mods.Lorenz63.core import step, dfdx

m = 3
p = m

t = Chronology(0.01,dkObs=25,T=4**5,BurnIn=4)

m = 3
f = {
    'm'    : m,
    'model': lambda x,t,dt: step(x,t,dt),
    'jacob': dfdx,
    'noise': 0
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=2,mu=mu0)

h = {
    'm'    : p,
    'model': lambda x,t: x,
    'jacob': lambda x,t: eye(3),
    'noise': GaussRV(C=2,m=p)
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
#config = DAC(Climatology)                                      # 8.5
#config = DAC(D3Var)                                            # 1.26
#config = DAC(ExtKF, infl=90);                                  # 0.87
#config = DAC(EnKF,'Sqrt',    N=3 , infl=1.30)                  # 
#config = DAC(EnKF ,'Sqrt',   N=10, infl=1.02,rot=True)         # 0.63 (sak: 0.65)
#config = DAC(EnKF ,'PertObs',N=500,infl=0.95,rot=False)        # 0.56
#config = DAC(iEnKF,'Sqrt',   N=10, infl=1.02,rot=True,iMax=10) # 0.31
#config = DAC(PartFilt,       N=800,NER=0.05)                   # 0.275 (with N=4000)

