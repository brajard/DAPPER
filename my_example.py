# Illustrate how to use DAPPER:
# Basic benchmarking of DA methods.

# Load DAPPER. Assumes pwd is <path-to-dapper>
from common import *


print (' HOST !!!')

# Load "twin experiment" setup
from mods.SW.sw import setup
from mods.SW.core import SW,rstfile
#setup.t.T = 10*1800.

# Specify a DA method configuration
config = EnKF('Sqrt', N=10, infl=1.02, rot=True, liveplotting=False)

# Simulate synthetic truth (xx) and noisy obs (yy)
xx,yy = simulate(setup)

# Assimilate yy (knowing the twin setup). Assess vis-a-vis xx.
stats = config.assimilate(setup,xx,yy)

# Average stats time series
avrgs = stats.average_in_time()

# Print averages
print_averages(config,avrgs,[],['rmse_a','rmv_a'])

# Plot some diagnostics
plot_time_series(stats, dim=1100)

# "Explore" objects individually
#print(setup)
#print(config)
#print(stats)
#print(avrgs)

