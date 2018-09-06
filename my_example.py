# Illustrate how to use DAPPER:
# Basic benchmarking of DA methods.

# Load DAPPER. Assumes pwd is <path-to-dapper>
from common import *
make_simu = True
print('simple ENKF with SW model')

# Load "twin experiment" setup
from mods.SW.utils import save_stats,plot_from_file
from mods.SW.core import SW,rstfile
#setup.t.T = 10*1800.
import numpy as np
# Specify a DA method configuration

if make_simu:
	seed(13)
	from mods.SW.sw import setup

	config = EnKF('Sqrt', N=50, infl=1.02, rot=True, liveplotting=False)

	# Simulate synthetic truth (xx) and noisy obs (yy)
	xx,yy = simulate(setup)
	# Assimilate yy (knowing the twin setup). Assess vis-a-vis xx.
	stats = config.assimilate(setup,xx,yy)

	# Average stats time series
	avrgs = stats.average_in_time()

	# Print averages
	print_averages(config,avrgs,[],['rmse_a','rmv_a'])

	# Plot some diagnostics
	plot_time_series(stats, dim=1620)
	plt.savefig('time_series.png')
	plt.show()

	save_stats('stats.npz',stats)
stats2,fig1,fig2 = plot_from_file('stats.npz',t=0,
	vmin2=-50,vmax2=50,
	vmax = 5)


fig1.savefig('mean_t0.png')
fig2.savefig('std_t0.png')


fig1.show()
plt.figure()
fig2.show()
stats2,fig1,fig2 = plot_from_file('stats.npz',t=-1,
	vmin2=-10,vmax2=10,
	vmax=0.5)


fig1.savefig('mean_tend.png')
fig2.savefig('std_tend.png')
fig1.show()
plt.figure()
fig2.show()
#plt.savefig('state_100.png')
# "Explore" objects individually
#print(setup)
#print(config)
#print(stats)
#print(avrgs)

