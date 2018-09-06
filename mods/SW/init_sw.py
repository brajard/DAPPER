#!/usr/bin/env python

# coding: utf-8

"""
Script use to init the sw data assimilation experiment
First generate a restart and then generate some samples
"""

from neuralsw.model.shalw import SWda as SWmodel
import os,sys
sys.path.append(os.getcwd())

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mods.SW.core import gen_sample, month_int



# GENERATE restart
def generate_restart(rstfile,endtime,outname=None,PLOT=True):
	SW = SWmodel(nx=80, ny=80)
	SW.initstate_cst(0, 0, 0)
	if outname:
		SW.save(time=np.arange(0, endtime, 48 * 30), name=outname)

	# run the model
	for i in tqdm(range(endtime)):
		SW.next()

	# Save the restart
	SW.save_rst(rstfile)

	# In[ ]:


	## Plots conservative quantities
	if PLOT:
		## For some unresolved reasons , the keras modul (included in model) should not be called before dealing with nc files
		import neuralsw.model.modeltools as model

		ds = xr.open_dataset(outname)

		fig, ax = plt.subplots(nrows=3, sharex='all')
		Ec = model.cinetic_ener(ds=ds)
		Ep = model.potential_ener(ds=ds)
		Pv = model.potential_vor(ds=ds)
		Ec.plot(ax=ax[0])
		Ep.plot(ax=ax[1])
		Pv.plot(ax=ax[2])
		ax[0].set_title('mean kinetic energy')
		ax[0].set_ylabel('Ec')
		ax[0].set_xlabel('')
		ax[1].set_title('mean potential energy')
		ax[1].set_ylabel('Ep')
		ax[1].set_xlabel('')
		ax[2].set_title('mean potential vorticity')
		ax[2].set_ylabel('Pv')
		plt.show()

def generate_sample(sample_filename,rewrite=False):
	if os.path.isfile(sample_filename) and not rewrite:
		raise ValueError('sample file already exists!')

	sample = gen_sample(100, 0, 2 * month_int)
	# spacing : 2 months
	np.savez(sample_filename, sample=sample)

if __name__=='__main__':
	# rootdir
	datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../shallownn/data')

	# suffix (modify default parameters if not empty)
	suf = '_dafull'

	# savefile (to check the run)
	outname = os.path.join(datadir, 'restartrun' + suf + '.nc')

	# restartfile
	rstfile = os.path.join(datadir, 'restart_10years' + suf + '.nc')

	#sample file
	sample_filename = os.path.join(datadir, 'SW_samples_dafull.npz')

	# Duration of the integration
	endtime = 48 * 30 * 12 * 10  # 10 years

	print('data directory:', datadir)

	#TODO Correct, does not work if no restart.
	if not os.path.isfile(rstfile):
		generate_restart(rstfile,endtime,outname,PLOT=False)
	generate_sample(sample_filename,True)