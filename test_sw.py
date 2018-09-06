from unittest import TestCase
from tools.chronos import Chronology
from tools.randvars import GaussRV
from tools.admin import Operator
from mods.SW.core import SW,stack,unstack, rstfile, step_1, step, gen_sample, m, \
	field_index, ordered_varname_ext,nxy,SWSetup, sample_filename
from tools.math import ens_compatible


import numpy as np

@ens_compatible
def hmod(E,t):
	return E[0]
# Localization has to be added here
h = {
	'm': 1,
	'model': hmod,
	'noise': GaussRV(C=4*np.eye(1))
}

#Forward model
f = {
	'm' :    m, #size of the state vector
	'noise': 0, #noise to be added to the model output ?
}


class Testswsetup(TestCase):
	def setUp( self ):
		t = Chronology(dt=float(1800), dkObs=1, T=1800 * 200, BurnIn=-1)
		self.setup = SWSetup(sample_filename,t=t,h=h,fdict=f)

	def test_members( self ):
		members = {'f','h','t','X0'}
		for m in members:
			self.assertTrue(hasattr(self.setup,m),'setup has no ' + m + ' attribute')
		self.assertTrue(isinstance(self.setup.t,Chronology))
		self.assertTrue(isinstance(self.setup.h,Operator))
		self.assertTrue(isinstance(self.setup.f,Operator))

	def test_sample (self ):
		X0 = self.setup.X0
		E = X0.sample(5)
		self.assertTupleEqual(E.shape,(5,m))
	def test_model( self ):
		f,X0 = self.setup.f, self.setup.X0
		E01 = X0.sample(2)
		E02 = self.setup.Eintern.copy()
		E11 = f(E01,0.,SW.dt)
		E12 = step(E02,0.,SW.dt)
		self.assertAlmostEqual(np.linalg.norm(E11-E12[:,:m]),0,7)
		self.assertAlmostEqual(np.linalg.norm(self.setup.Eintern-E12),0,7)
	def test_xprec( self ):
		f, X0 = self.setup.f, self.setup.X0
		E0 = X0.sample(2)
		E1 = f(E0,0.,SW.dt)
		E1intern = self.setup.Eintern.copy()
		if 2*m <= E1intern.shape[1]:
			self.assertAlmostEqual(np.linalg.norm(E1intern[:,2*m:]-E0[:,:m]),0,7)

class Testcore(TestCase):
	def test_varrestart( self ):
		"""Test is the variance of the restart is realistic"""
		var = np.var(SW.hphy)
		self.assertGreater(var,500,'Variance of hphy is low')
		self.assertGreater(50000,var,'Variance of hphy is high')

	def test_unstack ( self ):
		"""test the stack/unstack function"""
		h0 = SW.hphy
		x = stack(SW,ordered_varname_ext)
		SW.initstate_cst()
		h1 = SW.hphy
		unstack(x,SW,ordered_varname_ext)
		h2 = SW.hphy
		self.assertAlmostEqual(np.linalg.norm(h1-h2),0,7)
	def test_stack( self ):
		x = stack(SW,ordered_varname_ext)
		self.assertEqual(len(x),nxy*len(ordered_varname_ext),'wrong state size for stack output')

	def test_step1( self ):
		"""test the step_1 function in comparison with the next method of SW"""
		SW.inistate_rst(rstfile)
		x0 = stack(SW,ordered_varname_ext)
		SW.initstate_cst()
		for t in np.arange(200.):
			x0 = step_1(x0, t, SW.dt)
			SW.initstate_cst()
		unstack(x0, SW,ordered_varname_ext)
		h0 = dict()
		for n in ordered_varname_ext:
			h0[n] = SW.get_state(n).copy()
		SW.inistate_rst(rstfile)
		SW.set_time(0)
		for t in np.arange(200.):
			SW.next()
		h1 = dict()
		for n in ordered_varname_ext:
			h1[n] = SW.get_state(n).copy()
			self.assertAlmostEqual(np.linalg.norm(h1[n]-h0[n]),0,7)

	def test_step( self ):
		"""test the step function in comparison with the next method of SW on a single vector"""
		SW.inistate_rst(rstfile)
		x0 = stack(SW,ordered_varname_ext)
		SW.initstate_cst()
		for t in np.arange(200.):
			x0 = step(x0, t, SW.dt)
			SW.initstate_cst()
		unstack(x0, SW,ordered_varname_ext)
		h0 = dict()
		for n in ordered_varname_ext:
			h0[n] = SW.get_state(n).copy()
		SW.inistate_rst(rstfile)
		SW.set_time(0)
		for t in np.arange(200.):
			SW.next()
		h1 = dict()
		for n in ordered_varname_ext:
			h1[n] = SW.get_state(n).copy()
			self.assertAlmostEqual(np.linalg.norm(h1[n]-h0[n]),0,7)

	def test_step_ens( self ):
		""" test the step function in comparison with the next method of SW on a ensemble"""
		SW.inistate_rst(rstfile)
		x0 = stack(SW,ordered_varname_ext)
		x1 = step(x0,0.,SW.dt)
		x = np.stack((x0,x1))

	def test_gen_sample_size( self ):
		"""test the shape gen_sample output """
		sample = gen_sample(3, 0, 10)
		self.assertEqual(sample.shape[0],3,'wrong size of ensemble')
		self.assertEqual(sample.shape[1],SW.nx*SW.ny*len(ordered_varname_ext))

	def test_gen_sample_filling( self ):
		"""test if hte set of samples are different from zero"""
		sample = gen_sample(5,0,20)
		minvar = np.min(np.var(sample,axis=1))
		self.assertGreater(minvar,0)
	def test_field_index( self ):
		x = np.array(range(m))
		all_index = set.union(*[set(x[field_index(n,ordered_varname_ext)]) for n in ordered_varname_ext])
		self.assertEqual(len(all_index),m)

