from unittest import TestCase
from myexample import SW,stack,unstack, rstfile, step_1, step, gen_sample
import numpy as np

class Testcore(TestCase):
	def test_varrestart( self ):
		"""Test is the variance of the restart is realistic"""
		var = np.var(SW.hphy)
		self.assertGreater(var,500,'Variance of hphy is low')
		self.assertGreater(50000,var,'Variance of hphy is high')

	def test_unstack ( self ):
		"""test the stack/unstack function"""
		h0 = SW.hphy
		x = stack(SW)
		SW.initstate_cst()
		h1 = SW.hphy
		unstack(x,SW)
		h2 = SW.hphy
		self.assertAlmostEqual(np.linalg.norm(h1-h2),0,7)

	def test_step1( self ):
		"""test the step_1 function in comparison with the next method of SW"""
		SW.inistate_rst(rstfile)
		x0 = stack(SW)
		for t in np.arange(200.):
			x0 = step_1(x0, t, SW.dt)
		unstack(x0, SW)
		h0 = SW.hphy
		SW.inistate_rst(rstfile)
		SW.set_time(0)
		for t in np.arange(200.):
			SW.next()
		h1 = SW.hphy
		self.assertAlmostEqual(np.linalg.norm(h1-h0),0,7)

	def test_step( self ):
		"""test the step function in comparison with the next method of SW on a single vector"""
		SW.inistate_rst(rstfile)
		x0 = stack(SW)
		for t in np.arange(200.):
			x0 = step(x0, t, SW.dt)
		unstack(x0, SW)
		h0 = SW.hphy
		SW.inistate_rst(rstfile)
		SW.set_time(0)
		for t in np.arange(200.):
			SW.next()
		h1 = SW.hphy
		self.assertAlmostEqual(np.linalg.norm(h1-h0),0,7)

	def test_gen_sample_size( self ):
		"""test the shape gen_sample output """
		sample = gen_sample(3, 0, 10)
		self.assertEqual(sample.shape[0],3,'wrong size of ensemble')
		self.assertEqual(sample.shape[1],SW.nx*SW.ny*len(SW._restVar))

	def test_gen_sample_filling( self ):
		"""test if hte set of samples are different from zero"""
		sample = gen_sample(5,0,20)
		minvar = np.min(np.var(sample,axis=1))
		self.assertGreater(minvar,0)
