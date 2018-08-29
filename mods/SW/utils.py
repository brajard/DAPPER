import numpy as np
from mods.SW.core import prms, field_index
import matplotlib.pyplot as plt

def save_stats(fname,stats,fields={'mu','var'},store_u=False):
	#Make a dict
	save_dic = dict()
	assert store_u == False #not implemented yet
	cat = {'f','a'}
	for k in fields:
		serie =  getattr(stats,k)
		for c in cat:
			newk = '_'.join([k,c])
			save_dic[newk] = getattr(serie,c)
	save_dic['xx'] = stats.xx
	save_dic['yy'] = stats.yy
	np.savez(fname,**save_dic)
	#TODO : better save (with time)

def plot_from_file(fname,t=-1):
	stats = np.load(fname)
	mu_a = stats['mu_a']
	mu_f = stats['mu_f']
	xx = stats['xx']
	FA = mu_a[t,field_index('hphy')].reshape((prms['ny'],prms['nx']))
	FF = mu_f[t,field_index('hphy')].reshape((prms['ny'],prms['nx']))
	FT = xx[t,field_index('hphy')].reshape((prms['ny'],prms['nx']))
	fig,ax = plt.subplots(ncols=3,nrows=2)
	vmin1,vmax1 = -120,120
	vmin2,vmax2 = -10,10
	ax[0,2].imshow(FA,vmin=vmin1,vmax=vmax1)
	ax[0,2].set_title('Analysis')
	im1 = ax[0,1].imshow(FF,vmin=vmin1,vmax=vmax1)
	ax[0,1].set_title('Forecast')
	ax[0,0].imshow(FT,vmin=vmin1,vmax=vmax1)
	ax[0,0].set_title('Truth')
	ax[1,0].imshow(FA-FF,vmin=vmin2,vmax=vmax2,cmap='coolwarm')
	ax[1,0].set_title('A-F')
	im2 = ax[1,1].imshow(FF-FT,vmin=vmin2,vmax=vmax2,cmap='coolwarm')
	ax[1,1].set_title('F-T')
	ax[1,2].imshow(FA-FT,vmin=vmin2,vmax=vmax2,cmap='coolwarm')
	ax[1,2].set_title('A-T')
	cax1 = fig.add_axes([0.05,0.05,0.4,0.02])
	cax2 = fig.add_axes([0.55,0.05,0.4,0.02])
	fig.colorbar(im1,cax=cax1,orientation='horizontal')
	fig.colorbar(im2,cax=cax2,orientation='horizontal')

	return stats
