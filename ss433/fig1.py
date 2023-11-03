import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.modeling.models import Models
from regions import  CircleSkyRegion
from gammapy.modeling.models import Models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.patheffects as PathEffects
from gammapy.estimators import FluxPoints
from gammapy.visualization import colormap_hess
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LinearStretch
import plot_params as pparams
from matplotlib.gridspec import GridSpec

data_path = './ss433/hess_data/'

# Read H.E.S.S. Map
hess_map = Map.read(data_path + '/maps/full_significance_without_1908.fits.gz')

# Read H.E.S.S. flux points and models
model_east = Models.read(data_path+'/spectral_fits/aperture_photometry_eastern_pwl.fits')[0]
model_east_syst = Models.read(data_path+'/spectral_fits/eastern_model_with_systematics.fits.gz')[0].spectral_model

model_west = Models.read(data_path+'/spectral_fits/aperture_photometry_western_pwl.fits')[0]
model_west_syst = Models.read(data_path+'/spectral_fits/western_model_with_systematics.fits.gz')[0].spectral_model

# flux point errors already include systematics
east_fp = FluxPoints.read(data_path+'/spectral_fits/eastern_fluxpoints_with_syst.fits.gz',  reference_model = model_east)
west_fp = FluxPoints.read(data_path+'/spectral_fits/western_fluxpoints_with_syst.fits.gz',  reference_model = model_west)

# HAWC point including systematics
hawc_point_e = 2.4e-16*u.TeV**-1*u.s**-1*u.cm**-2
hawc_point_w = 2.1e-16*u.TeV**-1*u.s**-1*u.cm**-2
e_hawc = 20*u.TeV
hawc_err_e = (np.sqrt(1.3**2+0.5*+2)*1e-16, np.sqrt(1.3**2+0.6*+2)*1e-16)*u.TeV**-1*u.s**-1*u.cm**-2
hawc_err_w = (np.sqrt(1.2**2+0.5*+2)*1e-16, np.sqrt(1.2**2+0.6*+2)*1e-16)*u.TeV**-1*u.s**-1*u.cm**-2

# Define H.E.S.S. colormap
normalize = ImageNormalize(vmin=-5, vmax=10, stretch=LinearStretch())
transition = normalize(7.5).data[0]
width = normalize(0.1).data[0]
cmap = colormap_hess(transition=transition, width=width)

# x-ray contours
xr = np.genfromtxt(data_path + 'ss433_rosat_brinkmann_xray_fixed.csv', delimiter=',')
xx = np.asarray([i[0] for i in xr])
xy = np.asarray([i[1] for i in xr])
c = SkyCoord(ra=xx*u.degree, dec=xy*u.degree,frame='icrs')


# Zoom to SS 433 region
position = SkyCoord(39.68907589, -2.5, unit="deg", frame='galactic')
r = 2.25*u.deg
hess_small_sig_1908 = hess_map.cutout(position, r).reduce_over_axes()

ra_contour_zoom = hess_small_sig_1908.geom.to_image().coord_to_pix(c)[0]
dec_contour_zoom = hess_small_sig_1908.geom.to_image().coord_to_pix(c)[1]


# Define the x-ray regions for the plot
w2 = SkyCoord(l = 39.49459783913566, b = -1.7422062350119902, unit='deg', frame='galactic')
w1 = SkyCoord(l = 39.603841536614645, b=-1.9520383693045562, unit='deg', frame='galactic')
e1 = SkyCoord(l = 39.85354141656663, b=-2.664268585131895, unit='degree', frame='galactic')
e2 = SkyCoord(l = 39.91596638655462, b=-2.833333333333333, unit='degree', frame='galactic')
e3 = SkyCoord(l = 40.05402160864346, b=-3.248201438848921, unit='degree', frame='galactic')

w2_pix = hess_small_sig_1908.geom.to_image().coord_to_pix(w2)
w1_pix = hess_small_sig_1908.geom.to_image().coord_to_pix(w1)
e1_pix = hess_small_sig_1908.geom.to_image().coord_to_pix(e1)
e2_pix = hess_small_sig_1908.geom.to_image().coord_to_pix(e2)
e3_pix = hess_small_sig_1908.geom.to_image().coord_to_pix(e3)

# PSF containment
lw = 2
path_effects = [PathEffects.withStroke(linewidth=lw+1, foreground="k")]
cont_full  =0.05168437035203161*u.deg
psf_full_region = CircleSkyRegion(SkyCoord(l=38.9*u.deg, b=-3.4*u.deg, frame='galactic'),
                            radius=cont_full)

pixel_psf_full_region= psf_full_region.to_pixel(hess_small_sig_1908.geom.wcs)

# Fix for the colormap
sky2 = plt.imshow(hess_small_sig_1908.data, cmap=cmap, vmax=7.5, vmin=-4)
plt.colorbar()
plt.close()

fig = plt.figure(constrained_layout=True,figsize=(16,10))

gs = GridSpec(2, 3, figure=fig)
ax = fig.add_subplot(gs[0:2, 0:2], projection = hess_small_sig_1908.geom.wcs)
ax2 = fig.add_subplot(gs[0:1, 2])
ax1 = fig.add_subplot(gs[1:2, 2])

ax.coords[0].set_ticklabel(size = pparams.MEDIUM_SIZE, color='k')
ax.coords[1].set_ticklabel(size = pparams.MEDIUM_SIZE, color='k')


ax= hess_small_sig_1908.plot(ax = ax, add_cbar=False, cmap=cmap, vmax=7.5, vmin=-4)
cbaxes = fig.add_axes([0.5, 0.7, 0.02, 0.25])
cbar = plt.colorbar(sky2, ticks = [-2.5, 0., 2.5, 5, 7.5],orientation="vertical",fraction=0.048,pad=0.03,cax=cbaxes)
cbar.outline.set_edgecolor('white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', weight = "bold")
cbar.ax.set_ylabel('', rotation=90)
cbar.set_label("       Significance" ,  fontsize=pparams.MEDIUM_SIZE, color='white', labelpad = -90 , weight = "bold", rotation = 90, y = 0.4)

ax.coords[0].set_ticks(spacing=1 *u.degree, color = 'white')
ax.coords[1].set_ticks(spacing=1 *u.degree, color = 'white')

ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.scatter(ra_contour_zoom, dec_contour_zoom, color='aqua', s=2.5, alpha=0.5)

color_c = "white"
edge_c = "k"
lw = 2
s = 150
path_effects = [PathEffects.withStroke(linewidth=lw+1, foreground="k")]
ax.scatter(w1_pix[0], w1_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
ax.scatter(w2_pix[0], w2_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
ax.scatter(e1_pix[0], e1_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
ax.scatter(e2_pix[0], e2_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
ax.scatter(e3_pix[0], e3_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
ax.text(w1_pix[0]-19, w1_pix[1]-3, s= "w1", color=color_c, weight='bold', 
        path_effects=path_effects)
ax.text(w2_pix[0]+4, w2_pix[1]+3, s= "w2", color=color_c, weight='bold', 
        path_effects=path_effects)

ax.text(e1_pix[0]+8, e1_pix[1]+0, s= "e1", color=color_c, weight='bold', 
        path_effects=path_effects)
ax.text(e2_pix[0]+6, e2_pix[1]-6, s= "e2", color=color_c, weight='bold', 
        path_effects=path_effects)
ax.text(e3_pix[0]+2, e3_pix[1]+2, s= "e3", color=color_c, weight='bold', 
        path_effects=path_effects)

ax.set_xlabel('Galactic Longitude (J2000)', fontsize=pparams.BIGGER_SIZE)
ax.set_ylabel('Galactic Latitude (J2000)', fontsize=pparams.BIGGER_SIZE)
ax.text(40.65,-3.45,"H.E.S.S.", color = 'white',transform = ax.get_transform('galactic'), fontsize = pparams.HUGE_SIZE+10, fontweight = 'bold',path_effects=path_effects)
pixel_psf_full_region.plot(ax=ax, path_effects=path_effects, edgecolor='white', lw=3)
ax.text(200,20, s= "PSF", color=color_c, weight='bold', 
        path_effects=path_effects)
ax.text(40.65,-1.65,"a", color = 'white',transform = ax.get_transform('galactic'), fontsize = pparams.HUGE_SIZE+10, fontweight = 'bold',path_effects=path_effects)

eran = [0.8,120]*u.TeV
lw_lin=3.5
lw_p=3

east_fp.plot(ax=ax1, color='teal', energy_power=2, markersize=8,label='H.E.S.S.', lw=lw_p)

model_east_syst.plot_error(ax=ax1, facecolor='turquoise', energy_bounds = eran, energy_power=2)
model_east.spectral_model.plot(lw=lw_lin,ax=ax1, color='teal', energy_bounds = eran, energy_power=2, label='H.E.S.S.')
model_east.spectral_model.plot_error(ax=ax1, facecolor='teal', energy_bounds = eran, energy_power=2)
ax1.errorbar(e_hawc.value, (e_hawc**2*hawc_point_e).to_value('TeV s-1 cm-2'), ls='',lw=lw_p,markersize=10, yerr=(e_hawc**2*hawc_err_e).to_value('TeV s-1 cm-2')[:,None], marker='s', color='navy', label='HAWC')
# get handles
handles, labels = ax1.get_legend_handles_labels()
# remove the errorbars
handles = [handles[0], handles[1], handles[2]]
# use them in the legend
ax1.legend(handles, labels,loc='lower left')
ax1.text(45, 3e-13,"c", color = 'k', fontsize = pparams.HUGE_SIZE+10, fontweight = 'bold')
ax1.text(30, 1.5e-14,"east", color = 'k', fontsize = pparams.HUGE_SIZE-5, fontweight = 'bold')


west_fp.plot(ax=ax2, color='chocolate', energy_power=2, markersize=8, lw=lw_p, label='H.E.S.S.')

model_west_syst.plot_error(ax=ax2, facecolor='orange', energy_bounds = eran,energy_power=2)
model_west.spectral_model.plot(lw=lw_lin,ax=ax2, color='chocolate', energy_bounds = eran, energy_power=2, label='H.E.S.S.')
model_west.spectral_model.plot_error(ax=ax2, facecolor='chocolate', energy_bounds = eran, energy_power=2)
ax2.errorbar(e_hawc.value, (e_hawc**2*hawc_point_w).to_value('TeV s-1 cm-2'), ls='', lw=lw_p, markersize=10, yerr=(e_hawc**2*hawc_err_w).to_value('TeV s-1 cm-2')[:,None], marker='s', color='maroon', label='HAWC')
ax2.legend(loc='lower left')
ax2.text(45, 3e-13,"b", color = 'k', fontsize = pparams.HUGE_SIZE+10, fontweight = 'bold')
ax2.text(30, 1.5e-14,"west", color = 'k', fontsize = pparams.HUGE_SIZE-5, fontweight = 'bold')

for ax in [ax1, ax2]:
    ax.set_ylim(1e-14, 6e-13)
    ax.set_ylabel('')
    ax.set_xlabel('Energy (TeV)')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel(r'E$^2 \frac{dN}{dE}$ (TeV·cm$^{-2}$·s$^{-1}$)', rotation=270, labelpad=35)
    ax.set_xlim(0.8,110)

plt.savefig('ss433/plots/fig1.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0.1)
plt.savefig('ss433/plots/fig1.png', transparent = False, bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

