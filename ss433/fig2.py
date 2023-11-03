import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from regions import  CircleSkyRegion
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.patheffects as PathEffects
from gammapy.visualization import colormap_hess
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LinearStretch
import plot_params as pparams

data_path = './ss433/hess_data/'

# Read H.E.S.S. Map
hess_map = Map.read(data_path + '/maps/significance_energy_bands_no_J1908.fits.gz')
geom = hess_map.geom.to_image()

m1 = hess_map.slice_by_idx({"energy":slice(0,1)})
m2 = hess_map.slice_by_idx({"energy":slice(1,2)})
m3 = hess_map.slice_by_idx({"energy":slice(2,3)})

# Define the x-ray regions for the plot
w2 = SkyCoord(l = 39.49459783913566, b = -1.7422062350119902, unit='deg', frame='galactic')
w1 = SkyCoord(l = 39.603841536614645, b=-1.9520383693045562, unit='deg', frame='galactic')
e1 = SkyCoord(l = 39.85354141656663, b=-2.664268585131895, unit='degree', frame='galactic')
e2 = SkyCoord(l = 39.91596638655462, b=-2.833333333333333, unit='degree', frame='galactic')
e3 = SkyCoord(l = 40.05402160864346, b=-3.248201438848921, unit='degree', frame='galactic')

w2_pix = geom.to_image().coord_to_pix(w2)
w1_pix = geom.to_image().coord_to_pix(w1)
e1_pix = geom.to_image().coord_to_pix(e1)
e2_pix = geom.to_image().coord_to_pix(e2)
e3_pix = geom.to_image().coord_to_pix(e3)


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

ra_contour_zoom = geom.to_image().coord_to_pix(c)[0]
dec_contour_zoom = geom.to_image().coord_to_pix(c)[1]



# for the colorbars
vmax = 6
vmin = -4
skye1 = plt.imshow(hess_map.data[0], cmap=cmap, vmax=vmax, vmin=vmin)
plt.close()
skye2 = plt.imshow(hess_map.data[1], cmap=cmap, vmax=vmax, vmin=vmin)
plt.close()
skye3 = plt.imshow(hess_map.data[2], cmap=cmap, vmax=vmax, vmin=vmin)
plt.close()

# Containment radius at the center of the map for three energies
cont = [0.05168437035203161, 0.05069044015295409, 0.06460546294003952]*u.deg

# Regions signifying the PSF
l = 40.4*u.deg
b = -3.35*u.deg
lw = 2
path_effects = [PathEffects.withStroke(linewidth=lw+1, foreground="k")]
psf_range1_region = CircleSkyRegion(SkyCoord(l=l, b=b, frame='galactic'),
                            radius=cont[0])
pixel_psf_range1_region= psf_range1_region.to_pixel(geom.wcs)

psf_range2_region = CircleSkyRegion(SkyCoord(l=l, b=b, frame='galactic'),
                            radius=cont[1])
pixel_psf_range2_region= psf_range2_region.to_pixel(geom.wcs)

psf_range3_region = CircleSkyRegion(SkyCoord(l=l, b=b, frame='galactic'),
                            radius=cont[1])
pixel_psf_range3_region= psf_range3_region.to_pixel(geom.wcs)


fig = plt.figure(figsize=(21,7))
ax1 = plt.subplot(1,3,1, projection = geom.wcs)
ax2 = plt.subplot(1,3,2, projection = geom.wcs)
ax3 = plt.subplot(1,3,3, projection = geom.wcs)

fig.subplots_adjust(wspace= 0.05)
ax1.coords[0].set_ticklabel(size = pparams.MEDIUM_SIZE, color='k')
ax1.coords[1].set_ticklabel(size = pparams.MEDIUM_SIZE, color='k')
ax2.coords[0].set_ticklabel(size = pparams.MEDIUM_SIZE, color='k')
ax2.coords[1].set_ticklabel(size = 0, color='k')
ax3.coords[0].set_ticklabel(size = pparams.MEDIUM_SIZE, color='k')
ax3.coords[1].set_ticklabel(size = 0, color='k')
color_c = "white"
edge_c = "k"
ax1 = m1.plot(ax = ax1, add_cbar=False, cmap=cmap, vmax=vmax, vmin=vmin)
cbaxes1 = fig.add_axes([0.32, 0.545, 0.017, 0.25])
cbar1 = plt.colorbar(skye1, ticks = [-2.5, 0., 2.5, 5],orientation="vertical",fraction=0.046,pad=0.03,cax=cbaxes1)
cbar1.outline.set_edgecolor('white')
cbar1.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white', weight = "bold")
cbar1.ax.set_ylabel('', rotation=90)
cbar1.set_label("      Significance" ,  fontsize=pparams.MEDIUM_SIZE-3, color='white', labelpad = -90-5 , weight = "bold", rotation = 90, y = 0.38)

ax1.set_ylabel('Galactic Latitude (J2000)', fontsize=pparams.BIGGER_SIZE)
ax1.text(39.65,-3.4,"0.8 - 2.5 TeV", color = 'white',transform = ax1.get_transform('galactic'), fontsize = pparams.BIGGER_SIZE, fontweight = 'bold')

pixel_psf_range1_region.plot(ax=ax1, edgecolor='white', lw=3)
ax1.text(30,38, s= "PSF", color=color_c, weight='bold', path_effects=path_effects)


ax2 = m2.plot(ax = ax2, add_cbar=False, cmap=cmap, vmax=vmax, vmin=vmin)
cbaxes2 = fig.add_axes([0.585, 0.545, 0.017, 0.25])
cbar2 = plt.colorbar(skye2, ticks = [-2.5, 0., 2.5, 5],orientation="vertical",fraction=0.046,pad=0.03,cax=cbaxes2)
cbar2.outline.set_edgecolor('white')
cbar2.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white', weight = "bold")
cbar2.ax.set_ylabel('', rotation=90)
cbar2.set_label("      Significance" ,  fontsize=pparams.MEDIUM_SIZE-3, color='white', labelpad = -90-5 , weight = "bold", rotation = 90, y = 0.38)

ax2.set_ylabel(' ', fontsize=0)
ax2.text(39.6,-3.4,"2.5 - 10 TeV", color = 'white',transform = ax2.get_transform('galactic'), fontsize = pparams.BIGGER_SIZE, fontweight = 'bold')

pixel_psf_range2_region.plot(ax=ax2, edgecolor='white', lw=3)
ax2.text(30,38, s= "PSF", color=color_c, weight='bold', path_effects=path_effects)

ax3 = m3.plot(ax = ax3, add_cbar=False, cmap=cmap, vmax=vmax, vmin=vmin)
cbaxes3 = fig.add_axes([0.85, 0.545, 0.017, 0.25])
cbar3 = plt.colorbar(skye3, ticks = [-2.5, 0., 2.5, 5],orientation="vertical",fraction=0.046,pad=0.03,cax=cbaxes3)
cbar3.outline.set_edgecolor('white')
cbar3.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white', weight = "bold")
cbar3.ax.set_ylabel('', rotation=90)
cbar3.set_label("      Significance" ,  fontsize=pparams.MEDIUM_SIZE-3, color='white', labelpad = -90-5 , weight = "bold", rotation = 90, y = 0.38)


ax3.set_ylabel(' ', fontsize=0)
ax3.text(39.7,-3.4,"above 10 TeV", color = 'white',transform = ax3.get_transform('galactic'), fontsize = pparams.BIGGER_SIZE, fontweight = 'bold')

pixel_psf_range3_region.plot(ax=ax3, edgecolor='white', lw=3)
ax3.text(30,38, s= "PSF", color=color_c, weight='bold',
        path_effects=path_effects)

for axtemp in [ax1,ax2, ax3]:
    axtemp.scatter(ra_contour_zoom, dec_contour_zoom, color='aqua', s=0.8, alpha=0.3)

    axtemp.coords[0].set_ticks(spacing=1 *u.degree, color = 'white')
    axtemp.coords[1].set_ticks(spacing=1 *u.degree, color = 'white')

    axtemp.coords[0].display_minor_ticks(True)
    axtemp.coords[1].display_minor_ticks(True)
    axtemp.text(40.6,-1.7,"H.E.S.S.", color = 'white',transform = axtemp.get_transform('galactic'), fontsize = pparams.BIGGER_SIZE, fontweight = 'bold',path_effects=path_effects)
    axtemp.set_xlabel('Galactic Longitude (J2000)', fontsize=pparams.BIGGER_SIZE)

    color_c = "white"
    edge_c = "k"
    lw = 2
    s = 150
    path_effects = [PathEffects.withStroke(linewidth=lw+1, foreground="k")]
    axtemp.scatter(w1_pix[0], w1_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
    axtemp.scatter(w2_pix[0], w2_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
    axtemp.scatter(e1_pix[0], e1_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
    axtemp.scatter(e2_pix[0], e2_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)
    axtemp.scatter(e3_pix[0], e3_pix[1], color=color_c, s=s, alpha=1, marker='X', lw=lw, edgecolors=edge_c)

plt.savefig('ss433/plots/fig2.pdf', transparent=True, bbox_inches='tight', pad_inches=0.08)
plt.savefig('ss433/plots/fig2.png', transparent=False, bbox_inches='tight', pad_inches=0.08)

plt.show()

