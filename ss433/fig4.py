import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.patheffects as PathEffects
from gammapy.estimators import FluxPoints
import plot_params as pparams
import scipy
from spatial_model import AdvectionInstance
from ss433.prepare_model import prepare_model


east = AdvectionInstance.read('ss433/east.pkl')
west = AdvectionInstance.read('ss433/west.pkl')


def prepare_data():
    x_axis_vertical = np.load("ss433/hess_data/profiles_along_jet/x_axis.npy")*u.deg
    data_points =FluxPoints.read("ss433/hess_data/profiles_along_jet/SS433_flux_profile_along_jet_J1908_subtracted.fits", format='profile')
    data_points.sqrt_ts_threshold_ul = 0

    # get flux and eror and flip correctly
    flux = np.flip(data_points.flux.data, axis=0)
    flux_err = np.flip(data_points.flux_err.data, axis=0)
    x = np.flip(x_axis_vertical)

    # split energy tanges
    data_low = flux[:,0,0,0].copy()
    data_error_low = flux_err[:,0,0,0].copy()
    data_mid = flux[:,1,0,0].copy()
    data_error_mid = flux_err[:,1,0,0].copy()

    data_high = flux[:,2,0,0].copy()
    data_error_high = flux_err[:,2,0,0].copy()

    return x, np.stack([data_low, data_mid, data_high]), np.stack([data_error_low, data_error_mid, data_error_high])

x, data, data_err = prepare_data()

# axis for plotting
final_x_m = np.linspace(-1.5, 1.5,500)
x_model, models_best = prepare_model(east, west,x)

for idx, i in enumerate(models_best):
    f0 = scipy.interpolate.interp1d(x_model, i, fill_value=0, bounds_error=False)
    models_best[idx] = f0(final_x_m)*models_best.unit

# read the systematics band
systematics = np.loadtxt('ss433/fig4_systematics.txt', delimiter=',')

x_syst = systematics[:,0]*u.deg
low_energy_1 = systematics[:,1]
low_energy_2 = systematics[:,2]
mid_energy_1 = systematics[:,3]
mid_energy_2 = systematics[:,4]
high_energy_1 = systematics[:,5]
high_energy_2 = systematics[:,6]


# Functions to make the axis in pc
d = 5.5*u.kpc
def ang_to_dist(ang):
    """
    Angle to distance in pc
    """
    return (((ang*u.deg).to_value('rad')*d)).to_value('pc')


def convert_ax_ang_to_dist(ax):
    """
    Update second axis according with first axis.
    """
    x1, x2 = ax.get_xlim()
    ax_pc.set_xlim(ang_to_dist(x1), ang_to_dist(x2))
    ax_pc.figure.canvas.draw()

# x-ray interaction regions to mark in the plot
ss433 = SkyCoord(287.9497,4.9806, unit="deg", frame="icrs").galactic
w2 = SkyCoord(l = 39.49459783913566, b = -1.7422062350119902, unit='deg', frame='galactic')
w1 = SkyCoord(l = 39.603841536614645, b=-1.9520383693045562, unit='deg', frame='galactic')
e1 = SkyCoord(l = 39.85354141656663, b=-2.664268585131895, unit='degree', frame='galactic')
e2 = SkyCoord(l = 39.91596638655462, b=-2.833333333333333, unit='degree', frame='galactic')
e3 = SkyCoord(l = 40.05402160864346, b=-3.248201438848921, unit='degree', frame='galactic')
x_reg = [e1, e2, e3, w1, w2]
x_reg_lab =  ["e1", "e2", "e3", "w1", "w2"]
x_reg_distances = []
for temp in x_reg:
    ang_sep = ss433.separation(temp).deg
    if temp.b > ss433.b:
        ang_sep = -ang_sep
    x_reg_distances.append(ang_sep)


# And finally, plot
fig, axes = plt.subplots(3,1,figsize=(12,10))
fig.subplots_adjust(hspace= 0.3)

f = 1e-14 # absorb this factor into axes

ax_pc = axes[0].twiny()
axes[0].callbacks.connect("xlim_changed", convert_ax_ang_to_dist)
ax_pc.set_xlim(ang_to_dist(-0.8), ang_to_dist(0.8))
ax_pc.set_xlabel('Distance from central binary (pc)')
ax_pc.tick_params(axis='x', which='major', pad=4)

for ax in axes:
    for xray in x_reg_distances:
        ax.axvline(xray, ls=':', color='k', lw=4, alpha=0.4)

axes[0].plot(final_x_m,models_best[0]/f,color='teal',lw=3, ls='-')
axes[0].fill_between(final_x_m, low_energy_1/f, low_energy_2/f, color='teal', alpha=0.3)
axes[0].errorbar(x.value,data[0]/f, yerr = data_err[0]/f, ls='', color='teal', elinewidth=2.3, marker='o', label='0.8 - 2.5 TeV')
axes[0].legend(loc='lower right')


axes[1].plot(final_x_m,models_best[1]/f,color='goldenrod',lw=3, ls='-',)
axes[1].errorbar(x.value,data[1]/f, yerr = data_err[1]/f, ls='', color='goldenrod', label='2.5 - 10 TeV', elinewidth=2.3, marker='o')
axes[1].fill_between(final_x_m, mid_energy_1/f, mid_energy_2/f, color='goldenrod', alpha=0.3)
axes[1].legend(loc='lower right')


axes[2].plot(final_x_m,models_best[2]/f,color='darkred',lw=3, ls='-')
axes[2].errorbar(x.value,data[2]/f, yerr = data_err[2]/f, ls='', color='darkred', label='above 10 TeV', elinewidth=2.3, marker='o')
axes[2].fill_between(final_x_m, high_energy_1/f, high_energy_2/f, color='darkred', alpha=0.3)
axes[2].legend(loc='lower center')

axes[2].set_xlabel('Distance from central binary (deg)')
axes[1].set_ylabel(r'Photon flux (10$^{-14}$ · s$^{-1}$ · cm$^{-2}$)', labelpad=20)


axes[2].set_xlim(-0.8,0.8)
axes[1].set_xlim(-0.8,0.8)
axes[0].set_xlim(-0.8,0.8)

axes[0].set_ylim(-9,18)
axes[1].set_ylim(-1.5,4)
axes[2].set_ylim(-0.6, 0.9)

axes[0].text(-0.75, 1.1e-13/f, "west", weight='bold', fontsize=17)
axes[0].text(0.65, 1.1e-13/f, "east", weight='bold', fontsize=17)

fig.text(0.23,0.15, "w2", weight='bold', rotation=90, color='gray')
fig.text(0.34,0.15, "w1", weight='bold', rotation=90, color='gray')
fig.text(0.705,0.15, "e1", weight='bold', rotation=90, color='gray')
fig.text(0.795,0.15, "e2", weight='bold', rotation=90, color='gray')

plt.savefig('ss433/plots/fig4.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0.1)
plt.savefig('ss433/plots/fig4.png', transparent = False, bbox_inches = 'tight', pad_inches = 0.1)

plt.show()