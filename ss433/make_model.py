import numpy as np
import astropy.units as u
import astropy.constants as const
from spatial_model import AdvectionInstance

## Gte radiation field
xr_west = np.genfromtxt('files/radiation_5p5kpc_west.txt', delimiter=',')
energy_west = np.asarray([i[0] for i in xr_west])
number_density_west = np.asarray([i[1] for i in xr_west])

RADIATION_FIELD_WEST = list(zip(energy_west, (number_density_west)))

xr_east = np.genfromtxt('files/radiation_5p5kpc_east.txt', delimiter=',')
energy_east = np.asarray([i[0] for i in xr_east])
number_density_east = np.asarray([i[1] for i in xr_east])

RADIATION_FIELD_EAST = list(zip(energy_east, (number_density_east)))

########################## Common values #######################################
v0 = 0.08*const.c
e_index = -2
efficiency = 0.00129
Ee_cut = 630.957*u.TeV
Ee_min = 1e-5*u.TeV
N_g = 50
Eg_min = 1e-10*u.TeV
t_step = 4*u.yr
t_max = 1e4*u.yr
d0 = 5e27*u.cm**2/u.s

########################## western jet #########################################
v = np.loadtxt("ss433/velocity_profiles/western_jet_velocity.txt", delimiter=',')
v_x = v[:,0]*u.pc
v_y = (v0*v[:,1]).to('pc/yr')

B_w = 21.07*u.uG

west = AdvectionInstance(v_profile_x=v_x,
                        v_profile_y = v_y,
                        e_index=e_index,
                        efficiency=efficiency,
                        B=B_w,
                        radiation_field = RADIATION_FIELD_WEST,
                        Ee_cut=Ee_cut,
                        Ee_min=Ee_min,
                        Eg_min = Eg_min,
                        N_g = N_g,
                        t_step = t_step,
                        t_max = t_max,
                        d0=d0,
                        meta = {"which" : "west"})

_ = west.electron_dNdE
_ = west.photon_distribution


west.write("ss433/west.pkl", overwrite=True)


########################## eastern jet #########################################
v = np.loadtxt("ss433/velocity_profiles/eastern_jet_velocity.txt", delimiter=',')
v_x = v[:,0]*u.pc
v_y = (v0*v[:,1]).to('pc/yr')

B_e = 19.49*u.uG

east = AdvectionInstance(v_profile_x=v_x,
                        v_profile_y = v_y,
                        e_index=e_index,
                        efficiency=efficiency,
                        B=B_e,
                        radiation_field = RADIATION_FIELD_EAST,
                        Ee_cut=Ee_cut,
                        Ee_min=Ee_min,
                        Eg_min = Eg_min,
                        N_g = N_g,
                        t_step = t_step,
                        t_max = t_max,
                        d0=d0,
                        meta = {"which" : "east"})

_ = east.electron_dNdE
_ = east.photon_distribution


east.write("ss433/east.pkl", overwrite=True)
