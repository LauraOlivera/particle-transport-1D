import gappa as gp
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from gammapy.maps import MapAxis
import functools
import pickle
import math
import warnings
import os

names =  ['B', 'radiation_field', 'd0', 'dindex', 'E_ref_d',
        'v_profile_x', "v_profile_y", 'e_index', 't_max', 't_step', 'Ee_min',
        'Ee_cut', 'Eg_min', 'Eg_max', 'N_e', 'N_g', 'distance',
        'name', 'meta', 'jet_luminosity', 'efficiency']

## default radiation field
xr = np.genfromtxt(os.path.expandvars('$PROF_PATH/radiation_field/radiation_5p5kpc.txt'), delimiter=',')
energy = np.asarray([i[0] for i in xr])
number_density = np.asarray([i[1] for i in xr])
RADIATION_FIELD = list(zip(energy, (number_density)))


def interpolate_log10(x_target, x_data, y_data, filler=0):
    """ Interpolate arrays with log spacing.
    Inputs should NOT have units, you need to take care
    of them before so that the result is what you need
    it to be.

    If some of the x_target nodes are outside of the
    range of x_data, the result will the value given by
    the input filler, default is zero.

    Parameters:
    ----------
    x_target : ~np.array
        array of x values at which we want the quantity interpolated
    x_data: ~np.array
            array of x values we have
    y_data: ~np.array
            array of y values we have
    filler : float
            value with which to fill values outside of interpolation bounds

    Returns:
    -------
    result : ~np.array
        array with the interpolated quantity
    """

    log10_x_target = np.log10(x_target)
    log10_x_data = np.log10(x_data)
    log10_y_data = np.log10(y_data)

    log_interp = np.interp(log10_x_target, log10_x_data, log10_y_data,  left=np.nan, right=np.nan)

    result = 10**log_interp

    result[np.isnan(result)] = filler
    return result

def get_tcool_e(E_e,total, ic, synch, which='total'):
    """ Given pre-computed cooling times, interpolate the
    value to the desire electron energy.

    Parameters:
    ----------
    E_e : ~astropy.quantity
        energy (or array of energies) at which we want the cooling time
    total: ~np.array
            total cooling time in yr as a function of electron energy
    ic : ~np.array
            IC cooling time  in yr as a function of electron energy
    synch : ~np.array
            Synchrotron cooling time in yr as a function of electron energy
    which : str
            whether to return the total cooling time ('total'), or also the separate IC
            and synchrotron cooling times ('all')

    Returns:
    -------
    total : ~astropy.quantity
            total cooling time at E_e
    ic : (if which = 'all'), ~astropy.quantity
            IC cooling time at E_e
    synch : (if which = 'all'), ~astropy.quantity
            synchrotron cooling time at E_e
    """
    E_e = E_e.to_value('erg')

    if which=='total':
        total = interpolate_log10(E_e, total[:,0], total[:,1])*u.yr
        return total
    elif which=='all':
        total = interpolate_log10(E_e, total[:,0], total[:,1])*u.yr
        ic = interpolate_log10(E_e, ic[:,0], ic[:,1])*u.yr
        synch = interpolate_log10(E_e, synch[:,0], synch[:,1])*u.yr
        return total, ic, synch


class AdvectionInstance():
    """ Bundle together all the parameters required to produce profiles.

    Parameters:
    ----------
    B : ~astropy.quantity
        magnetic field
    radiation_field : ~list
                    Number density of the target photon field in cm-3 erg-1
    d0 : ~astropy.quantity [cm2/s]
            value of the coefficient at E_ref
    dindex : float
            index describing the dependence of coefficient in energy
    E_ref_d : ~astropy.quantity
            reference energy for the diffusion
    v_profile_x, v_profile_y : position as astropy quantities
                The velocity profile x and y values
    e_index : float
            injected electron spectrum. It is negative!
    jet_luminosity : ~astropy.quantity
            total power of the jet. Will be used to normalized injected spectrum
    efficiency : float
            which fraction of the total power goes to electrons
    t_max : ~astropy.quantity
            total time
    t_step : ~astropy.quantity
            time interval
    Ee_min, Ee_cut : ~astropy.quantity
            minimum and cutoff injected electron energies (given separately)
    Eg_min, Eg_max : ~astropy.quantity
            minimum and maximum photon energies (given separately)
    N_e, N_g : int
            number of energy bins for electrons and gammas (given separately)
    N_z : int
            number of bins for the spatial axis
    distance :  ~astropy.quantity
                distance
    name : str
            name to identify this instance
    meta : dict
            dictionary containing extra info
    """

    def __init__(self,
        B =16*u.uG,
        radiation_field = RADIATION_FIELD,
        d0=1e27*u.cm**2/u.s,
        dindex=0.33,
        E_ref_d=1*u.TeV,
        v_profile_x = None,
        v_profile_y = None,
        e_index = -2,
        jet_luminosity = 1e39*u.erg/u.s,
        efficiency = 1e-3,
        t_max = 3e4*u.yr,
        t_step=4.0*u.yr,
        Ee_min = 1*u.TeV,
        Ee_cut = 200*u.TeV,
        Eg_min = 1e-18*u.TeV,
        Eg_max = 150*u.TeV,
        N_e = 100,
        N_g = 100,
        N_z = 200,
        distance = 5.5*u.kpc,
        name = None,
        meta = {"delta":False}):

        self.B = B
        self.radiation_field = radiation_field
        self.d0 = d0
        self.dindex = dindex
        self.E_ref_d = E_ref_d
        self.v_profile_x = v_profile_x
        self.v_profile_y = v_profile_y
        self.e_index = e_index
        self.jet_luminosity = jet_luminosity
        self.efficiency = efficiency
        self.t_max = t_max
        self.t_step = t_step
        self.Ee_min = Ee_min
        self.Ee_cut = Ee_cut
        self.Eg_min = Eg_min
        self.Eg_max = Eg_max
        self.N_e = N_e
        self.N_g = N_g
        self.N_z = N_z
        self.distance = distance
        self.name = name
        self.meta = meta
        self._electron_dNdE = None
        self._photon_distribution = None

    @property
    def electron_energies(self):
        """Axis of electron energies. Uses Gammapy MapAxis """

        edge_bins = np.logspace(np.log10(self.Ee_min.to_value('TeV')),
                        np.log10(self.Ee_cut.to_value('TeV'))+1,
                        self.N_e+1)*u.TeV

        return MapAxis.from_edges(edge_bins, interp='log')

    @property
    def injected_EdNdE(self):
        """Injected electron distribution in EdNdE form """

        e_electrons = self.electron_energies.center

        exp_cutoff = np.exp(-(e_electrons/self.Ee_cut.to('TeV')).value)
        power_law = (e_electrons.to_value('TeV') ** (self.e_index))*exp_cutoff
        power_law *=e_electrons.to_value('TeV') # because we are doing EdNdE
        norm = self.efficiency*self.jet_luminosity.to_value('TeV s-1')
        fu = gp.Utils()
        power_law *= norm/fu.Integrate(list(zip(e_electrons.to_value('TeV'),power_law)))
        return power_law*u.s**-1

    @property
    def injected_dNdE(self):
        """Injected electron distribution in dNdE form """

        e_electrons = self.electron_energies.center

        exp_cutoff = np.exp(-(e_electrons/self.Ee_cut.to('TeV')).value)
        power_law = (e_electrons.to_value('TeV') ** (self.e_index))*exp_cutoff
        norm = self.efficiency*self.jet_luminosity.to_value('TeV s-1')
        fu = gp.Utils()
        power_law *= norm/fu.Integrate(list(zip(e_electrons.to_value('TeV'),e_electrons.to_value('TeV')*power_law)))
        return power_law*u.TeV**-1*u.s**-1

    def inject_delta_to_test(self, E=20*u.TeV):
        """Delta function for tests """

        e_electrons = self.electron_energies.center
        idx = (np.abs(e_electrons - E)).argmin()
        weights = np.zeros(len(e_electrons))
        weights[idx] = (self.efficiency*self.jet_luminosity*self.t_step).to_value('TeV')
        return weights

    def get_velocity(self,x):
        """ For a given array with x values, get the velocity from the
        input velocit profile

        Parameters
        ----------
        x : np.array, units equivalent to pc
            x values at which we want the velocity

        Returns
        -------
        v : np.array, units of pc yr-1
            velocity in x
        """
        v_x = self.v_profile_x.to_value('pc')
        v_y = self.v_profile_y.to_value('pc yr-1')

        this_v = np.interp(x.to_value('pc'), v_x, v_y)*u.pc*u.yr**-1

        return this_v

    def plot_velocity(self, x = np.linspace(0,100,100)*u.pc, ax=None, **kwargs):
        """Plot the velocity profile

        Parameters
        ----------
        x : np array, with units equivalent to pc
            x values in which to plot
        ax : matplotlib ax
             Default is None, so a new axis is created
        fontsize : int
            font size for the injection parameters text
        **kwargs : any parameters passes to plt.loglog

        Returns:
        --------
        ax : matplotlib ax
        """
        plt.figure(figsize=(8,6))
        if ax is None:
            ax = plt.gca()
        ax.plot(x, ((self.get_velocity(x))/const.c).to_value(""), **kwargs)
        ax.set_xlabel('Distance [pc]')
        ax.set_ylabel('Velocity [c]')
        return ax

    def plot_injected_spectrum(self, ax=None, fontsize=14, **kwargs):
        """Plot the injected spectrum

        Parameters
        ----------
        ax : matplotlib ax
             Default is None, so a new axis is created
        fontsize : int
            font size for the injection parameters text
        **kwargs : any parameters passes to plt.loglog

        Returns:
        --------
        ax : matplotlib ax
        """
        power_law = self.injected_dNdE
        e_electrons = self.electron_energies.center
        plt.figure(figsize=(8,6))
        if ax is None:
            ax = plt.gca()
        ax.loglog(e_electrons.to_value('TeV'),power_law, **kwargs )
        ax. text(0.65, 0.75,
                 'efficiency = ' + str(self.efficiency) + "\n power = " + str(self.jet_luminosity) + " \n index = " + str(self.e_index),
                  horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=fontsize)
        ax.set_xlabel('electron energy [TeV]')
        ax.set_ylabel('dN/dE [TeV-1 s-1]')
        return ax

    def make_tcool_e(self):
        """" Compute the cooling times for a given B and radiation field

        Returns:
        --------
        total: ~np.array
                total cooling time in yr as a function of electron energy
        ic : ~np.array
                IC cooling time  in yr as a function of electron energy
        synch : ~np.array
                Synchrotron cooling time in yr as a function of electron energy
        """

        # Define the electron energy array
        e = self.electron_energies.edges # here it is edges because we want the range to exist!

        e = e.to_value('erg')

        # Get magnetic field
        B = self.B.to_value('G') # in Gauss

        # set up particle spectrum with random environmental parameters
        fp = gp.Particles()
        fp.SetType("electrons")

        # define the radiation field and magnetic field
        fp.AddArbitraryTargetPhotons(self.radiation_field)
        fp.SetBField(B)

        # extract the cooling time scales at energy points 'e'
        total = np.array(fp.GetCoolingTimeScale(e,"sum"))
        ic = np.array(fp.GetCoolingTimeScale(e,"inverse_compton"))
        synch = np.array(fp.GetCoolingTimeScale(e,"synchrotron"))

        return total, ic, synch

    def plot_tcool_e(self, ax=None, **kwargs):
        """Plot the cooling times

        Parameters
        ----------
        ax : matplotlib ax
             Default is None, so a new axis is created
        **kwargs : any parameters passes to plt.loglog

        Returns:
        --------
        ax : matplotlib ax
        """
        total, ic, synch = self.make_tcool_e()

        f = plt.figure(figsize=(8,6))
        if ax is None:
            ax = plt.gca()
        ax.loglog(ic[:,0],ic[:,1],c="orange",label="IC", **kwargs)
        ax.loglog(synch[:,0],synch[:,1],c="blue",label="synch", **kwargs)
        ax.loglog(total[:,0],total[:,1],c="black",ls="--",label="sum", **kwargs)
        ax.set_title('B = ' + str(self.B.to('G')))
        ax.set_xlabel("Electron energy [erg]")
        ax.set_ylabel("Cooling time scale [yrs]")
        ax.legend()

        return ax

    def diffuse(self, energy):
        """Compute the diffusion coefficient at a given energy

        Parameters:
        ----------
        energy: ~astropy.quantity
                energy at which we want the coefficient

        Returns:
        -------
        d : ~astropy.quantity
            diffusion cioefficient [cm2/s]

        """
        a = (energy/self.E_ref_d).to_value("")
        d = self.d0*a**self.dindex
        return d

    def plot_diffuse(self, ax=None, fontsize=14, **kwargs):
        """Plot the diffusion coefficient

        Parameters
        ----------
        ax : matplotlib ax
             Default is None, so a new axis is created
        fontsize : int
            font size for the injection parameters text
        **kwargs : any parameters passes to plt.loglog

        Returns:
        --------
        ax : matplotlib ax
        """
        e_electrons = self.electron_energies.center


        d = self.diffuse(e_electrons)

        f = plt.figure(figsize=(8,6))
        if ax is None:
            ax = plt.gca()
        plt.loglog(e_electrons,d,c="k", **kwargs)
        ax. text(0.35, 0.75,
            'D0 = ' + str(self.d0) + "\n E_ref = " + str(self.E_ref_d) + " \n d_index = " + str(self.dindex),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=fontsize)
        ax.set_xlabel('Electron energy [TeV]')
        ax.set_ylabel('Diffusion coefficient [cm2/s]')
        return ax


    def get_electron_dNdE(self):
        """Compute the electron distribution at t_max

        Returns:
        --------
        electron_distr : ~astropy.quantity, array
                        2D distribution of differential number of electrons
                        in energy and distance
        x_edges : ~astropy.quantity, array
                    edges of the spatial bins
        energy_electrons : ~gammapy.maps.MapAxis
        """

        print('Preparing to compute the electron distribution')
        # define the time nodes
        t_step = self.t_step
        t_max = self.t_max

        t = 0*u.yr # start at zero

        # define the electron energies
        e_electrons = self.electron_energies.center


        # compute the cooling time
        total, ic, synch = self.make_tcool_e()
        if total[:,1].min()<t_step.to_value('yr'):
            warnings.warn('The timestep you provided is longer than the cooling time at the highest energies \n This might cause weirdness in that energy range')

        # initial position (all the particles at zero because we just injected it)
        x = np.zeros(self.N_e)*u.pc # the position of the electrons at each energy

        # define the shape of the injected electron spectrum
        power_law = self.injected_EdNdE# this is per second

        # inject the right amount of energy given the time step
        power_law *= t_step.to('s')
        # power_law = power_law.to_value('TeV-1')
        power_law = power_law.to_value('')

        # prepare empty array to hold the energies, position and weights
        x_array = np.array([])*x.unit
        energ_e = np.array([])*e_electrons.unit
        weights = np.array([])


        while t<t_max:
            print('time ' + str(t) + ' out of ' + str(t_max))
            # a new batch of particles is injected at 0 in every time step
            x_array = np.append(x_array, x)
            energ_e = np.append(energ_e, e_electrons)
            if "delta" in self.meta.keys() and self.meta['delta']:
                 weights = np.append(weights, self.inject_delta_to_test())
            else:
                 weights = np.append(weights, power_law)

            # get the velocity that is local to each energy
            v = self.get_velocity(x_array)

            # advect
            x_array += v*t_step

            # diffuse
            d = self.diffuse(energ_e)

            r = np.sqrt(2*d*t_step.to('s')).to_value('pc')
            x_array += np.random.normal(loc=0.0, scale=r, size=len(x_array))*u.pc

            # cool
            t_cool_tot= get_tcool_e(energ_e,total, ic, synch, which='total')

            # Uncomment to check the cooling times
            # ax = self.plot_tcool_e()
            # ax.scatter(energ_e.to('erg'), t_cool_tot.to('yr'))
            # plt.show()

            # if the energy is too low, the tcool would be out of
            # interpolation bounds and thus zero.... need to rethink this
            # for now let's just say that if energy is too low,
            # that is energ_e < E_min_e, then no more cooling for these particles
            # which is reasonable because the cooling times would be really long
            # anyway. But this might be an issue if you pick very low energies
            # and care for very long times.

            cooling_factor =1/(1+(t_step/t_cool_tot))
            cooling_factor[energ_e<self.Ee_min] = 1


            energ_e *= cooling_factor


            t+=t_step



        # make the histogram of electron energies and positions
        # we ignore the particles with energy lower than the input Ee_min
        E_edges = np.log10(self.electron_energies.edges.to_value('TeV'))
        x_edges = np.linspace(x_array.min(),x_array.max(),self.N_z) # in pc

        electron_distr, x_edges, E_edges = np.histogram2d(x_array.to_value('pc'),
                                np.log10(energ_e.to_value('TeV')),
                                bins = (x_edges.to_value('pc'), E_edges),
                                weights =weights)

        electron_distr = electron_distr.T
        x_edges *=u.pc

        electron_distr_dNdE =electron_distr*self.electron_energies.center[:, np.newaxis]**-1

        return electron_distr_dNdE, x_edges

    @property
    @functools.lru_cache()
    def electron_dNdE(self):
        """ Histogram of electron distribution as a function of energy
        and distance from injection point

        Returns:
        --------
        electron_distr : ~astropy.quantity, array
                        2D distribution of differential number of electrons
                        in energy and distance
        x_edges : ~astropy.quantity, array
                    edges of the spatial bins
        energy_electrons : ~gammapy.maps.MapAxis
                    axis describing the electron binning
        """

        if self._electron_dNdE is None:
            self.electron_dNdE = self.get_electron_dNdE()
        return self._electron_dNdE

    @electron_dNdE.setter
    def electron_dNdE(self, value=None):
        self._electron_dNdE = value



    def radiate(self):
        """
        Radiate

        Returns:
        --------
        ic_distr : ~astropy.quantity, array
                        2D distribution of differential number of IC photons
                        in energy and distance
        synch_distr : ~astropy.quantity, array
                        2D distribution of differential number of SC photons
                        in energy and distance
        photon_energies_axis : ~gammapy.maps.MapAxis
                    axis describing the photon binning

        """
        # access the electron distribution
        electron_distr, x_edges = self.electron_dNdE

        energy_electrons = self.electron_energies

        # get the quantities to the right units
        B = self.B.to_value('G')
        distance = self.distance.to_value('pc')

        # make empty arrays to contain the photon quantities
        ic_distr = np.zeros((self.N_g, electron_distr.shape[1]))
        synch_distr = np.zeros((self.N_g, electron_distr.shape[1]))

        # set Gamera
        fr = gp.Radiation()
        fr.AddArbitraryTargetPhotons(self.radiation_field)
        fr.SetBField(B)
        fr.SetDistance(distance)
        fr.ToggleQuietMode()

        # Loop over the different spatial positions of the electron distr
        for idx, distr in enumerate(electron_distr.T):
            print(str(idx) + ' out of ' + str(electron_distr.T.shape[0]) + " spatial steps")
            # get the spectrum at that position in a way that gamera likes
            e_energy = energy_electrons.center.to_value('erg')
            distr = distr.to_value('erg-1')
            electron_spectrum = np.array(list(zip(e_energy, distr)))

            fr.SetElectrons(electron_spectrum)

            # Define the photon energies at which we'll get the spectrum
            Eg_min_log = np.log10(self.Eg_min.to_value('TeV'))
            Eg_max_log = np.log10(self.Eg_max.to_value('TeV'))

            photon_energies = np.logspace(Eg_min_log,Eg_max_log,self.N_g) * u.TeV
            photon_energies = photon_energies.to_value('erg')

            # Compute photon spectrum at those energies
            fr.CalculateDifferentialPhotonSpectrum(photon_energies)

            # Get the different contributions
            # the units will be 1 / erg / cm^2 / s vs erg
            tot = np.array(fr.GetTotalSpectrum())
            ic = np.array(fr.GetICSpectrum())
            synch = np.array(fr.GetSynchrotronSpectrum())

            # fill the photon distributions at this location
            ic_distr[:,idx] = interpolate_log10(photon_energies, ic[:,0], ic[:,1])
            synch_distr[:,idx] = interpolate_log10(photon_energies, synch[:,0], synch[:,1])

        # add the right units to all quantities
        ic_distr *=u.erg**-1*u.cm**-2*u.s**-1
        synch_distr *=u.erg**-1*u.cm**-2*u.s**-1

        ic_distr = ic_distr.to('TeV-1 cm-2 s-1')
        synch_distr = synch_distr.to('TeV-1 cm-2 s-1')

        photon_energies *= u.erg

        # make a gammapy axis with photon energies
        photon_energies_axis = MapAxis.from_nodes(photon_energies.to_value('TeV'), unit='TeV', interp="log")


        return ic_distr, synch_distr, photon_energies_axis


    @property
    @functools.lru_cache()
    def photon_distribution(self):
        """ Histograms of photon distribution as a function of energy
        and distance from injection point for both IC and synchrotron

        Returns:
        --------
        ic_distr : ~astropy.quantity, array
                        2D distribution of differential number of IC photons
                        in energy and distance
        synch_distr : ~astropy.quantity, array
                        2D distribution of differential number of SC photons
                        in energy and distance
        photon_energies_axis : ~gammapy.maps.MapAxis
                    axis describing the photon binning
        """

        if self._photon_distribution is None:
            self._photon_distribution = self.radiate()
        return self._photon_distribution

    @photon_distribution.setter
    def photon_distribution(self, value=None):
        self._photon_distribution = value

    def pc_to_deg(self, array):
        rad_array = (array/(self.distance)).to_value('')

        deg_array = np.rad2deg(rad_array)
        return deg_array

    def plot_electron_distribution(self, e_factor = 0, **kwargs):
        """Plot the resulting electron distribution. If it has not been
        already computed and cached, it is computed here.

        Parameters
        ----------
        e_factor : power of E to multiply distirbution by
        **kwargs : any parameters passes to plt.imshow

        Returns:
        --------
        ax : matplotlib ax
        """
        electron_distr, x_edges = self.electron_dNdE
        energy_electrons = self.electron_energies
        E_edges = np.log10(energy_electrons.edges.value)
        plt.figure(figsize=(10,6))
        ax = plt.gca()
        en = energy_electrons.center[:,None]**e_factor
        plt.imshow((en*electron_distr).value, interpolation='nearest', origin='lower', aspect='auto',
                    extent=[x_edges[0].value, x_edges[-1].value, E_edges[0], E_edges[-1]],  **kwargs
                )

        plt.colorbar()
        # plt.ylim(-0.2,2.4)
        plt.tick_params(which='both')
        plt.xlabel("distance from injection [pc]")
        plt.ylabel("energy [log10(TeV)]")
        # plt.show()

        return ax


    def plot_photon_distribution(self, e_factor = 0, **kwargs):
        """Plot the resulting photon distribution. If it has not been
        already computed and cached, it is computed here.

        Parameters
        ----------
        e_factor : power of E to multiply distirbution by
        **kwargs : any parameters passes to plt.imshow

        Returns:
        --------
        ax : matplotlib ax
        """
        ic_distr, synch_distr, photon_energies_axis = self.photon_distribution
        _, x_edges = self.electron_dNdE

        plt.figure(figsize=(10,6))
        ax = plt.gca()
        en = photon_energies_axis.center[:,None]**e_factor

        plt.imshow((en*(synch_distr+ic_distr)).value, interpolation='nearest', origin='lower', aspect='auto',
            extent=[x_edges[0].value, x_edges[-1].value, np.log10(photon_energies_axis.edges[0].value),
                    np.log10(photon_energies_axis.edges[-1].value)], **kwargs)
        plt.colorbar()
        plt.tick_params(which='both')
        plt.xlabel("distance from injection [pc]")
        plt.ylabel("energy [log10(TeV)]")
        # plt.show()
        return ax

    def get_total_SED(self):
        """Get the total SED E^2dNdE


        Returns:
        --------
        SED : quantity
        """
        ic_distr, synch_distr, photon_energies_axis = self.photon_distribution
        em = ic_distr + synch_distr
        photon_dNdE = em.sum(axis=1)

        return photon_energies_axis, photon_energies_axis.center**2*photon_dNdE

    def plot_SED(self,ax=None, total=False):
        """Plot the resulting SED. If it has not been
        already computed and cached, it is computed here.


        Returns:
        --------
        ax : matplotlib ax
        """
        ic_distr, synch_distr, photon_energies_axis = self.photon_distribution
        _, x_edges = self.electron_dNdE
        x_center = x_edges[:-1] + (x_edges[2]-x_edges[1])/2

        spatial_steps = len(x_center)

        ic = 0
        n = int(math.ceil(spatial_steps/10))
        color = plt.cm.viridis(np.linspace(0, 1,n))

        plt.figure(figsize=(10,8))
        if ax is None:
            ax = plt.gca()
        tot = np.zeros(len(photon_energies_axis.center))*synch_distr.unit
        for idx in np.arange(ic_distr.shape[1]):

            em = ic_distr[:,idx] + synch_distr[:,idx]
            tot += em

            max_s = photon_energies_axis.center[np.nanargmax(photon_energies_axis.center**2*synch_distr[:,idx])]
            max_ic = photon_energies_axis.center[np.nanargmax(photon_energies_axis.center**2*ic_distr[:,idx])]

            if (em.value < 1e-100).all():
                continue

            if not idx%10 ==0:
                continue
            if not total:
                ax.loglog(photon_energies_axis.center,
                          (photon_energies_axis.center**2*em).to("erg s-1 cm-2"),
                          color=color[ic],
                          label=round(x_center[idx].value,2)*u.pc)
                ax.axvline(max_s.value, ls='--', color=color[ic])
                ax.axvline(max_ic.value, ls='--', color=color[ic])

            ic+=1

        ax.loglog(photon_energies_axis.center,
                  (photon_energies_axis.center.to('TeV')**2*tot).to("erg s-1 cm-2"),
                  color='red', label="total")

        ax.legend()
        ax.set_xlabel('Energy [TeV]')
        ax.set_ylabel('$E^{2}\cdot \frac{dN}{dE}$ [erg s-1 cm-2]')
        ax.set_ylim((photon_energies_axis.center.to('TeV')**2*tot).to_value("erg s-1 cm-2").min())
        ax.set_xlim(1e-14, 1e3)
        # plt.show()
        return ax

    def get_profile_energy_range(self, e_min, e_max):
        """Get the spatial flux profile for a given energy range in units of cm-2 s-1.
            NOT CONVOLVED WITH PSF YET

        Parameters:
        -----------
        e_min : astropy.quantity
                low energy bound, units of energy
        e_max : astropy.quantity
                high energy bound, units of energy
                """
        energy_axis_edges = np.logspace(np.log10(e_min.to_value('TeV')), np.log10(e_max.to_value('TeV')), 50)*u.TeV
        energy_axis = MapAxis.from_edges(energy_axis_edges, name='energy', interp='log')

        ic_distr, synch_distr, photon_energies_axis = self.photon_distribution
        tot_distr = ic_distr + synch_distr
        _, x_edges = self.electron_dNdE
        x_center = x_edges[:-1] + (x_edges[2]-x_edges[1])/2
        profile = np.zeros(len(x_center))*u.cm**-2*u.s**-1

        for idx, distr in enumerate(tot_distr.T):
            this_distr = interpolate_log10(energy_axis.center.to_value('TeV'),
                                           photon_energies_axis.center.to_value('TeV'),
                                           distr.value)*distr.unit

            value = (energy_axis.bin_width*this_distr).sum()

            profile[idx] = value.to('cm-2 s-1')

        return profile, x_center


    def write(self, filename):
        """"Save the class instance as a dictionary (.pkl)

        Parameters:
        -----------
        filename : str
                path and file .pkl in which to save

        """
        dictionary = self.__dict__.copy()

        if dictionary['_electron_dNdE'] is not None:
            electron_distr, x_edges = dictionary['_electron_dNdE']
            dictionary['electron_dNdE'] = electron_distr
            dictionary['x_edges'] = x_edges
        if dictionary['_photon_distribution'] is not None:
            ic_distr, synch_distr, photon_energies_axis = dictionary['_photon_distribution']
            dictionary['ic_distr'] = ic_distr
            dictionary['synch_distr'] = synch_distr
            dictionary['energy_photons'] = photon_energies_axis

        with open(filename, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def from_dict(cls, d):
        """ Load an AdvectionInstance from a dictionary """
        df = {k : v for k, v in d.items() if k in names}
        return cls(**df)


    @classmethod
    def read(cls, filename):
        """ Read from a file (.pkl)

        Parameters:
        -----------
        filename : str
                path and file .pkl from which to read
        """
        with open(filename, 'rb') as handle:
            dictionary  = pickle.load(handle)

        new = cls.from_dict(dictionary)

        if '_photon_distribution' in dictionary.keys():
            new.photon_distribution = dictionary['_photon_distribution']
        if '_electron_dNdE' in dictionary.keys():
            new.electron_dNdE = dictionary['_electron_dNdE']
        return new

