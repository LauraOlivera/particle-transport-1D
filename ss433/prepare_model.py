import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian1DKernel


jet_starts_e = SkyCoord(l = 39.79517695,b =  -2.50072695, unit='deg', frame='galactic')
jet_starts_w = SkyCoord(l=39.61333002, b=-2.01249932, unit='deg', frame='galactic')
ss433 = SkyCoord(287.9497,4.9806, unit="deg", frame="icrs").galactic

jet_distance_east = jet_starts_e.separation(ss433).deg

jet_distance_west = jet_starts_w.separation(ss433).deg


def get_gaussian_filter(binsz, cont):
    """Get a Gaussian filter to convolve with that is equivalent
        to the PSF 68% containment in that energy range.

        Inputs:
        ------
        binsz : astropy.quantity
                size of the x bins in degrees or equivalent


        Returns:
        -------
        1D Gaussian filter to np.convolve with
        """

    cont_pix = (cont/binsz).to_value('')
    filter = Gaussian1DKernel(cont_pix)

    return filter


def prepare_model(east, west, x_data=None):
    """"From two advection instances (east and west) and the spatial
        bins used to plot the data, this method joins the model profiles
        along the jets, combining the west and eastern side, and normalizes
        the prediction to the bin size of the data.

        The prediction is also smoothed with a gaussian filter thar represents
        the 68% containment radius of the H.E.S.S. PSF weighted by an E^-2.3
        spectrum derived between each of the energy ranges.
        """

    # Get the profiles - these will be in physical (pc) size, starting at 0
    east_p_low = east.get_profile_energy_range(0.8*u.TeV, 2.5*u.TeV)
    east_p_mid = east.get_profile_energy_range(2.5*u.TeV, 10*u.TeV)
    east_p_high = east.get_profile_energy_range(10*u.TeV, 100*u.TeV)

    west_p_low = west.get_profile_energy_range(0.8*u.TeV, 2.5*u.TeV)
    west_p_mid = west.get_profile_energy_range(2.5*u.TeV, 10*u.TeV)
    west_p_high = west.get_profile_energy_range(10*u.TeV, 100*u.TeV)

    # Shift 0 to distance from SS 433
    east_x_low = east.pc_to_deg(east_p_low[1]) + jet_distance_east
    east_x_mid = east.pc_to_deg(east_p_mid[1]) + jet_distance_east
    east_x_high = east.pc_to_deg(east_p_high[1]) + jet_distance_east

    west_x_low = -(west.pc_to_deg(west_p_low[1]) + jet_distance_west)
    west_x_mid = -(west.pc_to_deg(west_p_mid[1]) + jet_distance_west)
    west_x_high = -(west.pc_to_deg(west_p_high[1]) + jet_distance_west)

    # Add zeroes to the region we added - the model doesn't predict anything there
    x_step_east = np.median(np.diff(east_x_low))
    zero_to_start_east = np.arange(0,east_x_low.min(),x_step_east)
    pad_zeros_east = np.zeros(len(zero_to_start_east))*east_p_low[0].unit

    east_x_low = np.append(zero_to_start_east, east_x_low)
    east_y_low = np.append(pad_zeros_east, east_p_low[0])
    east_x_mid = np.append(zero_to_start_east, east_x_mid)
    east_y_mid = np.append(pad_zeros_east, east_p_mid[0])
    east_x_high = np.append(zero_to_start_east, east_x_high)
    east_y_high = np.append(pad_zeros_east, east_p_high[0])

    x_step_west = np.median(np.diff(west_x_low))
    zero_to_start_west = np.arange(0,west_x_low.max(),x_step_west)
    pad_zeros_west = np.zeros(len(zero_to_start_west))*west_p_low[0].unit

    west_x_low = np.append(zero_to_start_west, west_x_low)
    west_y_low = np.append(pad_zeros_west, west_p_low[0])
    west_x_mid = np.append(zero_to_start_west, west_x_mid)
    west_y_mid = np.append(pad_zeros_west, west_p_mid[0])
    west_x_high = np.append(zero_to_start_west, west_x_high)
    west_y_high = np.append(pad_zeros_west, west_p_high[0])

    # Now the spatial bins were dermined dynamically by the code using 200
    # equispaced bins from the minimum to the maximum distance at which electrons
    # are advected. We will now unify both sides to the same axis. Note that
    # the prediction of the model is a histogram, i.e. a number of photons
    # per spatial bin. So this axis readjustment requires to correct for the
    # new bin size.

    # Size of existing bins (different on each side)
    binsz_east = np.median(np.diff(east_x_low))*u.deg
    binsz_west = np.abs(np.median(np.diff(west_x_low)))*u.deg

    # New axis that joins both
    bins_target = np.linspace(west_x_high.min(), east_x_high.max(), 500)
    binsz_target = np.median(np.diff(bins_target))*u.deg

    # Interpolate and correct for  bin size
    east_y_low /= binsz_east.to_value('deg')
    east_y_low = np.interp(bins_target, east_x_low, east_y_low, left=0, right=0)
    east_y_mid /= binsz_east.to_value('deg')
    east_y_mid = np.interp(bins_target, east_x_mid, east_y_mid, left=0, right=0)
    east_y_high /= binsz_east.to_value('deg')
    east_y_high = np.interp(bins_target, east_x_high, east_y_high, left=0, right=0)

    west_y_low /= binsz_west.to_value('deg')
    west_y_low = np.interp(bins_target, np.flip(west_x_low), np.flip(west_y_low), left=0, right=0)
    west_y_mid /= binsz_west.to_value('deg')
    west_y_mid = np.interp(bins_target, np.flip(west_x_mid), np.flip(west_y_mid), left=0, right=0)
    west_y_high /= binsz_west.to_value('deg')
    west_y_high = np.interp(bins_target, np.flip(west_x_high), np.flip(west_y_high), left=0, right=0)


    # Smooth with PSF
    containment_west = [0.05064, 0.04881, 0.067575]*u.deg
    containment_east = [0.05051, 0.04927, 0.064235]*u.deg

    west_g_low = get_gaussian_filter(binsz_target,containment_west[0])
    west_g_mid = get_gaussian_filter(binsz_target,containment_west[1])
    west_g_high = get_gaussian_filter(binsz_target,containment_west[2])

    east_g_low = get_gaussian_filter(binsz_target,containment_east[0])
    east_g_mid = get_gaussian_filter(binsz_target,containment_east[1])
    east_g_high = get_gaussian_filter(binsz_target,containment_east[2])


    # Get the bin size of the data we will be plotting with
    binsz_data = np.median(np.diff(x_data))
    f = binsz_data # this will be the new normalization

    # convolve with gaussian filter
    east_p_low_smooth = f*np.convolve(east_y_low, east_g_low, mode="same")
    east_p_mid_smooth = f*np.convolve(east_y_mid, east_g_mid, mode="same")
    east_p_high_smooth = f*np.convolve(east_y_high, east_g_high, mode="same")

    west_p_low_smooth = f*np.convolve(west_y_low, west_g_low, mode="same")
    west_p_mid_smooth = f*np.convolve(west_y_mid, west_g_mid, mode="same")
    west_p_high_smooth = f*np.convolve(west_y_high, west_g_high, mode="same")

    # Combine both sides
    x_low = bins_target
    y_low = east_p_low_smooth + west_p_low_smooth
    y_mid = east_p_mid_smooth + west_p_mid_smooth
    y_high = east_p_high_smooth + west_p_high_smooth


    return x_low, np.stack([y_low, y_mid, y_high])

