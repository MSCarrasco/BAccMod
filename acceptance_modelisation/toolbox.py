import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from astropy.coordinates.earth import OMEGA_EARTH, EarthLocation
from astropy.time import Time
from gammapy.data import Observations

def compute_rotation_speed_fov(time_evaluation: Time,
                               pointing_sky: SkyCoord,
                               observatory_earth_location: EarthLocation) -> u.Quantity:
    """
    Compute the rotation speed of the FOV for a given evaluation time.

    Parameters
    ----------
    time_evaluation : astropy.time.Time
        The time at which the rotation speed should be evaluated.
    pointing_sky : astropy.coordinates.SkyCoord
        The direction pointed in the sky.
    observatory_earth_location : astropy.coordinates.EarthLocation
        The position of the observatory.

    Returns
    -------
    rotation_speed : astropy.units.Quantity
        The rotation speed of the FOV at the given time and pointing direction.
    """
    pointing_altaz = pointing_sky.transform_to(AltAz(obstime=time_evaluation,
                                                     location=observatory_earth_location))
    omega_earth = OMEGA_EARTH*u.rad
    omega = omega_earth * np.cos(observatory_earth_location.lat) * np.cos(pointing_altaz.az) / np.cos(
        pointing_altaz.alt)
    return omega

def get_unique_wobble_pointings(observations: Observations):
    """
    Compute the angular separation between pointings and return a list 
    of detected wobbles with their associated similar pointings

    Parameters
    ----------
    observations : gammapy.data.observations.Observations
        The list of observations

    Returns
    -------
    unique_wobble_list : list
        A list of the wobbles detected and their associated similar pointings (angular separation < 0.4°)
    """
    # TODO Make the maximum angular distance an input parameter
    all_ra_observations = np.array([obs.get_pointing_icrs(obs.tmid).ra.to_value(u.deg) for obs in observations])
    all_dec_observations = np.array([obs.get_pointing_icrs(obs.tmid).dec.to_value(u.deg) for obs in observations])
    ra_observations = deepcopy(all_ra_observations)
    dec_observations = deepcopy(all_dec_observations)
    wobbles = np.empty(shape=len(all_ra_observations), dtype=np.object_)
    wobbles_dict = {}
    i=0
    mask_allremaining = np.ones(shape=len(all_ra_observations),dtype=bool)
    while len(ra_observations)>0:
        i=i+1
        keywobble='W'+str(i)
        mask = (angular_separation(ra_observations[0]*u.deg, dec_observations[0]*u.deg,
                                   ra_observations*u.deg, dec_observations*u.deg) < 0.4*u.deg)
        mask_2 = (angular_separation(np.mean(ra_observations[mask])*u.deg, np.mean(dec_observations[mask])*u.deg,
                                     all_ra_observations*u.deg, all_dec_observations*u.deg) < 0.4*u.deg)
        wobbles_dict[keywobble] = [np.mean(all_ra_observations[mask_2 & mask_allremaining]), np.mean(all_dec_observations[mask_2 & mask_allremaining])]
        wobbles[mask_2 & mask_allremaining] = keywobble
        mask_allremaining = mask_allremaining * ~mask_2
        ra_observations = all_ra_observations[mask_allremaining]
        dec_observations = all_dec_observations[mask_allremaining]

    print(f"{len(wobbles_dict)} wobbles were found: \n", wobbles_dict)
    return wobbles
