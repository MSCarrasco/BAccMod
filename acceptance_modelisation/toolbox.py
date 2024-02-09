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
    ra_observations = np.round([obs.get_pointing_icrs(obs.tmid).ra.to_value(u.deg) for obs in observations],1)
    dec_observations = np.round([obs.get_pointing_icrs(obs.tmid).dec.to_value(u.deg) for obs in observations],1)
    radec_observations = np.column_stack((ra_observations, dec_observations))
    radec_list = np.unique(radec_observations,axis=0)

    pointing_angsep_list = []
    for i in range(len(radec_list)): 
        pointing_angsep_list.append(np.round([angular_separation(radec_list[i][0]*u.deg, radec_list[i][1]*u.deg,
                                                                 ra*u.deg, dec*u.deg).to_value(u.deg) for ra,dec in radec_list],1))

    is_same_wobble_arr = np.vstack(pointing_angsep_list) <=0.2

    wobble_list = [radec_list[np.where(is_same_wobble)] for is_same_wobble in is_same_wobble_arr]
    unique_wobble_list = []
    seen = set()

    for arr in wobble_list:
        arr_tuple = tuple(map(tuple, arr))
        if arr_tuple not in seen:
            unique_wobble_list.append(arr)
            seen.add(arr_tuple)
    print(f"{len(unique_wobble_list)} wobbles were found: ")
    for i in range(len(unique_wobble_list)): print(f"W{i+1} associated pointings (ra,dec): {list(unique_wobble_list[i])}")
    return unique_wobble_list