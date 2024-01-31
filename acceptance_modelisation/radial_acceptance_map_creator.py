from typing import List, Optional, Tuple

import astropy.units as u
from gammapy.data import Observation
from gammapy.datasets import MapDataset
from gammapy.maps import MapAxis, WcsNDMap, WcsGeom
from regions import SkyRegion

from .base_radial_acceptance_map_creator import BaseRadialAcceptanceMapCreator


class RadialAcceptanceMapCreator(BaseRadialAcceptanceMapCreator):

    def __init__(self,
                 energy_axis: MapAxis,
                 offset_axis: MapAxis,
                 oversample_map: int = 10,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 min_observation_per_cos_zenith_bin: int = 3,
                 initial_cos_zenith_binning: float = 0.01,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution_rotation_fov: u.Quantity = 0.1 * u.s,
                 verbose: bool = False) -> None:
        """
        Create the class for calculating radial acceptance model
        This class should be use when strict 2D model is good enough

        Parameters
        ----------
        energy_axis : MapAxis
            The energy axis for the acceptance model
        offset_axis : MapAxis
            The offset axis for the acceptance model
        oversample_map : int, optional
            Oversample in number of pixel of the spatial axis used for the calculation
        exclude_regions : list of regions.SkyRegion, optional
            Region with known or putative gamma-ray emission, will be excluded of the calculation of the acceptance map
        min_observation_per_cos_zenith_bin : int, optional
            Minimum number of runs per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution_rotation_fov : astropy.unit.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV
        verbose : bool, optional
            If True, print the informations related to the cos zenith binning
        """

        # Initiate upper instance
        super().__init__(energy_axis=energy_axis,
                         offset_axis=offset_axis,
                         oversample_map=oversample_map,
                         exclude_regions=exclude_regions,
                         min_observation_per_cos_zenith_bin=min_observation_per_cos_zenith_bin,
                         initial_cos_zenith_binning=initial_cos_zenith_binning,
                         max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
                         time_resolution_rotation_fov=time_resolution_rotation_fov,
                         verbose=verbose)

    def _create_base_computation_map(self, observations: Observation) -> Tuple[WcsNDMap, WcsNDMap, WcsNDMap, u.Unit]:
        """
        From a list of observations return a stacked finely binned counts and exposure map in camera frame to compute a model

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The list of observations

        Returns
        -------
        count_map_background : gammapy.map.WcsNDMap
            The count map
        exp_map_background : gammapy.map.WcsNDMap
            The exposure map corrected for exclusion regions
        exp_map_background_total : gammapy.map.WcsNDMap
            The exposure map without correction for exclusion regions
        livetime : astropy.unit.Unit
            The total exposure time for the model
        """
        count_map_background = WcsNDMap(geom=self.geom)
        exp_map_background = WcsNDMap(geom=self.geom, unit=u.s)
        exp_map_background_total = WcsNDMap(geom=self.geom, unit=u.s)
        livetime = 0. * u.s

        for obs in observations:
            geom = WcsGeom.create(skydir=obs.pointing.fixed_icrs, npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size, frame="icrs", axes=[self.energy_axis])
            count_map_obs, exclusion_mask = self._create_map(obs, geom, self.exclude_regions, add_bkg=False)

            exp_map_obs = MapDataset.create(geom=count_map_obs.geoms['geom'])
            exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms['geom'])
            exp_map_obs.counts.data = obs.observation_live_time_duration.value
            exp_map_obs_total.counts.data = obs.observation_live_time_duration.value

            for i in range(count_map_obs.counts.data.shape[0]):
                count_map_obs.counts.data[i, :, :] = count_map_obs.counts.data[i, :, :] * exclusion_mask
                exp_map_obs.counts.data[i, :, :] = exp_map_obs.counts.data[i, :, :] * exclusion_mask

            count_map_background.data += count_map_obs.counts.data
            exp_map_background.data += exp_map_obs.counts.data
            exp_map_background_total.data += exp_map_obs_total.counts.data
            livetime += obs.observation_live_time_duration

        return count_map_background, exp_map_background, exp_map_background_total, livetime

