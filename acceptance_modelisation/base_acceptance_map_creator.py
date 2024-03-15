import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
from gammapy.data import Observations, Observation
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D, Background3D
from gammapy.irf.background import BackgroundIRF
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom, Map, MapAxis, RegionGeom
from regions import CircleSkyRegion, SkyRegion
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .toolbox import compute_rotation_speed_fov, get_unique_wobble_pointings


class BaseAcceptanceMapCreator(ABC):

    def __init__(self,
                 energy_axis: MapAxis,
                 max_offset: u.Quantity,
                 spatial_resolution: u.Quantity,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 cos_zenith_binning_method='livetime',
                 min_observation_per_cos_zenith_bin: int = 3,
                 min_livetime_per_cos_zenith_bin: u.Quantity = 3000. * u.s,
                 initial_cos_zenith_binning: float = 0.01,
                 max_angular_separation: float = 0.4,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution_rotation_fov: u.Quantity = 0.1 * u.s,
                 verbose: bool = False) -> None:
        """
        Create the class for calculating radial acceptance model.

        Parameters
        ----------
        energy_axis : gammapy.maps.geom.MapAxis
            The energy axis for the acceptance model
        max_offset : astropy.units.Quantity
            The offset corresponding to the edge of the model
        spatial_resolution : astropy.units.Quantity
            The spatial resolution
        exclude_regions : list of regions.SkyRegion, optional
            Regions with known or putative gamma-ray emission, will be excluded from the calculation of the acceptance map
        cos_zenith_binning_method : str, optional
            Method to compute the cos zenith binning: "observation","livetime"
            "observation" method use the minimum number of observation criteria
            "livetime" method use the minimum amount of livetime criteria
        min_observation_per_cos_zenith_bin : int, optional
            Minimum number of observations per zenith bins
        min_livetime_per_cos_zenith_bin : astropy.units.Quantity, optional
            Minimum livetime per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation : float, optional
            The maximum angular separation between identified wobbles, in degrees
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution_rotation_fov : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV
        verbose : bool, optional
            If True, print the informations related to the cos zenith binning
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Store base parameter
        self.energy_axis = energy_axis
        self.max_offset = max_offset
        self.exclude_regions = exclude_regions
        self.cos_zenith_binning_method = cos_zenith_binning_method
        self.min_observation_per_cos_zenith_bin = min_observation_per_cos_zenith_bin
        self.min_livetime_per_cos_zenith_bin = min_livetime_per_cos_zenith_bin
        self.initial_cos_zenith_binning = initial_cos_zenith_binning
        self.max_angular_separation = max_angular_separation
        self.verbose = verbose

        # Calculate map parameter
        self.n_bins_map = 2 * int(np.rint((self.max_offset / spatial_resolution).to(u.dimensionless_unscaled)))
        self.spatial_bin_size = self.max_offset / (self.n_bins_map / 2)
        self.center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')
        self.geom = WcsGeom.create(skydir=self.center_map, npix=(self.n_bins_map, self.n_bins_map),
                                   binsz=self.spatial_bin_size, frame="icrs", axes=[self.energy_axis])
        logging.info(
            'Computation will be made with a bin size of {:.3f} arcmin'.format(
                self.spatial_bin_size.to_value(u.arcmin)))

        # Store rotation computation parameters
        self.max_fraction_pixel_rotation_fov = max_fraction_pixel_rotation_fov
        self.time_resolution_rotation_fov = time_resolution_rotation_fov

    @staticmethod
    def _transform_obs_to_camera_frame(obs: Observation) -> Observation:
        """
        Transform events, pointing and exclusion regions of an obs from a sky frame to camera frame

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation to transform

        Returns
        -------
        obs_camera_frame : gammapy.data.observations.Observation
            The observation transformed for reference in camera frame
        """

        # Transform to altaz frame
        camera_frame_events = obs.events.copy()
        if len(obs.events.time) > 0:
            altaz_frame = AltAz(obstime=obs.events.time,
                                location=obs.observatory_earth_location)
            events_altaz = obs.events.radec.transform_to(altaz_frame)
            pointing_altaz = obs.get_pointing_icrs(obs.events.time).transform_to(altaz_frame)

            # Rotation to transform to camera frame
            camera_frame = SkyOffsetFrame(origin=AltAz(alt=pointing_altaz.alt,
                                                       az=pointing_altaz.az,
                                                       obstime=obs.events.time,
                                                       location=obs.observatory_earth_location),
                                          rotation=[0., ] * len(obs.events.time) * u.deg)
            events_camera_frame = events_altaz.transform_to(camera_frame)

            # Formatting data for the output
            camera_frame_events.table['RA'] = events_camera_frame.lon
            camera_frame_events.table['DEC'] = events_camera_frame.lat
        camera_frame_obs_info = copy.deepcopy(obs.obs_info)
        camera_frame_obs_info['RA_PNT'] = 0.
        camera_frame_obs_info['DEC_PNT'] = 0.
        obs_camera_frame = Observation(obs_id=obs.obs_id,
                                       obs_info=camera_frame_obs_info,
                                       events=camera_frame_events,
                                       gti=obs.gti,
                                       aeff=obs.aeff)
        obs_camera_frame._location = obs.observatory_earth_location

        return obs_camera_frame

    def _transform_exclusion_region_to_camera_frame(self, pointing_altaz: AltAz) -> List[SkyRegion]:
        """
        Transform the list of exclusion regions in sky frame into a list in camera frame.

        Parameters
        ----------
        pointing_altaz : astropy.coordinates.AltAz
            The pointing position of the telescope.

        Returns
        -------
        exclusion_region_camera_frame : list of regions.SkyRegion
            The list of exclusion regions in camera frame.

        Raises
        ------
        Exception
            If the region type is not supported.
        """

        camera_frame = SkyOffsetFrame(origin=pointing_altaz,
                                      rotation=[0., ] * u.deg)
        exclude_region_camera_frame = []
        for region in self.exclude_regions:
            if isinstance(region, CircleSkyRegion):
                center_coordinate = region.center
                center_coordinate_altaz = center_coordinate.transform_to(pointing_altaz)
                center_coordinate_camera_frame = center_coordinate_altaz.transform_to(camera_frame)
                center_coordinate_camera_frame_arb = SkyCoord(ra=center_coordinate_camera_frame.lon[0],
                                                              dec=center_coordinate_camera_frame.lat[0])
                exclude_region_camera_frame.append(CircleSkyRegion(center=center_coordinate_camera_frame_arb,
                                                                   radius=region.radius))
            else:
                raise Exception(f'{type(region)} region type not supported')

        return exclude_region_camera_frame

    def _create_map(self,
                    obs: Observation,
                    geom: WcsGeom,
                    exclude_regions: List[SkyRegion],
                    add_bkg: bool = False
                    ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create a map and the associated exclusion mask based on the given geometry and exclusion region.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        geom : gammapy.maps.WcsGeom
            The geometry for the maps.
        exclude_regions : list of regions.SkyRegion
            The list of exclusion regions.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        maker = MapDatasetMaker(selection=["counts"])
        if add_bkg:
            maker = MapDatasetMaker(selection=["counts", "background"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=self.max_offset)

        geom_image = geom.to_image()
        exclusion_mask = ~geom_image.region_mask(exclude_regions) if len(exclude_regions) > 0 else ~Map.from_geom(
            geom_image)

        map_obs = maker.run(MapDataset.create(geom=geom), obs)
        map_obs = maker_safe_mask.run(map_obs, obs)

        return map_obs, exclusion_mask

    def _create_sky_map(self,
                        obs: Observation,
                        add_bkg: bool = False
                        ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create the sky map and the associated exclusion mask based on the observation and the exclusion regions.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        geom_obs = WcsGeom.create(skydir=obs.get_pointing_icrs(obs.tmid),
                                  npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size,
                                  frame="icrs",
                                  axes=[self.energy_axis])
        map_obs, exclusion_mask = self._create_map(obs, geom_obs, self.exclude_regions, add_bkg=add_bkg)

        return map_obs, exclusion_mask

    def _compute_time_intervals_based_on_fov_rotation(self, obs: Observation) -> Time:
        """
        Calculate time intervals based on the rotation of the Field of View (FoV).

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to calculate time intervals.

        Returns
        -------
        time_intervals : astropy.time.Time
            The time intervals for cutting the observation into time bins.
        """

        # Determine time interval for cutting the obs as function of the rotation of the Fov
        n_bin = max(2, int(np.rint(
            ((obs.tstop - obs.tstart) / self.time_resolution_rotation_fov).to_value(u.dimensionless_unscaled))))
        time_axis = np.linspace(obs.tstart, obs.tstop, num=n_bin)
        rotation_speed_fov = compute_rotation_speed_fov(time_axis, obs.get_pointing_icrs(obs.tmid),
                                                        obs.observatory_earth_location)
        rotation_fov = cumulative_trapezoid(x=time_axis.unix_tai,
                                            y=rotation_speed_fov.to_value(u.rad / u.s),
                                            initial=0.) * u.rad
        distance_rotation_fov = rotation_fov.to_value(u.rad) * np.pi * self.max_offset
        node_obs = distance_rotation_fov // (self.spatial_bin_size * self.max_fraction_pixel_rotation_fov)
        change_node = node_obs[2:] != node_obs[1:-1]
        time_interval = Time([obs.tstart, ] + [time_axis[1:-1][change_node], ] + [obs.tstop, ])

        return time_interval

    def _create_base_computation_map(self, observations: Observations) -> Tuple[WcsNDMap, WcsNDMap, WcsNDMap, u.Unit]:
        """
        From a list of observations return a stacked finely binned counts and exposure map in camera frame to compute a
        model

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The list of observations

        Returns
        -------
        count_map_background : gammapy.map.WcsNDMap
            The count map corrected with empty bin filling method
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
        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            for obs in observations:
                # Filter events in exclusion regions
                geom = RegionGeom.from_regions(self.exclude_regions)
                mask = geom.contains(obs.events.radec)
                obs._events = obs.events.select_row_subset(~mask)

                time_interval = self._compute_time_intervals_based_on_fov_rotation(obs)
                camera_frame_obs = self._transform_obs_to_camera_frame(obs)
                count_map_obs, _ = self._create_map(camera_frame_obs, self.geom, [], add_bkg=False)

                exp_map_obs = MapDataset.create(geom=count_map_obs.geoms['geom'])
                exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms['geom'])
                exp_map_obs.counts.data = camera_frame_obs.observation_live_time_duration.value
                exp_map_obs_total.counts.data = camera_frame_obs.observation_live_time_duration.value
                exclusion_mask = np.zeros(count_map_obs.counts.data.shape[1:])
                for i in range(len(time_interval) - 1):
                    # Compute the exclusion region in camera frame for the average time
                    average_alt_az_frame = AltAz(obstime=obs.tmid,
                                                 location=obs.observatory_earth_location)
                    average_alt_az_pointing = obs.get_pointing_icrs(obs.tmid).transform_to(average_alt_az_frame)
                    exclusion_region_camera_frame = self._transform_exclusion_region_to_camera_frame(
                        average_alt_az_pointing)
                    geom_image = self.geom.to_image()

                    exclusion_mask_t = ~geom_image.region_mask(exclusion_region_camera_frame) if len(
                        exclusion_region_camera_frame) > 0 else ~Map.from_geom(geom_image)
                    exclusion_mask += exclusion_mask_t * (time_interval[i + 1] - time_interval[i]).value
                exclusion_mask *= 1 / (time_interval[-1] - time_interval[0]).value

                for j in range(count_map_obs.counts.data.shape[0]):
                    exp_map_obs.counts.data[j, :, :] = exp_map_obs.counts.data[j, :, :] * exclusion_mask

                count_map_background.data += count_map_obs.counts.data
                exp_map_background.data += exp_map_obs.counts.data
                exp_map_background_total.data += exp_map_obs_total.counts.data
                livetime += camera_frame_obs.observation_live_time_duration

        return count_map_background, exp_map_background, exp_map_background_total, livetime

    @abstractmethod
    def create_acceptance_map(self, observations: Observations) -> BackgroundIRF:
        """
        Abstract method to calculate an acceptance map from a list of observations.

        Subclasses must implement this method to provide the specific algorithm for calculating the acceptance map.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to create the acceptance map.

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            The acceptance map calculated using the specific algorithm implemented by the subclass.
        """
        pass

    def _normalised_model_per_run(self,
                                  observations: Observations,
                                  acceptance_map: dict[Any, BackgroundIRF]) -> dict[Any, BackgroundIRF]:
        """
        Normalised the acceptance model associated to each run to the events associated with the run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        acceptance_map :dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
            This is the models that will be normalised
        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """

        normalised_acceptance_map = {}
        # Fit norm of the model to the observations
        for obs in observations:
            id_observation = obs.obs_id

            # replace the background model
            modified_observation = copy.deepcopy(obs)
            modified_observation.bkg = acceptance_map[id_observation]

            # Fit the background model
            logging.info('Fit to model to run ' + str(id_observation))
            map_obs, exclusion_mask = self._create_sky_map(modified_observation, add_bkg=True)
            maker_FoV_background = FoVBackgroundMaker(method='fit', exclusion_mask=exclusion_mask)
            map_obs = maker_FoV_background.run(map_obs)

            # Extract the normalisation
            parameters = map_obs.models.to_parameters_table()
            norm_background = parameters[parameters['name'] == 'norm']['value'][0]

            if norm_background < 0.:
                logging.error(
                    'Invalid normalisation value for run ' + str(id_observation) + ' : ' + str(norm_background))
            elif norm_background > 1.5 or norm_background < 0.5:
                logging.warning('High correction of the background normalisation normalisation for run ' + str(
                    id_observation) + ' : ' + str(norm_background))

            # Apply normalisation to the background model
            normalised_acceptance_map[id_observation] = copy.deepcopy(acceptance_map[id_observation])
            normalised_acceptance_map[id_observation].data = normalised_acceptance_map[
                                                                 id_observation].data * norm_background

        return normalised_acceptance_map

    def create_acceptance_map_cos_zenith_binned(self,
                                                observations: Observations
                                                ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map using cos zenith binning and interpolation

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key

        """

        cos_zenith_bin = np.sort(np.arange(1.0, 0. - self.initial_cos_zenith_binning, -self.initial_cos_zenith_binning))
        cos_zenith_observations = np.array([np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in observations])
        livetime_observations = np.array([obs.observation_live_time_duration.to_value(u.s) for obs in observations])

        wobble_observations = np.array(get_unique_wobble_pointings(observations, self.max_angular_separation))

        if self.cos_zenith_binning_method == "livetime":
            cut_variable_weights = livetime_observations
            min_cut_per_cos_zenith_bin = self.min_livetime_per_cos_zenith_bin.to_value(u.s)
        elif self.cos_zenith_binning_method == "observation":
            cut_variable_weights = np.ones(len(cos_zenith_observations), dtype=int)
            min_cut_per_cos_zenith_bin = self.min_observation_per_cos_zenith_bin
        else:
            raise ValueError(f"No {self.cos_zenith_binning_method} method available for the cos zenith binning")

        cut_variable_per_bin = np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=cut_variable_weights)[
            0]

        i = 0
        while i < len(cut_variable_per_bin):
            in_coszd_bin = (cos_zenith_observations >= cos_zenith_bin[i]) & (
                    cos_zenith_observations < cos_zenith_bin[i + 1])
            wobble_in_bin = wobble_observations[in_coszd_bin]
            obs_livetime_in_bin = cut_variable_weights[in_coszd_bin]
            wobble_livetime_in_bin = np.array(
                [obs_livetime_in_bin[wobble_in_bin == wobble].sum() for wobble in np.unique(wobble_in_bin)])
            at_least_2_wobble = len(np.unique(wobble_in_bin)) > 1
            if at_least_2_wobble:
                livetime_min_for_each_wobble = np.all(
                    wobble_livetime_in_bin >= min_cut_per_cos_zenith_bin / len(np.unique(wobble_in_bin)))
            if not at_least_2_wobble and (i + 1) < len(cut_variable_per_bin):
                cut_variable_per_bin[i] += cut_variable_per_bin[i + 1]
                cut_variable_per_bin = np.delete(cut_variable_per_bin, i + 1)
                cos_zenith_bin = np.delete(cos_zenith_bin, i + 1)
            elif not livetime_min_for_each_wobble and (i + 1) < len(cut_variable_per_bin):
                cut_variable_per_bin[i] += cut_variable_per_bin[i + 1]
                cut_variable_per_bin = np.delete(cut_variable_per_bin, i + 1)
                cos_zenith_bin = np.delete(cos_zenith_bin, i + 1)
            elif cut_variable_per_bin[i] < min_cut_per_cos_zenith_bin and (i + 1) < len(cut_variable_per_bin):
                cut_variable_per_bin[i] += cut_variable_per_bin[i + 1]
                cut_variable_per_bin = np.delete(cut_variable_per_bin, i + 1)
                cos_zenith_bin = np.delete(cos_zenith_bin, i + 1)
            elif cut_variable_per_bin[i] < min_cut_per_cos_zenith_bin and (i + 1) == len(
                    cut_variable_per_bin) and i > 0:
                cut_variable_per_bin[i - 1] += cut_variable_per_bin[i]
                cut_variable_per_bin = np.delete(cut_variable_per_bin, i)
                cos_zenith_bin = np.delete(cos_zenith_bin, i)
                i -= 1
            else:
                i += 1

        binned_observations = []
        for i in range((len(cos_zenith_bin) - 1)):
            binned_observations.append(Observations())
        for obs in observations:
            binned_observations[np.digitize(np.cos(obs.get_pointing_altaz(obs.tmid).zen), cos_zenith_bin) - 1].append(
                obs)

        binned_model = [self.create_acceptance_map(binned_obs) for binned_obs in binned_observations]
        bin_center = []
        for i in range(len(binned_observations)):
            livetime_per_obs = np.array(
                [obs.observation_live_time_duration.to_value(u.s) for obs in binned_observations[i]])
            weighted_cos_zenith_bin_per_obs = livetime_per_obs * np.array(
                [np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in binned_observations[i]])
            bin_center.append(np.sum(weighted_cos_zenith_bin_per_obs) / np.sum(livetime_per_obs))
        if self.verbose:
            print("cos zenith bins: ", list(np.round(cos_zenith_bin, 2)))
            print("cos zenith bin centers: ", list(np.round(bin_center, 2)))
            print(f"{self.cos_zenith_binning_method} per bin: ", list(cut_variable_per_bin.astype(int)))
            if self.cos_zenith_binning_method == "livetime": print(f"observation per bin: ",
                                                                   list(np.histogram(cos_zenith_observations,
                                                                                     bins=cos_zenith_bin)[0]))
            wobble_observations_bool_arr = [(np.array(wobble_observations.tolist()) == wobble) for wobble in
                                            np.unique(np.array(wobble_observations))]
            livetime_observations_and_wobble = [np.array(livetime_observations) * wobble_bool for wobble_bool in
                                                wobble_observations_bool_arr]
            for i, wobble in enumerate(np.unique(np.array(wobble_observations))):
                print(
                    f"{wobble} observation per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=1 * wobble_observations_bool_arr[i])[0])}")
                print(
                    f"{wobble} livetime per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=livetime_observations_and_wobble[i])[0].astype(int))}")

        acceptance_map = {}
        if len(binned_model) <= 1:
            logging.warning('Only one zenith bin, zenith interpolation deactivated')
            for obs in observations:
                acceptance_map[obs.obs_id] = binned_model[0]
        else:
            data_cube = np.zeros(tuple([len(binned_model), ] + list(binned_model[0].data.shape))) * binned_model[0].unit
            for i in range(len(binned_model)):
                data_cube[i] = binned_model[i].data * binned_model[i].unit
            interp_func = interp1d(x=np.array(bin_center),
                                   y=np.log10(data_cube.value + np.finfo(np.float64).tiny),
                                   axis=0,
                                   fill_value='extrapolate')
            for obs in observations:
                data_obs = (10. ** interp_func(np.cos(obs.get_pointing_altaz(obs.tmid).zen)))
                data_obs[data_obs < 100 * np.finfo(np.float64).tiny] = 0.
                if type(binned_model[0]) is Background2D:
                    acceptance_map[obs.obs_id] = Background2D(axes=binned_model[0].axes,
                                                              data=data_obs * data_cube.unit)
                elif type(binned_model[0]) is Background3D:
                    acceptance_map[obs.obs_id] = Background3D(axes=binned_model[0].axes,
                                                              data=data_obs * data_cube.unit)
                else:
                    raise Exception('Unknown background format')

        return acceptance_map

    def create_acceptance_map_per_observation(self,
                                              observations: Observations,
                                              zenith_bin: bool = True,
                                              runwise_normalisation: bool = True,
                                              ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map with the norm adjusted for each run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        zenith_bin : bool, optional
            If true the acceptance maps will be generated using zenith binning and interpolation
        runwise_normalisation : bool, optional
            If true the acceptance maps will be normalised runwise to the observations

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """

        acceptance_map = {}
        if zenith_bin:
            acceptance_map = self.create_acceptance_map_cos_zenith_binned(observations)
        else:
            unique_base_acceptance_map = self.create_acceptance_map(observations)
            for obs in observations:
                acceptance_map[obs.obs_id] = unique_base_acceptance_map

        if runwise_normalisation:
            acceptance_map = self._normalised_model_per_run(observations, acceptance_map)

        return acceptance_map
