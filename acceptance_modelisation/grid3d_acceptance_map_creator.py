from typing import List, Optional

import astropy.units as u
from gammapy.data import Observations
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis
from regions import SkyRegion

from .base_acceptance_map_creator import BaseAcceptanceMapCreator

from iminuit import Minuit
from .modeling import *

import matplotlib.pyplot as plt

FIT_FUNCTION = {'fit_gaussian': gaussian2d,
                'fit_ylin_gaussian': ylinear_1dgaussian,
                'fit_bilin_gaussian': bilinear_1dgaussian,
                'fit_ring_bi_gaussian': ring_bi_gaussian,
                'fit_bi_gaussian': bi_gaussian,
                'fit_radial_poly': radial_poly}


class Grid3DAcceptanceMapCreator(BaseAcceptanceMapCreator):

    def __init__(self,
                 energy_axis: MapAxis,
                 offset_axis: MapAxis,
                 oversample_map: int = 10,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 cos_zenith_binning_method: str = 'livetime',
                 min_observation_per_cos_zenith_bin: int = 15,
                 min_livetime_per_cos_zenith_bin: u.Quantity = 3000. * u.s,
                 initial_cos_zenith_binning: float = 0.01,
                 max_angular_separation: float = 0.4,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution_rotation_fov: u.Quantity = 0.1 * u.s,
                 method: str = 'stack',
                 fix_center: bool = True,
                 minuit_print_level: int = 0,
                 check_model: 'str' = 'nothing',
                 verbose: bool = False) -> None:
        """
        Create the class for calculating 3D grid acceptance model

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
        cos_zenith_binning_method : str, optional
            Method to compute the cos zenith binning: "observation","livetime"
            "observation" method use the minimum number of observation criteria
            "livetime" method use the minimum amount of livetime criteria
        min_observation_per_cos_zenith_bin : int, optional
            Minimum number of runs per zenith bins
        min_livetime_per_cos_zenith_bin : astropy.units.Quantity, optional
            Minimum livetime per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation : float, optional
            The maximum angular separation between identified wobbles, in degrees
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution_rotation_fov : astropy.unit.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV
        method : str, optional
            Decide if the acceptance is a direct event stacking or a fitted model
        fix_center: bool, optional
            Decide if models center should be fitted or not
        minuit_print_level: int, optional
            Define the verbosity of the call to minuit if fitting a model
        check_model: str, optional
            Define the level of verbosity of the model fitting. 'nothing' to display no information,
            'print' to display seeds and end parameters and total residual,
            'plot' to also show the counts, model and residual maps
        verbose : bool, optional
            If True, print the information related to the cos zenith binning
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Compute parameters for internal map
        self.offset_axis = offset_axis
        if not np.allclose(self.offset_axis.bin_width, self.offset_axis.bin_width[0]):
            raise Exception('Support only regular linear bin for offset axis')
        if not np.isclose(self.offset_axis.edges[0], 0. * u.deg):
            raise Exception('Offset axis need to start at 0')
        self.oversample_map = oversample_map
        spatial_resolution = np.min(
            np.abs(self.offset_axis.edges[1:] - self.offset_axis.edges[:-1])) / self.oversample_map
        max_offset = np.max(self.offset_axis.edges)

        self.method = method
        self.fix_center = fix_center
        self.minuit_print_level = minuit_print_level
        self.check_model = check_model

        # Initiate upper instance
        super().__init__(energy_axis, max_offset, spatial_resolution, exclude_regions,
                         cos_zenith_binning_method, min_observation_per_cos_zenith_bin, min_livetime_per_cos_zenith_bin,
                         initial_cos_zenith_binning, max_angular_separation, max_fraction_pixel_rotation_fov,
                         time_resolution_rotation_fov, verbose)

    def fit_background(self, count_map, exp_map_total, exp_map):
        fnc = FIT_FUNCTION[self.method]
        centers = self.offset_axis.center.to_value(u.deg)
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        seeds, bounds = fit_seed(count_map * exp_map_total / exp_map, centers, self.method)
        x, y = np.meshgrid(centers, centers)

        log_factorial_count_map = log_factorial(count_map)

        def f(*args):
            return -np.sum(
                log_poisson(count_map, fnc(x, y, *args) * exp_map / exp_map_total, log_factorial_count_map))

        if self.check_model != 'nothing':
            print("seeds : ", seeds)
        m = Minuit(f,
                   name=seeds.keys(),
                   *seeds.values())
        if self.fix_center and 'x_cm' in seeds.keys():
            m.fixed['x_cm'] = True
            m.fixed['y_cm'] = True
        for key, val in bounds.items():
            m.limits[key] = val
        m.print_level = self.minuit_print_level
        m.errordef = Minuit.LIKELIHOOD
        m.simplex().migrad()
        if self.check_model != 'nothing':
            residuals = count_map - fnc(x, y, **m.values.to_dict()) * exp_map / exp_map_total
            print("Fit results : ", m.values.to_dict())
            print(f"Average residuals : {np.sum(residuals)}, std = {np.std(residuals)}")

        if self.check_model == 'plot':
            fig, ax = plt.subplots(1, 3, sharey='all', figsize=(16, 4))
            fig.colorbar(ax[0].imshow(count_map))
            ax[0].set_title("Counts")
            fig.colorbar(ax[1].imshow(fnc(x, y, **m.values.to_dict())))
            ax[1].set_title("Model")
            fig.colorbar(ax[2].imshow(residuals))
            ax[2].set_title("Residual Counts-Model\n(Exposure corrected)")
            plt.show()
        return fnc(x, y, **m.values.to_dict())

    def create_acceptance_map(self, observations: Observations) -> Background3D:
        """
        Calculate a 3D grid acceptance map

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background3D
        """

        # Compute base data
        count_map_background, exp_map_background, exp_map_background_total, livetime = self._create_base_computation_map(
            observations)

        # Downsample map to bkg model resolution
        count_map_background_downsample = count_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_map_background_downsample = exp_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_map_background_total_downsample = exp_map_background_total.downsample(self.oversample_map,
                                                                                  preserve_counts=True)

        # Create axis for bkg model
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name='fov_lon')
        bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], extended_offset_axis_x.nbin, axis=1)
        extended_offset_axis_y = MapAxis.from_edges(extended_edges, name='fov_lat')
        bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], extended_offset_axis_y.nbin, axis=0)
        solid_angle = 4. * (np.sin(bin_width_x / 2.) * np.sin(bin_width_y / 2.)) * u.steradian

        # Compute acceptance_map
        if self.method == 'stack':
            corrected_counts = count_map_background_downsample.data * (exp_map_background_total_downsample.data /
                                                                       exp_map_background_downsample.data)
        else:
            corrected_counts = np.empty(count_map_background_downsample.data.shape)
            for e in range(count_map_background_downsample.data.shape[0]):
                if self.check_model != 'nothing':
                    print(f"Energy bin #{e}")
                corrected_counts[e] = self.fit_background(count_map_background_downsample.data[e].astype(int),
                                                          exp_map_background_total_downsample.data[e],
                                                          exp_map_background_downsample.data[e],
                                                          )
        data_background = corrected_counts / solid_angle[np.newaxis, :, :] / self.energy_axis.bin_width[:, np.newaxis,
                                                                             np.newaxis] / livetime

        acceptance_map = Background3D(axes=[self.energy_axis, extended_offset_axis_x, extended_offset_axis_y],
                                      data=data_background.to(u.Unit('s-1 MeV-1 sr-1')),
                                      fov_alignment=FoVAlignment.ALTAZ)

        return acceptance_map
