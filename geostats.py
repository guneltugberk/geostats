import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize
import math


class Geostatistics:
    def __init__(self, data, property, x_coord, y_coord, step_property=5, step_coords=100, unit_system='Imperial'):
        """
        Initialize the Geostatistics class with a dataset.

        Parameters:
            data (DataFrame): The input dataset.
            property (str): The input parameter for analysis.
            x_coord (str): X coordinate name.
            y_coord (str): Y coordinate name.
            step_property (int): Step size for property analysis (default=5).
            step_coords (int): Step size for coordinate analysis (default=100).
            unit_system (str): Unit system (default='Imperial').
        """
        
        self.data = data
        self.property = property
        self.step_property = step_property
        self.step_coords = step_coords
        self.unit_system = unit_system
        self.init_cloud = None
        self.init_exp = None

        # Coordinates
        self.X, self.Y = x_coord, y_coord

    def base_map(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

        sns.set_style('whitegrid')

        sns.scatterplot(data=self.data, x=self.X, y=self.property, hue=self.property, s=100, ax=ax[0])
        sns.scatterplot(data=self.data, x=self.Y, y=self.property, hue=self.property, s=100, ax=ax[1])

        lower_limit_y = math.floor(self.data[self.property].min() / self.step_property) * self.step_property
        upper_limit_y = math.ceil(self.data[self.property].max() / self.step_property) * self.step_property

        lower_limit_x0 = math.floor(self.data[self.X].min() / self.step_coords) * self.step_coords
        upper_limit_x0 = math.ceil(self.data[self.X].max() / self.step_coords) * self.step_coords

        lower_limit_x1 = math.floor(self.data[self.Y].min() / self.step_coords) * self.step_coords
        upper_limit_x1 = math.ceil(self.data[self.Y].max() / self.step_coords) * self.step_coords

        ax[0].set_xlim(lower_limit_x0, upper_limit_x0)
        ax[1].set_xlim(lower_limit_x1, upper_limit_x1)

        ax[0].set_ylim(lower_limit_y, upper_limit_y)
        ax[1].set_ylim(lower_limit_y, upper_limit_y)

        plt.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def calculate_variability(self, bin_col, bins):
        """
        Calculate mean, variance, and standard error for binned data.

        Parameters:
            bin_col (str): Column name for binning.
            bins (int): Number of bins.

        Returns:
            tuple: Midpoints, mean, variance, and standard error for each bin.
        """

        self.data[f'{bin_col}_bin'] = pd.cut(self.data[bin_col], bins=bins)

        # Group by bins
        grouped = self.data.groupby(f'{bin_col}_bin')[self.property]

        # Calculate statistics
        mean = grouped.mean()
        var = grouped.var()
        se = grouped.std() / grouped.size().apply(lambda n: n**0.5)

        # Extract bin midpoints
        midpoints = mean.index.map(lambda x: x.mid)

        return midpoints, mean, var, se

    def plot_variability(self, ax, midpoints, mean, se, title, xlabel, coord, include_observation=False):
        """
        Plot mean and standard error for binned data.

        Parameters:
            ax (AxesSubplot): Matplotlib subplot to plot on.
            midpoints (array): Midpoints of bins.
            mean (array): Mean values for each bin.
            se (array): Standard error for each bin.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.

        Returns:
            None
        """

        if include_observation:
            sns.lineplot(x=self.data[coord], y=self.data[self.property], ax=ax, lw=2, label=f'Observed {self.property}')
            
        sns.lineplot(x=midpoints, y=mean.values, ax=ax, lw=2, label=f'Moving Average {self.property}', linestyle='--')

        ax.fill_between(
            midpoints,
            mean - se,
            mean + se,
            color='blue',
            alpha=0.2,
            label='Standard Error'
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.property)
        ax.legend()

    def plot_property_variability(self, bins, include_observation=False):
        """
        Plot variability along latitude and longitude with mean and standard error bands.

        Parameters:
            bins (int): Number of bins.
            include_observation (bool): Whether to plot observation (default=False).

        Returns:
            None
        """

        fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
        sns.set_style('whitegrid')

        # Calculate statistics for latitude and longitude
        lat_midpoints, lat_mean, _, lat_se = self.calculate_variability(self.Y, bins)
        lon_midpoints, lon_mean, _, lon_se = self.calculate_variability(self.X, bins)

        if self.unit_system == 'Imperial':
            self.plot_variability(ax[0], lat_midpoints, lat_mean, lat_se, f'Variability of {self.property} on X', 'X (Midpoints), ft', self.Y, include_observation)
            self.plot_variability(ax[1], lon_midpoints, lon_mean, lon_se, f'Variability of {self.property} on Y', 'Y (Midpoints), ft', self.X, include_observation)

        elif self.unit_system == 'SI':
            self.plot_variability(ax[0], lat_midpoints, lat_mean, lat_se, f'Variability of {self.property} on X', 'X (Midpoints), m', self.Y, include_observation)
            self.plot_variability(ax[1], lon_midpoints, lon_mean, lon_se, f'Variability of {self.property} on Y', 'Y (Midpoints), m', self.X, include_observation)
            
        else:
            raise ValueError("Please provide a proper unit system!")
        
        lower_limit_y = math.floor(self.data[self.property].min() / self.step_property) * self.step_property
        upper_limit_y = math.ceil(self.data[self.property].max() / self.step_property) * self.step_property

        lower_limit_x0 = math.floor(min(lat_midpoints) / self.step_coords) * self.step_coords
        upper_limit_x0 = math.ceil(max(lat_midpoints) / self.step_coords) * self.step_coords

        lower_limit_x1 = math.floor(min(lon_midpoints) / self.step_coords) * self.step_coords
        upper_limit_x1 = math.ceil(max(lon_midpoints) / self.step_coords) * self.step_coords

        ax[0].set_ylim(lower_limit_y, upper_limit_y)
        ax[1].set_ylim(lower_limit_y, upper_limit_y)

        ax[0].set_xlim(lower_limit_x0, upper_limit_x0)
        ax[1].set_xlim(lower_limit_x1, upper_limit_x1)

        ax[0].grid(True, alpha=0.5)
        ax[1].grid(True, alpha=0.5)
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14

        plt.tight_layout()
        plt.show()


    def variogram_cloud(self, plot=False):
        """
        Constructs a variogram cloud based on input data.

        Parameters:
            plot (bool, optional): If True, plots the variogram cloud.

        Returns:
            int: Number of unique pairs used in the variogram cloud.
            np.array: Pairwise distances.
            np.array: Semi-variances for the pairs.
        """

        values = self.data[self.property].values 

        # Calculate squared differences between all pairs of values with the 0.5 factor (Semi-variance)
        pairwise_diff = 0.5 * (np.subtract.outer(values, values)**2)

        # Ensure the coordinates are in NumPy array format
        coordinates = self.data[[self.X, self.Y]].values  

        # Calculate pairwise distances
        pairwise_dist = pairwise_distances(coordinates)

        # Extract the upper triangular part (excluding diagonal) to get unique pairs
        self.distances = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
        self.squared_differences = pairwise_diff[np.triu_indices_from(pairwise_diff, k=1)]

        experimental_sill = self.data[self.property].var()

        if plot:
            plt.figure(figsize=(16, 9))
            plt.scatter(self.distances, self.squared_differences, alpha=0.6, edgecolor='k', label="Experimental Pairs")
            plt.axhline(y=experimental_sill, linestyle='--', color='red', label=f'Experimental Variance={round(experimental_sill, 3)}')

            plt.ylabel(r'$\frac{1}{2} \cdot (z(x_i) - z(x_j))^2$')

            if self.unit_system == 'Imperial':
                plt.xlabel('Distance (h), ft')

            elif self.unit_system == 'SI':
                plt.xlabel('Distance (h), m')
            
            else:
                raise ValueError("Please provide a proper unit system!")
            
            plt.grid(True, alpha=0.5)

            upper_limit_x = math.ceil(max(self.distances) / self.step_coords) * self.step_coords
            plt.xlim(0, upper_limit_x)

            plt.legend()
            plt.show()

        # Return number of pairs and data for further analysis if needed
        num_pairs = len(self.distances)

        self.init_cloud = True # Initilize the cloud variogram

        return num_pairs, self.distances, self.squared_differences
    

    def omnidirectional_variogram(self, lag_distance, max_distance=None, lag_tolerance=None, plot=False):
        """
        Calculate the experimental variogram with user-defined inputs.

        Parameters:
            lag_distance (float): User-defined lag distance for binning.
            max_distance (float, optional): Maximum distance to consider. Defaults to the maximum pairwise distance.
            lag_tolerance (float, optional): Tolerance around each lag. Defaults to half of lag_distance.
            plot (bool, optional): If True, plots the experimental variogram.

        Returns:
            tuple: Lag centers, experimental variogram values, and number of pairs.
        """
        if self.init_cloud is None:
            raise ValueError("Variogram cloud must be computed before constructing the experimental variogram.")

        if lag_distance <= 0:
            raise ValueError("`lag_distance` must be positive.")

        if max_distance is None:
            max_distance = np.max(self.distances)

        if lag_tolerance is None:
            lag_tolerance = lag_distance / 2

        # Initialize lists to store results
        lag_centers = []
        gamma_values = []
        num_pairs = []

        # Define lag bins
        current_lag = lag_distance  # Start with the first lag
        while current_lag <= max_distance:
            # Define the bounds for the current lag bin
            lower_bound = current_lag - lag_tolerance
            upper_bound = current_lag + lag_tolerance

            # Find distances within the current lag bin
            in_bin = (self.distances >= lower_bound) & (self.distances < upper_bound)

            # Calculate the average semi-variance (gamma) for this bin
            if np.any(in_bin):  # Check if there are pairs in this bin
                avg_gamma = np.mean(self.squared_differences[in_bin])
                gamma_values.append(avg_gamma)
                lag_centers.append(current_lag)
                num_pairs.append(np.sum(in_bin))
            else:  # No pairs in this bin
                gamma_values.append(0)
                lag_centers.append(current_lag)
                num_pairs.append(0)

            # Move to the next lag
            current_lag += lag_distance

        # Convert lists to arrays
        lag_centers = np.array(lag_centers)
        gamma_values = np.array(gamma_values)
        num_pairs = np.array(num_pairs)

        if plot:
            self._plot_variogram(lag_centers, gamma_values, num_pairs)
        
        self.init_exp = True

        return lag_centers, gamma_values, num_pairs
       
    def _plot_variogram(self, distances, semi_variance, pair_counts):
        """
        Plot the experimental variogram with the sill value.

        Parameters:
            x_col (str): Column name for x-coordinates.
            y_col (str): Column name for y-coordinates.
            lag_distance (float): User-defined lag distance for binning.
            max_distance (float, optional): Maximum distance to consider. Defaults to the maximum pairwise distance.

        Returns:
            None
        """

        exp_var = self.data[self.property].var()

        # Plot the variogram
        plt.figure(figsize=(16, 9))
        plt.plot(distances, semi_variance, marker='o', label='Experimental Variogram')

        if exp_var < 0.01:
            plt.axhline(y=exp_var, color='red', linestyle='--', label=f'Experimental Variance={round(exp_var, 5)}')
        
        else:
            plt.axhline(y=exp_var, color='red', linestyle='--', label=f'Experimental Variance={round(exp_var, 3)}')
            
        plt.title(f'Experimental Variogram of {self.property}')

        if self.unit_system == 'Imperial':
            plt.xlabel('Distance (h), ft')

        elif self.unit_system == 'SI':
            plt.xlabel('Distance (h), m')
        
        else:
            raise ValueError("Please provide a proper unit system!")
        
        # Annotate the number of data pairs
        for x, y, count in zip(distances, semi_variance, pair_counts):
            if not np.isnan(y):  # Only annotate non-NaN points
                plt.text(x, y, str(count), fontsize=14, ha='right', va='bottom')
        
        plt.ylabel(r'$\frac{1}{2} \cdot (z(x_i) - z(x_j))^2$')
        plt.legend()

        upper_limit_x = math.ceil((max(distances) + (max(distances) / self.step_coords)))
        plt.xlim(0, upper_limit_x)
        plt.ylim(0)

        plt.grid(alpha=0.5)
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14

        plt.tight_layout()

        plt.show()
    
    def directional_variogram(self, lag_distance, azimuth, tolerance, max_distance=None, plot=False):
        """
        Calculate a directional variogram for a given azimuth and tolerance.

        Parameters:
            lag_distance (float): Distance between bins (lag distance).
            azimuth (float): Direction in degrees (0° = North, 90° = East).
            tolerance (float): Tolerance in degrees for directional window.
            max_distance (float, optional): Maximum distance to consider. Defaults to the maximum pairwise distance.

        Returns:
            tuple: Bin midpoints, semi-variances, and number of pairs.
        """

        if lag_distance <= 0:
            raise ValueError("`lag_distance` must be positive.")
        if not (0 <= azimuth <= 360):
            raise ValueError("`azimuth` must be between 0 and 360 degrees.")
        if not (0 <= tolerance <= 90):
            raise ValueError("`tolerance` must be between 0 and 90 degrees.")

        coords = self.data[[self.X, self.Y]].values
        values = self.data[self.property].values

        distances = pairwise_distances(coords)
        angles = np.degrees(np.arctan2(
            coords[:, 1][:, None] - coords[:, 1],
            coords[:, 0][:, None] - coords[:, 0]
        )) % 360

        azimuth = azimuth % 360
        angle_min = (azimuth - tolerance) % 360
        angle_max = (azimuth + tolerance) % 360

        directional_mask = (
            (angles >= angle_min) & (angles <= angle_max)
            if angle_min < angle_max
            else (angles >= angle_min) | (angles <= angle_max)
        )

        if max_distance is None:
            max_distance = np.max(distances)

        bins = np.arange(0, max_distance + lag_distance, lag_distance)
        semi_variance = []
        pair_counts = []

        masked_distances = distances[directional_mask]
        masked_value_diffs = (values[:, None] - values)[directional_mask]

        # Calculate sill from the filtered data
        directional_values = values[directional_mask.any(axis=0)]  # Filter values
        directional_sill = np.var(directional_values)  # Variance of the filtered values

        for i in range(len(bins) - 1):
            bin_mask = (masked_distances >= bins[i]) & (masked_distances < bins[i + 1])
            pair_counts.append(np.sum(bin_mask))
            semi_variance.append(
                np.mean(masked_value_diffs[bin_mask] ** 2) / 2 if np.any(bin_mask) else np.nan
            )

        bin_midpoints = bins[:-1] + lag_distance / 2

        if plot:
            self._plot_directional_variogram(distances=bin_midpoints, semi_variance=semi_variance, 
                                             directional_sill=directional_sill, pair_counts=pair_counts, azimuth=azimuth)

        return bin_midpoints, semi_variance, pair_counts, directional_sill


    def _plot_directional_variogram(self, distances, semi_variance, directional_sill, pair_counts, azimuth):
        plt.figure(figsize=(16, 9))
        plt.plot(distances, semi_variance, marker='o', label=f'Directional Variogram (Azimuth={azimuth}°)')

        plt.axhline(y=directional_sill, color='red', linestyle='--', label=f'Sill = {directional_sill:.2f}')

        plt.title(f'Directional Variogram of {self.property} (Azimuth={azimuth}°)')

        if self.unit_system == 'Imperial':
            plt.xlabel('Distance (h), ft')

        elif self.unit_system == 'SI':
            plt.xlabel('Distance (h), m')
        
        else:
            raise ValueError("Please provide a proper unit system!")
        
        plt.ylabel(r'$\frac{1}{2} \cdot (z(x_i) - z(x_j))^2$')

        upper_limit_x = math.ceil(max(distances) / self.step_coords) * self.step_coords
        plt.xlim(0, upper_limit_x)
        plt.ylim(0)
        
        plt.grid(alpha=0.5)
        plt.legend()

        # Annotate the number of data pairs
        for x, y, count in zip(distances, semi_variance, pair_counts):
            if not np.isnan(y):  # Only annotate non-NaN points
                plt.text(x, y, str(count), fontsize=14, ha='right', va='bottom')

        plt.tight_layout()
        plt.show()


    def _fit_spherical_model(self, nugget, sill, range_):
        """
        Fit a spherical variogram model.

        Parameters:
            nugget (float): Nugget effect.
            sill (float): Sill (total variance).
            range_ (float): Range (distance where autocorrelation diminishes).

        Returns:
            callable: Spherical variogram function.
        """
        
        def spherical_model(h):
            h = np.array(h)
            gamma = np.zeros_like(h)

            # Within the range
            within_range = h <= range_
            gamma[within_range] = nugget + (sill - nugget) * (
                (3 * h[within_range]) / (2 * range_) - (h[within_range] ** 3) / (2 * range_ ** 3)
            )

            # Beyond the range
            gamma[h > range_] = sill

            # Check sill match
            gamma_at_range = nugget + (sill - nugget) * (
                (3 * range_) / (2 * range_) - (range_ ** 3) / (2 * range_ ** 3)
            )
            if not np.isclose(gamma_at_range, sill, atol=1e-2):
                print(f"Warning: Variogram at range ({gamma_at_range}) does not match sill ({sill}). Correcting...")
                gamma[h > range_] = sill

            return gamma

        return spherical_model


    def _plot_variogram_fitting(self, distances, semi_variances, variogram_func, nugget, sill, range_):
        """
        Plot the experimental variogram and the fitted spherical model, including residual analysis
        and a confidence interval band.

        Parameters:
            distances (array): Lag distances.
            semi_variances (array): Experimental semi-variance values.
            variogram_func (callable): Fitted spherical variogram function.
            nugget (float): Nugget effect.
            sill (float): Sill (total variance).
            range_ (float): Range (distance where autocorrelation diminishes).

        Returns:
            None
        """

        # Generate fitted variogram values
        fitted_variogram = variogram_func(distances)

        # Calculate residuals and RMSE
        residuals = semi_variances - fitted_variogram
        rmse = np.sqrt(np.nanmean(residuals**2))

        # Extend distances and fitted values to include the nugget point
        extended_distances = np.concatenate(([0], distances))
        extended_fitted_variogram = np.concatenate(([nugget], fitted_variogram))

        # Calculate extended confidence interval
        lower_bound = np.concatenate(([nugget - rmse], fitted_variogram - rmse))
        upper_bound = np.concatenate(([nugget + rmse], fitted_variogram + rmse))

        # Plot experimental and fitted variograms
        plt.figure(figsize=(16, 9))
        plt.plot(distances, semi_variances, 'o', label='Experimental Variogram', markersize=8)
        plt.plot(extended_distances, extended_fitted_variogram, '-', label='Fitted Spherical Model', linewidth=2)

        # Add confidence interval band
        plt.fill_between(extended_distances, lower_bound, upper_bound, color='orange', alpha=0.2, label='Confidence Interval')

        # Add annotations for sill and range
        plt.plot([0, range_], [sill, sill], color='red', linestyle='--', label=f'Sill = {round(sill, 2)}')  # Horizontal line limited to range
        plt.plot([range_, range_], [0, sill], color='purple', linestyle='--', label=f'Range = {round(range_, 2)}')  # Vertical line limited to sill

        # Add plot details
        plt.title('Variogram Fitting')

        if self.unit_system == 'Imperial':
            plt.xlabel('Distance (h), ft')

        elif self.unit_system == 'SI':
            plt.xlabel('Distance (h), m')

        else:
            raise ValueError("Please provide a proper unit system!")
        
        plt.ylabel(r'$\frac{1}{2} \cdot (z(x_i) - z(x_j))^2$')
        
        plt.xlim(0)
        plt.ylim(0)

        plt.legend()
        plt.grid(alpha=0.5)

        # Residual analysis
        plt.figtext(0.15, 0.85, f'RMSE: {rmse:.4f}', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

        plt.tight_layout()
        plt.show()


    def optimize_variogram(self, distances, semi_variances, pair_counts):
        """
        Optimize nugget, sill, and range for the spherical variogram model.

        Parameters:
            distances (array): Lag distances.
            semi_variances (array): Experimental semi-variance values.
            pair_counts (array): Number of data pairs in each bin.

        Returns:
            dict: Optimized nugget, sill, and range.
        """

        if self.init_exp is None:
            raise ValueError("Variogram cloud must be computed before constructing the experimental variogram.")
    
        # Adaptive binning strategy
        distances = np.array(distances)
        semi_variances = np.array(semi_variances)
        pair_counts = np.array(pair_counts)
        
        min_pairs = 10  # Minimum number of pairs required for a valid bin
        valid_bins = pair_counts >= min_pairs
        distances = distances[valid_bins]
        semi_variances = semi_variances[valid_bins]
        pair_counts = pair_counts[valid_bins]

        def objective(params):
            """
            Objective function to minimize during optimization.

            Parameters:
                params (list): [nugget, sill, range].

            Returns:
                float: Weighted RMSE with a penalty for mismatched sill.
            """

            nugget, sill, range_ = params
            variogram_func = self._fit_spherical_model(nugget, sill, range_)
            fitted_variogram = variogram_func(distances)

            # Residuals
            residuals = semi_variances - fitted_variogram

            # Weighted RMSE
            weights = np.array(pair_counts)
            weights = weights / weights.sum()  # Normalize weights
            weighted_rmse = np.sqrt(np.nansum(weights * residuals**2))

            # Enforce sill-range relationship
            sill_mismatch_penalty = np.abs(variogram_func(range_) - sill)

            return weighted_rmse + sill_mismatch_penalty * 10  # Combined objective

        # Initial guesses
        initial_guess = [0.01, max(semi_variances), max(distances) / 2]
        bounds = [(0, None), (0, max(semi_variances) * 2), (0, max(distances))]

        # Constraint to ensure variogram reaches sill at range
        constraints = ({
            'type': 'eq',
            'fun': lambda params: self._fit_spherical_model(params[0], params[1], params[2])(params[2]) - params[1]
        })

        # Optimization
        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        nugget, sill, range_ = result.x

        # Automatically plot the optimized variogram fitting
        self._plot_variogram_fitting(distances, semi_variances, 
                                    self._fit_spherical_model(nugget, sill, range_), 
                                    nugget, sill, range_)

        # Return optimized parameters
        return {"nugget": nugget, "sill": sill, "range": range_}