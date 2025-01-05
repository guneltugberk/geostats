import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
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

    def plot_variability(self, ax, midpoints, mean, se, title, xlabel):
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
        sns.lineplot(x=midpoints, y=mean.values, ax=ax, lw=2, label=f'Mean {self.property}')
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

    def plot_property_variability(self, bins):
        """
        Plot variability along latitude and longitude with mean and standard error bands.

        Parameters:
            latitude_col (str): Column name for latitude.
            longitude_col (str): Column name for longitude.
            bins (int): Number of bins.

        Returns:
            None
        """

        fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
        sns.set_style('whitegrid')

        # Calculate statistics for latitude and longitude
        lat_midpoints, lat_mean, _, lat_se = self.calculate_variability(self.Y, bins)
        lon_midpoints, lon_mean, _, lon_se = self.calculate_variability(self.X, bins)

        if self.unit_system == 'Imperial':
            self.plot_variability(ax[0], lat_midpoints, lat_mean, lat_se, f'Variability of {self.property} on X', 'X (Midpoints), ft')
            self.plot_variability(ax[1], lon_midpoints, lon_mean, lon_se, f'Variability of {self.property} on Y', 'Y (Midpoints), ft')

        elif self.unit_system == 'SI':
            self.plot_variability(ax[0], lat_midpoints, lat_mean, lat_se, f'Variability of {self.property} on X', 'X (Midpoints), m')
            self.plot_variability(ax[1], lon_midpoints, lon_mean, lon_se, f'Variability of {self.property} on Y', 'Y (Midpoints), m')
            
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

    def calculate_variogram(self, lag_distance, max_distance=None):
        """
        Calculate the experimental variogram with user-defined inputs.

        Parameters:
            x_col (str): Column name for x-coordinates.
            y_col (str): Column name for y-coordinates.
            lag_distance (float): User-defined lag distance for binning.
            max_distance (float, optional): Maximum distance to consider. Defaults to the maximum pairwise distance.

        Returns:
            tuple: Bin midpoints, semi-variances and number of pairs.
        """

        coords = self.data[[self.X, self.Y]].values
        values = self.data[self.property].values

        # Compute pairwise distances and squared differences
        distances = pairwise_distances(coords)
        value_diffs = np.subtract.outer(values, values) ** 2

        # Define bins based on lag distance
        if max_distance is None:
            max_distance = np.max(distances)
            
        bins = np.arange(0, max_distance + lag_distance, lag_distance)

        # Calculate mean semi-variance for each bin
        semi_variance = []
        pair_counts = []

        for i in range(len(bins) - 1):
            mask = (distances >= bins[i]) & (distances < bins[i + 1])
            pair_counts.append(np.sum(mask))  # Count the number of pairs in the bin
            semi_variance.append(value_diffs[mask].mean() if np.any(mask) else np.nan)

        bin_midpoints = bins[:-1] + lag_distance / 2
        return bin_midpoints, semi_variance, pair_counts

    def plot_variogram(self, lag_distance, max_distance=None):
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
        
        distances, semi_variance, pair_counts = self.calculate_variogram(lag_distance, max_distance)
        sill = self.data[self.property].var()

        # Plot the variogram
        plt.figure(figsize=(16, 9))
        plt.plot(distances, semi_variance, marker='o', label='Experimental Variogram')

        if sill < 0.01:
            plt.axhline(y=sill, color='red', linestyle='--', label=f'Sill = {sill:.4f}')
        
        else:
            plt.axhline(y=sill, color='red', linestyle='--', label=f'Sill = {sill:.2f}')
            
        plt.title(f'Experimental Variogram of {self.property}')

        if self.unit_system == 'Imperial':
            plt.xlabel('Distance, ft')

        elif self.unit_system == 'SI':
            plt.xlabel('Distance, m')
        
        else:
            raise ValueError("Please provide a proper unit system!")
        
        # Annotate the number of data pairs
        for x, y, count in zip(distances, semi_variance, pair_counts):
            if not np.isnan(y):  # Only annotate non-NaN points
                plt.text(x, y, str(count), fontsize=14, ha='right', va='bottom')
        
        plt.ylabel('Semi-Variance')
        plt.legend()

        upper_limit_x = math.ceil(max(distances) / self.step_coords) * self.step_coords
        plt.xlim(0, upper_limit_x)
        plt.ylim(0)

        plt.grid(alpha=0.5)
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14

        plt.show()