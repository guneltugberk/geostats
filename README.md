# Geostatistics Python Package

## Overview
The `Geostatistics` package provides a suite of tools for geostatistical analysis, including data visualization, variability analysis, and experimental variogram calculations. The package is designed to work with spatial data and offers flexibility for handling datasets in both Imperial and SI unit systems.

## Features
- Visualize spatial data distribution using scatter plots.
- Analyze variability of a property along coordinates with mean and standard error bands.
- Calculate experimental variograms to understand spatial dependence.
- Support for user-defined binning and lag distances.

## Installation
This project requires Python 3.7 or higher. Install the required dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
### Initialization
The `Geostatistics` class requires the following parameters for initialization:
- `data` (DataFrame): The input dataset.
- `property` (str): The name of the property column for analysis.
- `x_coord` (str): The name of the X coordinate column.
- `y_coord` (str): The name of the Y coordinate column.
- `step_property` (int, optional): Step size for property binning (default is 5).
- `step_coords` (int, optional): Step size for coordinate binning (default is 100).
- `unit_system` (str, optional): Unit system (default is 'Imperial').

### Example
```python
import pandas as pd
from geostatistics import Geostatistics

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize the Geostatistics class
gs = Geostatistics(data=data, property="Porosity", x_coord="X", y_coord="Y")

# Plot the base map
gs.base_map()

# Analyze variability
gs.plot_property_variability(bins=10)

# Calculate and plot experimental variogram
gs.plot_variogram(lag_distance=50)
```

## Methods
### `base_map()`
Visualizes the distribution of the property against X and Y coordinates using scatter plots.

### `calculate_variability(bin_col, bins)`
Calculates mean, variance, and standard error for binned data.
- **Parameters**:
  - `bin_col` (str): Column name for binning.
  - `bins` (int): Number of bins.
- **Returns**:
  - Tuple of bin midpoints, mean, variance, and standard error.

### `plot_variability(ax, midpoints, mean, se, title, xlabel)`
Plots mean and standard error for binned data.

### `plot_property_variability(bins)`
Visualizes the variability of the property along the X and Y coordinates with standard error bands.
- **Parameters**:
  - `bins` (int): Number of bins.

### `calculate_variogram(lag_distance, max_distance=None)`
Calculates the experimental variogram.
- **Parameters**:
  - `lag_distance` (float): Lag distance for binning.
  - `max_distance` (float, optional): Maximum distance for analysis.
- **Returns**:
  - Tuple of bin midpoints, semi-variances, and number of pairs.

### `plot_variogram(lag_distance, max_distance=None)`
Plots the experimental variogram with the sill value.
- **Parameters**:
  - `lag_distance` (float): Lag distance for binning.
  - `max_distance` (float, optional): Maximum distance for analysis.

## Dependencies
- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, please reach out to the project maintainer.

