# Geostats

**Geostats** is a Python package for geostatistical analysis, including experimental variogram calculation, variogram model fitting, and kriging interpolation. The package provides tools for both omnidirectional and directional variogram analysis and supports spherical variogram modeling.

## Features

- **Visualization**:
  - Base map plotting for property distribution.
  - Variability analysis for spatial data.
- **Variogram Analysis**:
  - Experimental omnidirectional and directional variogram calculation.
  - Variogram model fitting with spherical models.
  - Confidence interval visualization for fitted variogram models.
- **Kriging Interpolation**:
  - Perform kriging interpolation with a fitted spherical variogram model.

---

## Installation

To install and use the `geostats` package, ensure you have the required dependencies installed. Use the following commands to set up the environment:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## How to Use

### 1. Initialization
Create an instance of the `Geostatistics` class by providing your dataset and relevant parameters.

```python
from geostats import Geostatistics
import pandas as pd

# Example dataset
data = pd.DataFrame({
    'X': [100, 200, 300, 400],
    'Y': [500, 600, 700, 800],
    'Property': [1.5, 2.3, 2.8, 3.1]
})

geo = Geostatistics(data, property='Property', x_coord='X', y_coord='Y')
```

### 2. Base Map
Visualize the distribution of the property on a scatter plot.

```python
geo.base_map()
```

### 3. Variability Analysis
Plot the variability of the property along the X and Y coordinates.

```python
geo.plot_property_variability(bins=5)
```

### 4. Omnidirectional Variogram
Calculate and plot the experimental omnidirectional variogram.

```python
lag_distance = 100
max_distance = 500
bin_midpoints, semi_variances, pair_counts = geo.omnidirectional_variogram(
    lag_distance=lag_distance, max_distance=max_distance, plot=True
)
```

### 5. Variogram Model Fitting
Fit a spherical variogram model to the experimental variogram data.

```python
optimized_params = geo.optimize_variogram(bin_midpoints, semi_variances, pair_counts)
print("Optimized Parameters:", optimized_params)
```

### 6. Kriging Interpolation
Perform kriging interpolation to predict a property at a given point.

```python
prediction_point = [250, 650]
nugget, sill, range_ = optimized_params["nugget"], optimized_params["sill"], optimized_params["range"]

predicted_value = geo.spherical_kriging(
    prediction_point=prediction_point,
    lag_distance=lag_distance,
    max_distance=max_distance,
    nugget=nugget,
    sill=sill,
    range_=range_
)
print("Predicted Value:", predicted_value)
```

---

## API Reference

### `class Geostatistics`

#### Initialization
```python
Geostatistics(data, property, x_coord, y_coord, step_property=5, step_coords=100, unit_system='Imperial')
```

- **data**: `pd.DataFrame` – Input dataset.
- **property**: `str` – Property column for analysis.
- **x_coord**: `str` – X coordinate column name.
- **y_coord**: `str` – Y coordinate column name.
- **step_property**: `int` – Step size for property analysis.
- **step_coords**: `int` – Step size for coordinate analysis.
- **unit_system**: `str` – Unit system, either `'Imperial'` or `'SI'`.

---

### Methods

#### Base Map
```python
base_map()
```
Plots a scatter map of the property distribution across X and Y coordinates.

#### Variability Analysis
```python
plot_property_variability(bins)
```
Plots the mean and standard error of the property along X and Y.

#### Omnidirectional Variogram
```python
omnidirectional_variogram(lag_distance, max_distance=None, plot=False)
```
Calculates and optionally plots the experimental omnidirectional variogram.

#### Directional Variogram
```python
directional_variogram(lag_distance, azimuth, tolerance, max_distance=None, plot=False)
```
Calculates and optionally plots the experimental directional variogram.

#### Variogram Fitting
```python
optimize_variogram(distances, semi_variances, pair_counts)
```
Optimizes nugget, sill, and range for a spherical variogram model and plots the fitted variogram.

#### Kriging Interpolation
```python
spherical_kriging(prediction_point, lag_distance, max_distance, nugget, sill, range_, variogram_type='omnidirectional')
```
Predicts the property value at a given point using spherical kriging.

---

## Example Plots

### 1. Base Map
A scatter plot visualizing the spatial distribution of the property.

### 2. Experimental Variogram
A variogram plot showing experimental semi-variance values and fitted spherical model with confidence intervals.

---

## Dependencies

The `geostats` package requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

---

## Contributing

Contributions to the `geostats` package are welcome! Feel free to open issues or submit pull requests on the repository.

---

## License

This package is licensed under the MIT License.

