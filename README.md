
# Element Indicators Dataset

## Overview
This dataset contains nine sustainability indicators for 18 elements commonly used in High Entropy Alloys (HEAs).

## Elements
The dataset covers the following elements:
Al, Co, Cr, Cu, Fe, Hf, Mn, Mo, Nb, Ni, Re, Ru, Si, Ta, Ti, V, W, Zr.

## Indicators
| Indicator Name         | Description                                                                 | Unit        |
|------------------------|-----------------------------------------------------------------------------|------------|
| Raw Material Price     | The market price of the element in its pure form.                          | USD/kg     |
| Supply Risk            | Probability of supply disruptions due to geopolitical or natural factors.   | Probability (0-1) |
| Normalized Vulnerability to Supply Restriction | Scarcity metric adjusted for global availability. | Probability (0-1)   |
| Embodied Energy        | The energy required for primary production of the element.                 | MJ/kg      |
| Water Usage            | The water consumed in the production of the element.                      | l/kg      |
| Rock-to-Metal Ratio    | Intensity of land use during mining operations.                            | kg/kg      |
| Human Health Damage    | Aggregate impact of element production on human health.                   | Index (0-100) |
| Human Rights Pressure  | Metric of human rights concerns in sourcing regions.                      | Index (0-100) |
| Labor Rights Pressure  | Metric of labor rights concerns in sourcing regions.                      | Index (0-100) |

## File Formats
- **CSV:** Standard comma-delimited file (`gen_element_imputed_v202412.csv`).
- **JSON:** Machine-readable metadata (`gen_element_metadata_v202412.json`).

## Access
The files are available on GitHub for open access.
