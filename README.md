
# Element Indicators Dataset

## Overview
This dataset contains 9 sustainability indicators for 18 elements commonly used in High Entropy Alloys (HEAs).

## Elements
The dataset covers the following elements:
Al, Co, Cr, Cu, Fe, Hf, Mn, Mo, Nb, Ni, Re, Ru, Si, Ta, Ti, V, W, Zr.

## Indicators
| Indicator Name         | Description                                                                 | Unit        |
|------------------------|-----------------------------------------------------------------------------|------------|
| Raw material price     | The market price of the element in its pure form.                          | USD/kg     |
| Supply risk            | Probability of supply disruptions due to geopolitical or natural factors.   | Probability (0-1) |
| Normalized vulnerability to supply restriction | Scarcity metric adjusted for global availability. | Probability (0-1)   |
| Embodied energy        | The energy required for primary production of the element.                 | MJ/kg      |
| Water usage            | The water consumed in the production of the element.                      | l/kg      |
| Rock to metal ratio    | Intensity of land use during mining operations.                            | kg/kg      |
| Human health damage    | Aggregate impact of element production on human health.                   | Index (0-100) |
| Human rights pressure  | Metric of human rights concerns in sourcing regions.                      | Index (0-100) |
| Labor rights pressure  | Metric of labor rights concerns in sourcing regions.                      | Index (0-100) |

## File Formats
- **CSV:** Standard comma-delimited file (`gen_18element_imputed_v202412.csv`).
- **JSON:** Machine-readable metadata (`gen_18element_imputed_v202412.json`).

## Access
The files are available on GitHub for open access.
