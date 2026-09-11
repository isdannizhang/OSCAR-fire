# OSCAR-fire

## Overview

**OSCAR-fire** is an extension of the compact Earth system model **OSCAR v3**. This version uses a book-keeping scheme with prescribed burned area to assess wildfire carbon fluxes. For the original OSCAR v3 code, visit [OSCAR on GitHub](https://github.com/tgasser/OSCAR).

## Data Dimensions

OSCAR-fire uses the following dimensions for organizing its input and output data:

- **year**: time axis.
- **config**: Monte Carlo members, used for uncertainty analysis.
- **reg_code**: land regions in OSCAR v3
- **bio_land**: land biomes; the fire module is applied to Forest and Non-Forest.
- **veg_part**: vegetation parts (Leaf, Stem, Root).
- **pool_part**: combustible pools used for combustion completeness and mortality (Leaf, Stem, Root, Litter, CWD).

## Drivers

The model requires specific forcing data to run:

- **Aburn**: burned area during the annual time step (Mha yr<sup>-1</sup>).
- **cveg_0**: preindustrial vegetation carbon density (PgC Mha<sup>-1</sup>).
- **D_cveg**: change in vegetation carbon density relative to preindustrial (PgC Mha<sup>-1</sup>).
- **csoil_0**: preindustrial soil carbon density (PgC Mha<sup>-1</sup>).
- **D_csoil**: change in soil carbon density relative to preindustrial (PgC Mha<sup>-1</sup>).
- **veg_frac**: fraction of vegetation carbon in Leaf, Stem and Root (1).
- **frac_litter**: litter fraction of soil carbon (1).
- **Aland**: biome area (Mha).
- **D_tem**: land temperature anomaly used for CWD decay (K).

## Data Files and Accessibility

The model code is available at [GitHub repository](https://github.com/isdannizhang/OSCAR-fire). Experiment drivers, parameters and outputs are archived in netCDF format on [Zenodo](https://doi.org/10.5281/zenodo.22701517). The output data files represent constrained means and standard deviations from 2,000 configurations.
