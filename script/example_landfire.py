##################################################
##   EXAMPLE: OSCAR_landfire
##################################################
##
## GOAL: run a short land-fire-only simulation on synthetic drivers.
##
## Usage (from the OSCAR-fire root):
##   python script/example_landfire.py
##

import os
import sys
import numpy as np
import xarray as xr

## repository root
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo)
sys.path.insert(0, repo)

from core_fct.mod_process import OSCAR_landfire
from core_fct.fct_loadP import load_all_param, load_land_fire
from core_fct.fct_genMC import generate_config


## choose a regional aggregation (same names as OSCAR v3)
mod_region = 'RCP_5reg'
nMC = 3
n_year = 10

## load all OSCAR v3 primary parameters, including fire
Par0 = load_all_param(mod_region)

## fire parameters for this regionalization
Par_fire = load_land_fire(mod_region).sel(bio_land=['Forest', 'Non-Forest'])

## book-keeping turnover (not in land_fire__*.nc; prescribed here for the example)
reg = Par_fire.reg_code
bio = Par_fire.bio_land
ones_rb = xr.ones_like(Par_fire.cc.isel(unc_Triangle=0, pool_part=0, drop=True))
Par_fire['mu_0'] = (0.05 * ones_rb).assign_attrs(units='yr-1')
Par_fire['rh_0'] = (0.03 * ones_rb).assign_attrs(units='yr-1')

## small Monte Carlo ensemble
Par = generate_config(Par_fire, nMC=nMC, mod_noise=0.1, seed=1)

## synthetic forcing
year = np.arange(2001, 2001 + n_year)
For = xr.Dataset()
For.coords['year'] = year
For.coords['reg_code'] = reg
For.coords['bio_land'] = bio
For.coords['veg_part'] = ['Leaf', 'Stem', 'Root']
For.coords['config'] = Par.config

ones_yrbc = xr.DataArray(
    np.ones((n_year, For.sizes['reg_code'], For.sizes['bio_land'], nMC)),
    coords={'year': year, 'reg_code': reg, 'bio_land': bio, 'config': Par.config},
    dims=['year', 'reg_code', 'bio_land', 'config'],
)
ones_rb = xr.DataArray(
    np.ones((For.sizes['reg_code'], For.sizes['bio_land'])),
    coords={'reg_code': reg, 'bio_land': bio},
    dims=['reg_code', 'bio_land'],
)
ones_yrb = xr.DataArray(
    np.ones((n_year, For.sizes['reg_code'], For.sizes['bio_land'])),
    coords={'year': year, 'reg_code': reg, 'bio_land': bio},
    dims=['year', 'reg_code', 'bio_land'],
)
ones_yr = xr.DataArray(
    np.ones((n_year, For.sizes['reg_code'])),
    coords={'year': year, 'reg_code': reg},
    dims=['year', 'reg_code'],
)

For['Aburn'] = (0.01 * ones_yrbc).assign_attrs(units='Mha yr-1')
For['Aland'] = (10. * ones_rb).assign_attrs(units='Mha')
For['cveg_0'] = (xr.DataArray([8., 3.], coords={'bio_land': bio}, dims='bio_land') * ones_rb).assign_attrs(units='PgC Mha-1')
For['csoil_0'] = (xr.DataArray([12., 6.], coords={'bio_land': bio}, dims='bio_land') * ones_rb).assign_attrs(units='PgC Mha-1')
For['D_cveg'] = (0. * ones_yrb).assign_attrs(units='PgC Mha-1')
For['D_csoil'] = (0. * ones_yrb).assign_attrs(units='PgC Mha-1')
For['D_tem'] = (0. * ones_yr).assign_attrs(units='K')
For['frac_litter'] = (0.2 * ones_rb).assign_attrs(units='1')
For['veg_frac'] = (
    xr.DataArray([0.15, 0.70, 0.15], coords={'veg_part': For.veg_part}, dims='veg_part') * ones_rb
).assign_attrs(units='1')

var_keep = ['D_Efire_inst', 'D_Efire_legacy', 'D_Efire', 'D_Ccwd', 'D_Cpyc', 'D_Fnet']

Out = OSCAR_landfire(
    Ini=None, Par=Par, For=For,
    var_keep=var_keep, keep_prog=True,
    nt=4, adapt_nt=False, scheme='ExpInt',
)

os.makedirs('results', exist_ok=True)
Out.to_netcdf('results/Out_example_landfire.nc', encoding={var: {'zlib': True, 'dtype': np.float32} for var in Out})

print(Out)
print('saved results/Out_example_landfire.nc')
print('global D_Efire (PgC yr-1), mean over config:')
print(Out.D_Efire.sum(['reg_code', 'bio_land']).mean('config').to_series())
