import pickle
import numpy as np
import random
import os
import pandas as pd
import yaml
import copy
from tqdm import tqdm
import xarray as xr

from LMRt.proxy import ProxyDatabase
from LMRt.gridded import Dataset
from LMRt.utils import (
    pp,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
    cfg_abspath,
    cwd_abspath,
    geo_mean,
    nino_indices,
    calc_tpi,
    global_hemispheric_means,
)

from .solver import GraphEM

class ReconJob:
    ''' Reconstruction Job

    General rule of loading parameters: load from the YAML first if available, then update with the parameters in the function calling,
      so the latter has a higher priority
    '''
    def __init__(self, configs=None, proxydb=None, prior=None, obs=None):
        self.configs = configs
        self.proxydb = proxydb
        self.prior = prior
        self.obs = obs

    def copy(self):
        return copy.deepcopy(self)

    def load_configs(self, cfg_path=None, job_dirpath=None, verbose=False):
        ''' Load the configuration YAML file

        self.configs will be updated

        Parameters
        ----------

        cfg_path : str
            the path of a configuration YAML file

        '''
        pwd = os.path.dirname(__file__)
        if cfg_path is None:
            cfg_path = os.path.abspath(os.path.join(pwd, './cfg/cfg_template.yml'))

        self.cfg_path = cfg_path
        if verbose: p_header(f'GraphEM: job.load_configs() >>> loading reconstruction configurations from: {cfg_path}')

        self.configs = yaml.safe_load(open(cfg_path, 'r'))
        if verbose: p_success(f'GraphEM: job.load_configs() >>> job.configs created')

        if job_dirpath is None:
            if os.path.isabs(self.configs['job_dirpath']):
                job_dirpath = self.configs['job_dirpath']
            else:
                job_dirpath = cfg_abspath(self.cfg_path, self.configs['job_dirpath'])
        else:
            job_dirpath = cwd_abspath(job_dirpath)

        self.configs['job_dirpath'] = job_dirpath
        os.makedirs(job_dirpath, exist_ok=True)
        if verbose:
            p_header(f'GraphEM: job.load_configs() >>> job.configs["job_dirpath"] = {job_dirpath}')
            p_success(f'GraphEM: job.load_configs() >>> {job_dirpath} created')
            pp.pprint(self.configs)

    def load_proxydb(self, path=None, verbose=False, load_df_kws=None):
        ''' Load the proxy database

        self.proxydb will be updated

        Parameters
        ----------
        
        proxydb_path : str
            if given, should point to a pickle file with a Pandas DataFrame underlying

        '''
        # update self.configs with not None parameters in the function calling
        if path is None:
            if os.path.isabs(self.configs['proxydb_path']):
                path = self.configs['proxydb_path']
            else:
                path = cfg_abspath(self.cfg_path, self.configs['proxydb_path'])
        else:
            path = cwd_abspath(path)

        self.configs['proxydb_path'] = path
        if verbose: p_header(f'GraphEM: job.load_proxydb() >>> job.configs["proxydb_path"] = {path}')

        # load proxy database
        proxydb = ProxyDatabase()
        proxydb_df = pd.read_pickle(self.configs['proxydb_path'])
        load_df_kws = {} if load_df_kws is None else load_df_kws.copy()
        proxydb.load_df(proxydb_df, ptype_psm='linear', ptype_season=None, verbose=verbose, **load_df_kws)
        if verbose: p_success(f'GraphEM: job.load_proxydb() >>> {proxydb.nrec} records loaded')

        proxydb.source = self.configs['proxydb_path']
        self.proxydb = proxydb
        if verbose: p_success(f'GraphEM: job.load_proxydb() >>> job.proxydb created')

    def filter_proxydb(self, ptype_list=None, dt=None, pids=None, verbose=False):
        ''' Filter the proxy database by proxy types and/or time resolution
        '''
        if ptype_list is None:
            ptype_list = self.configs['ptype_list']
        else:
            self.configs['ptype_list'] = ptype_list
            if verbose: p_header(f'GraphEM: job.filter_proxydb() >>> job.configs["ptype_list"] = {ptype_list}')

        proxydb = self.proxydb.copy()
        if ptype_list != 'all':
            if verbose: p_header(f'GraphEM: job.filter_proxydb() >>> filtering proxy records according to: {ptype_list}')
            proxydb.filter_ptype(ptype_list, inplace=True)

        if dt is not None:
            if verbose: p_header(f'GraphEM: job.filter_proxydb() >>> filtering proxy records according to: dt={dt}')
            proxydb.filter_dt(dt, inplace=True)

        if pids is not None:
            self.configs['assim_pids'] = pids
            if verbose: p_header(f'GraphEM: job.filter_proxydb() >>> job.configs["assim_pids"] = {pids}')

        if 'assim_pids' in self.configs and self.configs['assim_pids'] is not None:
            proxydb.filter_pids(self.configs['assim_pids'], inplace=True)

        if verbose: p_success(f'GraphEM: job.filter_proxydb() >>> {proxydb.nrec} records remaining')

        self.proxydb = proxydb

    def seasonalize_proxydb(self, ptype_season=None, verbose=False):
        ''' Seasonalize the proxy database
        '''
        if ptype_season is None:
            if 'ptype_season' not in self.configs:
                ptype_season = {}
                for ptype in self.configs['ptype_list']:
                    ptype_season[ptype] = list(range(1, 13))

                self.configs['ptype_season'] = ptype_season
                if verbose: p_header(f'GraphEM: job.seasonalize_proxydb() >>> job.configs["ptype_season"] = {ptype_season}')
            else:
                ptype_season = self.configs['ptype_season']

        else:
            self.configs['ptype_season'] = ptype_season
            if verbose: p_header(f'GraphEM: job.seasonalize_proxydb() >>> job.configs["ptype_season"] = {ptype_season}')

        proxydb = self.proxydb.copy()
        if self.configs['ptype_season'] is not None:
            if verbose: p_header(f'GraphEM: job.seasonalize_proxydb() >>> seasonalizing proxy records according to: {self.configs["ptype_season"]}')
            proxydb.seasonalize(self.configs['ptype_season'], inplace=True)
            if verbose: p_success(f'GraphEM: job.seasonalize_proxydb() >>> {proxydb.nrec} records remaining')

        self.proxydb = proxydb
        if verbose: p_success(f'GraphEM: job.seasonalize_proxydb() >>> job.proxydb updated')

    def load_obs(self, path_dict=None, varname_dict=None, verbose=False, anom_period=None):
        ''' Load instrumental observations fields

        Parameters
        ----------

        path_dict: dict
            a dict of environmental variables

        varname_dict: dict
            a dict to map variable names, e.g. {'tas': 'sst'} means 'tas' is named 'sst' in the input NetCDF file

        '''
        if path_dict is None:
            obs_path = cfg_abspath(self.cfg_path, self.configs['obs_path'])
        else:
            obs_path = cwd_abspath(path_dict)
        self.configs['obs_path'] = obs_path

        if anom_period is None:
            anom_period = self.configs['anom_period']
        else:
            self.configs['obs_anom_period'] = anom_period
            if verbose: p_header(f'GraphEM: job.load_obs() >>> job.configs["anom_period"] = {anom_period}')

        vn_dict = {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon',
        }
        if 'obs_varname' in self.configs:
            vn_dict.update(self.configs['obs_varname'])
        if varname_dict is not None:
            vn_dict.update(varname_dict)
        self.configs['obs_varname'] = vn_dict

        if verbose: p_header(f'GraphEM: job.load_obs() >>> loading instrumental observation fields from: {self.configs["obs_path"]}')

        ds = Dataset()
        ds.load_nc(self.configs['obs_path'], varname_dict=vn_dict, anom_period=anom_period, inplace=True)
        self.obs = ds
        if verbose: p_success(f'GraphEM: job.load_obs() >>> job.obs created')

    def seasonalize_obs(self, season=None, verbose=False):
        ''' Seasonalize the instrumental observations
        '''
        if season is None:
            if 'obs_season' not in self.configs:
                season = list(range(1, 13))
                self.configs['obs_season'] = season
                if verbose: p_header(f'GraphEM: job.seasonalize_obs() >>> job.configs["obs_season"] = {season}')
            else:
                season = self.configs['obs_season']
        else:
            self.configs['obs_season'] = season
            if verbose: p_header(f'GraphEM: job.seasonalize_obs() >>> job.configs["obs_season"] = {season}')

        ds = self.obs.copy()
        ds.seasonalize(self.configs['obs_season'], inplace=True)
        if verbose:
            p_hint(f'GraphEM: job.seasonalize_obs() >>> seasonalized obs w/ season {season}')
            print(ds)

        self.obs = ds
        if verbose: p_success(f'GraphEM: job.seasonalize_obs() >>> job.obs updated')

    def regrid_obs(self, ntrunc=None, verbose=False):
        ''' Regrid the instrumental observations
        '''
        if ntrunc is None:
            ntrunc = self.configs['obs_regrid_ntrunc']
        self.configs['obs_regrid_ntrunc'] = ntrunc

        ds = self.obs.copy()
        ds.regrid(self.configs['obs_regrid_ntrunc'], inplace=True)
        if verbose:
            p_hint('LMRt: job.regrid_obs() >>> regridded obs')
            print(ds)

        self.obs = ds
        if verbose: p_success(f'LMRt: job.regrid_obs() >>> job.obs updated')

    def prep_data(self, recon_period=None, calib_period=None, verbose=False):
        ''' A shortcut of the steps for data preparation
        '''
        if recon_period is None:
            recon_period = self.configs['recon_period']
        else:
            self.configs['recon_period'] = recon_period
            if verbose: p_header(f'GraphEM: job.prep_data() >>> job.configs["recon_period"] = {recon_period}')

        if calib_period is None:
            calib_period = self.configs['calib_period']
        else:
            self.configs['calib_period'] = calib_period
            if verbose: p_header(f'GraphEM: job.prep_data() >>> job.configs["calib_period"] = {calib_period}')

        recon_time = np.arange(recon_period[0], recon_period[1]+1)
        calib_time = np.arange(calib_period[0], calib_period[1]+1)
        self.recon_time = recon_time
        self.calib_time = calib_time
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.recon_time created')
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.calib_time created')

        tas = self.obs.fields['tas']
        tas_nt = np.shape(tas.value)[0]
        tas_2d = tas.value.reshape(tas_nt, -1)
        tas_npos = np.shape(tas_2d)[-1]

        nt = np.size(recon_time)
        temp = np.ndarray((nt, tas_npos))
        temp[:] = np.nan

        temp_calib_idx = [list(recon_time).index(t) for t in calib_time]
        self.calib_idx = temp_calib_idx
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.calib_idx created')

        tas_calib_idx = [list(tas.time).index(t) for t in calib_time]
        temp[temp_calib_idx] = tas_2d[tas_calib_idx]

        self.temp = temp
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.temp created')

        lonlat = np.ndarray((tas_npos+self.proxydb.nrec, 2))

        k = 0
        for i in range(tas.nlon):
            for j in range(tas.nlat):
                lonlat[k] = [tas.lon[i], tas.lat[j]]
                k += 1

        df_proxy = pd.DataFrame(index=recon_time)
        for pid, pobj in self.proxydb.records.items():
            series = pd.Series(index=pobj.time, data=pobj.value, name=pid)
            df_proxy = pd.concat([df_proxy, series], axis=1)
            lonlat[k] = [pobj.lon, pobj.lat]
            k += 1

        mask = (df_proxy.index>=recon_time[0]) & (df_proxy.index<=recon_time[-1])
        df_proxy = df_proxy[mask]

        self.df_proxy = df_proxy
        self.proxy = df_proxy.values
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.df_proxy created')
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.proxy created')

        self.lonlat = lonlat
        if verbose: p_success(f'GraphEM: job.prep_data() >>> job.lonlat created')

    def run_solver(self, save_path, verbose=False):
        ''' Run the GraphEM solver
        '''
        if os.path.exists(save_path):
            self.G = pd.read_pickle(save_path)
            if verbose: p_success(f'GraphEM: job.run_solver() >>> job.G created with the existing result at: {save_path}')
        else:
            G = GraphEM()
            G.fit(self.temp, self.proxy, self.calib_idx, lonlat=self.lonlat, graph_method='neighborhood')
            self.G = G
            pd.to_pickle(self.G, save_path)
            if verbose: p_success(f'GraphEM: job.run_solver() >>> job.G created and saved to: {save_path}')

        nt = np.shape(self.temp)[0]
        _, nlat, nlon = np.shape(self.obs.fields['tas'].value)
        self.recon = self.G.temp_r.reshape((nt, nlat, nlon))
        if verbose: p_success(f'GraphEM: job.run_solver() >>> job.recon created')

    def save(self, prep_savepath=None, verbose=False):
        ''' Save the job object for later use
        '''
        if prep_savepath is None:
            prep_savepath = os.path.join(self.configs['job_dirpath'], f'job.pkl')

        pd.to_pickle(self, prep_savepath)
        self.configs['prep_savepath'] = prep_savepath

        if verbose:
            p_header(f'LMRt: job.save_job() >>> Prepration data saved to: {prep_savepath}')
            p_header(f'LMRt: job.save_job() >>> job.configs["prep_savepath"] = {prep_savepath}')

    def save_recon(self, save_path, compress_dict={'zlib': True, 'least_significant_digit': 1}, verbose=False):
        ''' Save the reconstruction to a netCDF file
        '''
        output_dict = {}
        output_dict['recon'] = (('year', 'lat', 'lon'), self.recon)

        ds = xr.Dataset(
            data_vars=output_dict,
            coords={
                'year': self.recon_time,
                'lat': self.obs.fields['tas'].lat,
                'lon': self.obs.fields['tas'].lon,
            }
        )

        if compress_dict is not None:
            encoding_dict = {}
            for k in output_dict.keys():
                encoding_dict[k] = compress_dict

            ds.to_netcdf(save_path, encoding=encoding_dict)
        else:
            ds.to_netcdf(save_path)

        if verbose: p_header(f'LMRt: job.save_recon() >>> Reconstruction saved to: {save_path}')


    def run_cfg(self, cfg_path, job_dirpath=None, save_G_path=None, save_recon_path=None,
                verbose=False, obs_varname=None):
        ''' The top-level workflow to get reconstruction based on a configuration file directly
        '''
        self.load_configs(cfg_path, verbose=verbose)

        if job_dirpath is None:
            if os.path.isabs(self.configs['job_dirpath']):
                job_dirpath = self.configs['job_dirpath']
            else:
                job_dirpath = cfg_abspath(self.cfg_path, self.configs['job_dirpath'])
        else:
            job_dirpath = cwd_abspath(job_dirpath)

        self.configs['job_dirpath'] = job_dirpath
        os.makedirs(job_dirpath, exist_ok=True)
        if verbose:
            p_header(f'LMRt: job.load_configs() >>> job.configs["job_dirpath"] = {job_dirpath}')
            p_success(f'LMRt: job.load_configs() >>> {job_dirpath} created')

        self.load_proxydb(verbose=verbose)
        self.seasonalize_proxydb(verbose=verbose)

        if obs_varname is None:
            obs_varname = self.configs['obs_varname']

        self.load_obs(varname_dict=obs_varname, verbose=verbose)
        self.seasonalize_obs(verbose=verbose)

        self.prep_data(verbose=verbose)
        self.save(verbose=verbose)

        if save_G_path is None:
            save_G_path = os.path.join(job_dirpath, 'G.pkl')
            p_header(f'LMRt: job.run_cfg() >>> G will be saved to: {save_G_path}')
        self.run_solver(save_path=save_G_path, verbose=verbose)

        if save_recon_path is None:
            save_recon_path = os.path.join(job_dirpath, 'recon.nc')
            p_header(f'LMRt: job.run_cfg() >>> recon. will be saved to: {save_recon_path}')
        self.save_recon(save_path=save_recon_path, verbose=verbose)