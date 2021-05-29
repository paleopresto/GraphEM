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
        proxydb.load_df(proxydb_df, ptype_psm='linear',
            ptype_season=self.configs['ptype_season'], verbose=verbose, **load_df_kws)
        if verbose: p_success(f'GraphEM: job.load_proxydb() >>> {proxydb.nrec} records loaded')

        proxydb.source = self.configs['proxydb_path']
        self.proxydb = proxydb
        if verbose: p_success(f'GraphEM: job.load_proxydb() >>> job.proxydb created')