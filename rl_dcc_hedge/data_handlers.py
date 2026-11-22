from datetime import datetime
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import os
import ta

from torch.utils.data import DataLoader, Dataset
from configs.data_config import DataConfig
import yfinance as yf
import yaml
from dataclasses import asdict, fields
import rpy2.robjects as robjects
from rpy2.robjects import conversion, default_converter

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Taken from finrl.meta.preprocessor.yahoodownloader
    Modified for compatibility with latest yfinance==1.0

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list, interval: str = "1d",):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.interval = interval

    def fetch_data(self, auto_adjust=True) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=auto_adjust,
                interval=self.interval
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            if not auto_adjust:
                # convert the column names to standardized names
                data_df.rename(
                    columns={
                        "Date": "date",
                        "Adj Close": "adjcp",
                        "Close": "close",
                        "High": "high",
                        "Low": "low",
                        "Volume": "volume",
                        "Open": "open",
                        "tic": "tic",
                    },
                    inplace=True,
                )
            else:
                # convert the column names to standardized names
                data_df.rename(
                    columns={
                        "Date": "date",
                        "Close": "close",
                        "High": "high",
                        "Low": "low",
                        "Volume": "volume",
                        "Open": "open",
                        "tic": "tic",
                    },
                    inplace=True,
                )

        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df


def _parquet_caching_DataHandler_instance_function(f):
    def _parquet_caching_f(*args, **kwargs):
        this_instance = args[0]
        a = this_instance.args
        
        dataset_dir = os.path.join(a.datasets_root, a.dataset_name)
        cache_dir = os.path.join(dataset_dir, ".cache")
        cache_parquet_path = os.path.join(cache_dir, f.__name__+".parquet")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            df = pd.read_parquet(cache_parquet_path)
        except FileNotFoundError as e:
            print(f"{f.__name__} data could not be found at {cache_parquet_path}, fetching data and creating local copy...")
            
            df = f(*args, **kwargs)

            df.to_parquet(cache_parquet_path)

        return df
    return _parquet_caching_f

class DataHandler:
    PER_TIC_FEATURES_COLUMN_TAGS = ["innov", "condstd"]
    STANDALONE_FEATURES_COLUMN_TAGS = ["condcorr"]

    def __init__(self, args: DataConfig, init_test_only: bool = False):
        self.args = args
        self.seed_df_path = os.path.join(
            args.datasets_root,
            args.dataset_name,
            "seed_df.parquet"
        )

        self.eval_sim_seed = self.args.eval_sim_seed

        self.train_sim_seed = self.args.train_sim_seed

        self.test_sim_seed = self.args.test_sim_seed

        self.train_sim_save_path = os.path.join(args.datasets_root, args.dataset_name, f"train_simulation_{self.train_sim_seed}_{args.sim_steps}_{args.m_sim}_{args.startMethod}.parquet")
        self.eval_sim_save_path = os.path.join(args.datasets_root, args.dataset_name, f"eval_simulation_{self.eval_sim_seed}_{args.eval_sim_steps}_{args.eval_m_sim}_{args.startMethod}.parquet")
        self.test_sim_save_path = os.path.join(args.datasets_root, args.dataset_name, f"test_simulation_{self.test_sim_seed}_{args.test_sim_steps}_{args.test_m_sim}_{args.startMethod}.parquet")
        self.test_partition_save_path = os.path.join(args.datasets_root, args.dataset_name, f"test_partition.parquet")
        self.bootstrap_traj_dir = os.path.join(args.datasets_root, args.dataset_name, f"bootstrapped_test_simulation_{self.test_sim_seed + args.test_m_sim}_{args.test_sim_steps}_{args.num_bootstrapped_models}_{args.b_m_sim}_{args.startMethod}")

        # the R file replaces the problematic X.GSPC or ^GSPC with plain GSPC when reading the seed df
        # these tics match that convention
        self.tics = [tic.replace("^","") for tic in args.tics]

        self.anchor_asset = args.anchor_asset.replace("^","")
        self.secondary_asset = self.tics[0] if self.tics[1]==args.anchor_asset else self.tics[1]


        self._starting_prices = self._get_starting_prices()

        if (not init_test_only) and all((
            not os.path.exists(self.train_sim_save_path),
            not os.path.exists(self.eval_sim_save_path),
            not os.path.exists(self.test_sim_save_path),
            not os.path.exists(self.test_partition_save_path),
        )):
            print("No data found, running set-up...")
            self.create_seed_df()
            print("Processing seed df, will run the simulations and prepare datasets")
            self.process_seed_df()
        else:
            if not init_test_only:
                if not os.path.exists(self.train_sim_save_path):
                    print("Training data not found, running set-up...")
                    self.create_seed_df()
                    print("Processing seed df, will run the simulations and prepare the training data")
                    self.generate_training_data()
                if not os.path.exists(self.eval_sim_save_path):
                    print("Evaluation data not found, running set-up...")
                    self.create_seed_df()
                    print("Processing seed df, will run the simulations and prepare the evaluation data")
                    self.generate_evaluation_data()
            if not os.path.exists(self.test_sim_save_path):
                print("Synthetic test data not found, running set-up...")
                self.create_seed_df()
                print("Processing seed df, will run the simulations and prepare the synthetic test data")
                self.generate_synth_test_data()
            if not os.path.exists(self.test_partition_save_path):
                print("Training data not found, running set-up...")
                self.create_seed_df()
                print("Processing seed df, will run the simulations and prepare the test data")
                self.generate_testing_partition()


    def _get_starting_prices(self):
        raw_df = self.get_raw_df()
        seed_df = self.get_seed_df()

        price_obs_date = seed_df.index[seed_df.index < self.args.modeling_end_date][-1] # last date in modeling data

        # use raw (with ^) asset names to talk to raw_df
        raw_anchor_asset = self.args.anchor_asset
        raw_secondary_asset = self.args.tics[0] if self.args.tics[1]==raw_anchor_asset else self.args.tics[1]

        starting_prices = {
            "anchor": float(raw_df[(raw_df["date"]==price_obs_date) & (raw_df["tic"]==raw_anchor_asset)]["close"].values[0]),
            "secondary": float(raw_df[(raw_df["date"]==price_obs_date) & (raw_df["tic"]==raw_secondary_asset)]["close"].values[0])
        }

        # use internal names for internal business
        starting_prices[self.anchor_asset] = starting_prices["anchor"]
        starting_prices[self.secondary_asset] = starting_prices["secondary"]

        return starting_prices

    @property
    def starting_prices(self):
        return self._starting_prices

    @_parquet_caching_DataHandler_instance_function
    def get_raw_df(self):
        args = self.args

        df = YahooDownloader(start_date = args.dataset_start_date,
            end_date = args.dataset_end_date,
            ticker_list = args.tics,
            interval="1d"
        ).fetch_data()

        return df

    @_parquet_caching_DataHandler_instance_function
    def get_seed_df(self):
        df = self.get_raw_df()
        if self.args.df_transformation is not None:
            df = self.args.df_transformation(df)
        
        return df

    def create_seed_df(self):
        if os.path.exists(self.seed_df_path):
            print("WARNING: The seed df exists, continuing...")
            return

        seed_df = self.get_seed_df()
        seed_df.to_parquet(self.seed_df_path)

    def generate_training_data(self):
        """
        Parameters
        ----------
        args: 
        Requires DataHandler.create_seed_df to have been run
        - Reads seed df from disk
        - Fits model to seed df
        - Generates trajectories according to the fitted model
        - Filters the trajectories through the fitted model
        - Saves the data to disk
        """
        args = self.args

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)
    
        r_generate_training_data = robjects.globalenv['generate_training_data']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['model_save_path'] = 'fitted_model.RData'
        args_dict['train_sim_save_path'] = self.train_sim_save_path

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        args_dict['train_sim_seed'] = self.train_sim_seed

        args_dict['starting_prices'] = robjects.ListVector(self.starting_prices)

        r_generate_training_data(robjects.ListVector(args_dict))

    def generate_evaluation_data(self):
        """
        Parameters
        ----------
        args: 
        Requires DataHandler.create_seed_df to have been run
        - Reads seed df from disk
        - Fits model to seed df
        - Generates trajectories according to the fitted model
        - Filters the trajectories through the fitted model
        - Saves the data to disk
        """
        args = self.args

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)
    
        r_generate_evaluation_data = robjects.globalenv['generate_evaluation_data']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['model_save_path'] = 'fitted_model.RData'
        args_dict['eval_sim_save_path'] = self.eval_sim_save_path

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        args_dict['eval_sim_seed'] = self.eval_sim_seed

        args_dict['starting_prices'] = robjects.ListVector(self.starting_prices)

        r_generate_evaluation_data(robjects.ListVector(args_dict))

    def generate_synth_test_data(self):
        """
        Parameters
        ----------
        args: 
        Requires DataHandler.create_seed_df to have been run
        - Reads seed df from disk
        - Fits model to seed df
        - Generates trajectories according to the fitted model
        - Filters the trajectories through the fitted model
        - Saves the data to disk
        """
        args = self.args

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)
    
        r_generate_synth_test_data = robjects.globalenv['generate_synth_test_data']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['model_save_path'] = 'fitted_model.RData'
        args_dict['test_sim_save_path'] = self.test_sim_save_path

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        args_dict['test_sim_seed'] = self.test_sim_seed

        args_dict['starting_prices'] = robjects.ListVector(self.starting_prices)

        r_generate_synth_test_data(robjects.ListVector(args_dict))

    def generate_bootstrapped_synth_test_data(self, bootstrap_traj_dir: str = None):
        """
        Parameters
        ----------
        bootstrap_traj_dir: str
            Directory to save the bootstrapped models and trajectories to, defaults to DataHandler.bootstrap_traj_dir
        Requires DataHandler.create_seed_df to have been run
        - Reads seed df from disk
        - Fits model to seed df
        - Simulates bootstrap samples from the fitted model and refits a model to each
        - Generates b_m_sim trajectories according to each bootstrapped model
        - Filters the trajectories through the seed model
        - Rejects (and replaces) degenerate bootstrap samples, refits and trajectories, see the b_* guardrail fields of SimDataConfig,
          rejections are logged to <bootstrap_traj_dir>/bootstrap_rejections.csv
        - Saves the accepted models and their data to disk
        """
        args = self.args

        if bootstrap_traj_dir is None:
            bootstrap_traj_dir = self.bootstrap_traj_dir

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)

        r_generate_bootstrapped_synth_test_data = robjects.globalenv['generate_bootstrapped_synth_test_data']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['model_save_path'] = 'fitted_model.RData'
        args_dict['bootstrap_traj_dir'] = bootstrap_traj_dir

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        args_dict['test_sim_seed'] = self.test_sim_seed

        args_dict['starting_prices'] = robjects.ListVector(self.starting_prices)

        r_generate_bootstrapped_synth_test_data(robjects.ListVector(args_dict))

    def export_bootstrap_params(self, bootstrap_traj_dir: str = None):
        """
        Parameters
        ----------
        bootstrap_traj_dir: str
            Directory of the bootstrapped models, defaults to DataHandler.bootstrap_traj_dir
        Requires DataHandler.generate_bootstrapped_synth_test_data to have been run
        - Refits the seed model to the seed df
        - Loads every bootstrap_model_{i}.RData
        - Saves the parameters (plus derived persistence / unconditional moments) of
          all models to <bootstrap_traj_dir>/bootstrap_params.csv, the seed model as model_idx -1
        """
        args = self.args

        if bootstrap_traj_dir is None:
            bootstrap_traj_dir = self.bootstrap_traj_dir

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)

        r_export_bootstrap_params = robjects.globalenv['export_bootstrap_params']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['bootstrap_traj_dir'] = bootstrap_traj_dir

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        r_export_bootstrap_params(robjects.ListVector(args_dict))

    def generate_testing_partition(self):
        """
        Parameters
        ----------
        args: 
        Requires DataHandler.create_seed_df to have been run
        - Reads seed df from disk
        - Fits model to seed df
        - Generates trajectories according to the fitted model
        - Filters the trajectories through the fitted model
        - Saves the data to disk
        """
        args = self.args

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)
    
        r_generate_testing_partition = robjects.globalenv['generate_testing_partition']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['model_save_path'] = 'fitted_model.RData'
        args_dict['test_partition_save_path'] = self.test_partition_save_path

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        args_dict['starting_prices'] = robjects.ListVector(self.starting_prices)

        r_generate_testing_partition(robjects.ListVector(args_dict))


    def process_seed_df(self):
        """
        Parameters
        ----------
        args: 
        Requires DataHandler.create_seed_df to have been run
        - Reads seed df from disk
        - Fits model to seed df
        - Generates trajectories according to the fitted model
        - Filters the trajectories through the fitted model
        - Saves the data to disk
        """
        args = self.args

        # push cwd
        cwd = os.getcwd()
        os.chdir("r")
        robjects.r.source("fit.r")
        # pop cwd
        os.chdir(cwd)
    
        r_process_seed_df = robjects.globalenv['process_seed_df']

        args.prime_for_r()

        data_config_field_names = {f.name for f in fields(DataConfig)}
        args_dict = {name: getattr(args, name) for name in data_config_field_names} # DataConfig subset as a dict

        args_dict['seed_df_path'] = self.seed_df_path
        args_dict['seed_df_format'] = 'parquet'
        args_dict['model_save_path'] = 'fitted_model.RData'
        args_dict['train_sim_save_path'] = self.train_sim_save_path
        args_dict['eval_sim_save_path'] = self.eval_sim_save_path
        args_dict['test_sim_save_path'] = self.test_sim_save_path
        args_dict['test_partition_save_path'] = self.test_partition_save_path

        args_dict['subjects'] = [self.anchor_asset, self.secondary_asset] # align column ordering

        args_dict['train_sim_seed'] = self.train_sim_seed
        args_dict['eval_sim_seed'] = self.eval_sim_seed
        args_dict['test_sim_seed'] = self.test_sim_seed

        args_dict['starting_prices'] = robjects.ListVector(self.starting_prices)

        r_process_seed_df(robjects.ListVector(args_dict))

    def load_rl_evaluation_data(self):
        return pd.read_parquet(self.eval_sim_save_path)

    def load_rl_training_data(self):
        return pd.read_parquet(self.train_sim_save_path)

    def load_bootstrapped_synth_test_data(self, bootstrap_traj_dir: str = None) -> List[pd.DataFrame]:
        """
        Returns the bootstrapped synthetic test trajectories, generating them first if bootstrap_traj_dir does not exist

        Parameters
        ----------
        bootstrap_traj_dir: str
            Directory of the bootstrapped models and trajectories, defaults to DataHandler.bootstrap_traj_dir

        Returns
        -------
        dfs: List[pd.DataFrame]
            The trajectories of each bootstrapped model, ordered by model index
        """
        if bootstrap_traj_dir is None:
            bootstrap_traj_dir = self.bootstrap_traj_dir

        if not os.path.exists(bootstrap_traj_dir):
            print("Bootstrapped synthetic test data not found, running set-up...")
            self.create_seed_df()
            print("Processing seed df, will run the simulations and prepare the bootstrapped synthetic test data")
            self.generate_bootstrapped_synth_test_data(bootstrap_traj_dir)

        paths = [
            os.path.join(bootstrap_traj_dir, f"bootstrap_model_{i}.parquet")
            for i in range(self.args.num_bootstrapped_models)
        ]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{bootstrap_traj_dir} is missing {len(missing)} of {len(paths)} bootstrapped trajectory files: {missing}")

        return [pd.read_parquet(p) for p in paths]

    def load_bootstrap_params(self, bootstrap_traj_dir: str = None) -> pd.DataFrame:
        """
        Returns the parameter table of the bootstrapped models (model_idx -1 = the seed model),
        exporting it first if <bootstrap_traj_dir>/bootstrap_params.csv does not exist
        """
        if bootstrap_traj_dir is None:
            bootstrap_traj_dir = self.bootstrap_traj_dir

        params_path = os.path.join(bootstrap_traj_dir, "bootstrap_params.csv")
        if not os.path.exists(params_path):
            print(f"Bootstrapped model parameters not found, exporting them to {params_path}...")
            self.export_bootstrap_params(bootstrap_traj_dir)

        return pd.read_csv(params_path)

    def load_test_partition_data(self, get_in_parts:bool = True):
            """
            Returns the formatted data from start_date to end_date from the seed df
    
            Parameters
            ----------
            get_in_parts: bool
                If True the log_returns and features data will be returned in a dict
    
            Returns
            -------
            df: pd.DataFrame | Dict[str, pd.DataFrame]
                The corresponding data
            """
    
            test_partition = pd.read_parquet(self.test_partition_save_path)
            sampling_times = test_partition['dates'].values
            test_partition.drop(columns=['dates'], inplace=True)
    
            if get_in_parts:
                ret_cols = [
                    self.anchor_asset + "_returns",
                    self.secondary_asset + "_returns"
                ]
                prices_cols = [
                    self.anchor_asset + "_prices",
                    self.secondary_asset + "_prices"
                ]

                feature_cols = [
                    f"{tic}_{tag}"
                    for tic in [self.anchor_asset, self.secondary_asset]
                    for tag in self.PER_TIC_FEATURES_COLUMN_TAGS
                ]
                feature_cols.extend(self.STANDALONE_FEATURES_COLUMN_TAGS)

                out = {
                    "log_returns": test_partition[ret_cols],
                    "prices": test_partition[prices_cols],
                    "features": test_partition[feature_cols],
                    "sampling_times": sampling_times
                }
            else:
                out = test_partition
    
            return out
    