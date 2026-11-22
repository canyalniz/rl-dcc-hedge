import pandas as pd
import numpy as np
from typing import Tuple

def resampled_weekly_returns_factory(resampling_rule):
    def resampled_weekly_returns(df):
        df.date = pd.to_datetime(df.date)
        resampled_df = (
            df.set_index('date')
            .groupby('tic')
            .resample(resampling_rule)['close']
            .last()
            .reset_index()
        )
        resampled_df.sort_values(by=['tic', 'date'], inplace=True)
        resampled_df['log_returns'] = np.log(resampled_df['close'] / resampled_df.groupby('tic')['close'].shift(1))

        resampled_df['date'] = resampled_df['date'].dt.strftime('%Y-%m-%d')

        resampled_df = resampled_df.dropna().pivot(index='date', columns='tic', values='log_returns')

        return resampled_df

    return resampled_weekly_returns
    
def two_step_aggregation(
        tuple_of_trajectory_dfs:Tuple[pd.DataFrame],
        column_name:str,
        agg_func_1:callable=pd.Series.sum,
        agg_func_2:callable=pd.Series.mean,
):
    columns = [traj_df[column_name] for traj_df in tuple_of_trajectory_dfs]
    first_agg = pd.Series([agg_func_1(column) for column in columns])
    second_agg = agg_func_2(first_agg)
    return second_agg

def global_aggregation(
        tuple_of_trajectory_dfs:Tuple[pd.DataFrame],
        column_name:str,
        agg_func:callable = np.mean,
):
    values = np.concatenate([traj_df[column_name].values for traj_df in tuple_of_trajectory_dfs])
    return agg_func(values)