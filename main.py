from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.agents.stablebaselines3.models import DRLAgent

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import torch
import numpy as np
import random


def get_data(path):
    return pd.read_csv(path)

def process_data(df):
    df['date'] = pd.to_datetime(df['Open_time'], unit='ms')
    df['tic'] = 'BTC'
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df = df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]

    df = df.sort_values("date")
    return df

def arrange_dimensions(train):

    stock_dimension = len(train.tic.unique())
    num_stock_shares = [0] * stock_dimension
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    return stock_dimension, state_space, num_stock_shares

def split_data_by_date(date_str, df):
    train = df[df['date'] < date_str]
    test = df[df['date'] >= date_str]

    return train, test

def save_figure(df_account_value, model, name):
    plt.figure(figsize=(12,6))
    plt.plot(df_account_value['date'], df_account_value['account_value'])
    plt.xlabel("Tarih")
    plt.ylabel("Portföy Değeri (USD)")
    plt.title(f"{model} Ajanın Portföy Değeri Zaman İçinde")
    plt.grid()
    plt.savefig(name)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    df = get_data("solana_data.csv")
    df = process_data(df)
    dp = DataProcessor(data_source='binance', time_interval='4h') 
    processed = dp.clean_data(df)
    processed = dp.add_technical_indicator(processed, INDICATORS)
    processed = processed.dropna()  # Drop any rows with NaNs
    train,test = split_data_by_date('2024-11-01', processed)
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)


    stock_dimension, state_space, num_stock_shares = arrange_dimensions(train)
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension


    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct":sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,  # teknik göstergeler eklenmediyse boş bırak
        "action_space": stock_dimension,
        "reward_scaling": 1e-1,
    }
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env = env_train)
    model_ppo = agent.get_model("sac", model_kwargs={"gamma": 0.50})

    model = agent.train_model(model=model_ppo, 
                                tb_log_name='sac',
                                total_timesteps=50000)
    df_account_value, df_actions = agent.DRL_prediction(model=model, environment = e_train_gym)
    save_figure(df_account_value, "sac", "sac_model_train_gamma_50.png")
    perf_stats = backtest_stats(df_account_value)
    print(perf_stats)
    model.save("sac_trained_model_gamma_50")

