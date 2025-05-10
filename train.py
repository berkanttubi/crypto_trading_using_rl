from finrl.meta.data_processor import DataProcessor
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import pandas as pd
import matplotlib.pyplot as plt

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

def split_data_by_ratio(df, train_ratio=0.8):
    df = df.sort_values(by="date").reset_index(drop=True)  # Ensure chronological order
    split_index = int(len(df) * train_ratio)

    train = df[:split_index]
    test = df[split_index:]

    return train, test

def preprocess_data(df):
    df = process_data(df)
    dp = DataProcessor(data_source='binance', time_interval='4h') 
    processed = dp.clean_data(df)
    processed = dp.add_technical_indicator(processed, INDICATORS)
    processed = processed.dropna()  # Drop any rows with NaNs
    train,test = split_data_by_ratio(processed)
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)

    return train, test

def create_environment(env_variables, train):
    e_train_gym = StockTradingEnv(df = train, **env_variables)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train, e_train_gym


def create_agent(env_train, epsilon_value):
    agent = DRLAgent(env = env_train)
    model = agent.get_model("ppo", model_kwargs={"gamma": float(epsilon_value)})
    return model, agent


def train(model, total_timesteps, agent, train_env):
    model = agent.train_model(model=model,
                              tb_log_name='ppo',
                                total_timesteps=int(total_timesteps))
    df_account_value, df_actions = agent.DRL_prediction(model=model, environment = train_env)
    return df_account_value


