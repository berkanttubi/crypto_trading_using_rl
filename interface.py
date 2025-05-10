import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from get_data import get_data
from train import create_environment, preprocess_data, create_agent, train

class StockApp:
    def __init__(self, root):
        self.train_environment = None
        self.train_environment_gym = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.agent = None
        self.trade_history = None
        self.total_timesteps = 0
        self.root = root
        self.root.title("BerkantFinRL")

        # Sol Panel
        self.left_frame = tk.Frame(root, padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Hisse Senedi Combobox
        tk.Label(self.left_frame, text="Hisse Senedi:").pack(anchor=tk.W)
        self.symbol_combo = ttk.Combobox(self.left_frame, values=["BTC", "ETH", "SOL"])
        self.symbol_combo.set("BTC")
        self.symbol_combo.pack(anchor=tk.W, fill=tk.X)

        # Periyot Combobox
        tk.Label(self.left_frame, text="Interval:").pack(anchor=tk.W)
        self.interval_combo = ttk.Combobox(self.left_frame, values=["1M", "3M", "6M", "1Y", "2Y", "5Y"])
        self.interval_combo.set("4h")
        self.interval_combo.pack(anchor=tk.W, fill=tk.X)

        # Butonlar
        self.load_button = tk.Button(self.left_frame, text="Veri Yükle", command=self.load_data)
        self.load_button.pack(pady=5, fill=tk.X)

        self.csv_button = tk.Button(self.left_frame, text="CSV'den Yükle", command=self.load_csv)
        self.csv_button.pack(pady=5, fill=tk.X)

        # Eğitim Bölümü
        self.train_frame = tk.LabelFrame(self.left_frame, text="Eğitim", padx=10, pady=10, bd=2, relief=tk.GROOVE, font=('Arial', 10, 'bold'))
        self.train_frame.pack(pady=10, fill=tk.X)

        self.environment_button = tk.Button(self.train_frame, text="Environment Yarat", command=self.open_env_window)
        self.environment_button.pack(pady=5, fill=tk.X)

        self.algorithm_combo = ttk.Combobox(self.train_frame, values=["PPO", "DDPG", "SAC"])
        self.algorithm_combo.set("PPO")
        self.algorithm_combo.pack(pady=5, fill=tk.X)

        self.agent_button = tk.Button(self.train_frame, text="Agent Yarat", command=self.open_agent_window)
        self.agent_button.pack(pady=5, fill=tk.X)

        self.train_button = tk.Button(self.train_frame, text="Train Model", command=self.train)
        self.train_button.pack(pady=5, fill=tk.X)

        # Grafik Paneli
        self.right_frame = tk.Frame(root, padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, (self.ax_data, self.ax_trade) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def train(self):
        self.trade_history = train(self.model, self.total_timesteps, self.agent, self.train_environment_gym)
        self.plot_trade_history()

    def plot_data(self, df):
        self.ax_data.clear()
        self.ax_data.plot(df['Close_time'], df['Close'], label="Close Fiyatı", color='blue')
        self.ax_data.set_title("Kapanış Fiyatı Zaman Grafiği")
        self.ax_data.set_ylabel("Fiyat")
        self.ax_data.tick_params(axis='x', labelrotation=45)
        self.ax_data.legend()
        self.canvas.draw()

    def plot_trade_history(self):
        self.ax_trade.clear()
        self.ax_trade.plot(self.trade_history['date'], self.trade_history['account_value'], label="Portföy Değeri", color='green')
        self.ax_trade.set_xlabel("Tarih")
        self.ax_trade.set_ylabel("Değer (USD)")
        self.ax_trade.set_title("Ajanın Portföy Değeri Zaman İçinde")
        self.ax_trade.tick_params(axis='x', labelrotation=45)
        self.ax_trade.legend()
        self.canvas.draw()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV dosyaları", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
            self.train_data, self.test_data = preprocess_data(df)
            self.plot_data(df)

    def load_data(self):
        symbol = self.symbol_combo.get()
        period = self.interval_combo.get()
        df = get_data(symbol=f'{symbol}USDT', interval="4h", period=period)
        df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
        self.train_data, self.test_data = preprocess_data(df)
        self.plot_data(df)

    def open_agent_window(self):
        env_window = tk.Toplevel(self.root)
        env_window.title("Agent Parametreleri")
        fields = {
            "epsilon_value": 0.99,
            "total_timesteps": 100
        }
        self.agent_entries = {}
        for i, (label, default) in enumerate(fields.items()):
            tk.Label(env_window, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = tk.Entry(env_window)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.agent_entries[label] = entry

        tk.Button(env_window, text="Kaydet", command=self.save_agent_params).grid(row=len(fields), columnspan=2, pady=10)

    def open_env_window(self):
        env_window = tk.Toplevel(self.root)
        env_window.title("Environment Parametreleri")
        fields = {
            "hmax": "10",
            "initial_amount": "100000",
            "num_stock_shares": "0",
            "buy_cost_pct": "0.001",
            "sell_cost_pct": "0.001",
            "state_space": "11",
            "stock_dim": "1",
            "tech_indicator_list": "['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma']",
            "action_space": "1",
            "reward_scaling": "0.0001",
        }

        self.env_entries = {}
        for i, (label, default) in enumerate(fields.items()):
            tk.Label(env_window, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = tk.Entry(env_window)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.env_entries[label] = entry

        tk.Button(env_window, text="Kaydet", command=self.save_env_params).grid(row=len(fields), columnspan=2, pady=10)

    def save_agent_params(self):
        agent_kwargs = {}
        for key, entry in self.agent_entries.items():
            agent_kwargs[key] = entry.get()

        self.model, self.agent = create_agent(self.train_environment, agent_kwargs["epsilon_value"])
        self.total_timesteps = int(agent_kwargs["total_timesteps"])

    def save_env_params(self):
        env_kwargs = {}
        for key, entry in self.env_entries.items():
            value = entry.get()
            if key in ["hmax", "initial_amount", "state_space", "stock_dim", "action_space"]:
                env_kwargs[key] = int(value)
            elif key in ["num_stock_shares", "buy_cost_pct", "sell_cost_pct"]:
                env_kwargs[key] = [eval(value)]
            elif key == "tech_indicator_list":
                env_kwargs[key] = eval(value)
            elif "." in value:
                env_kwargs[key] = float(value)
            else:
                env_kwargs[key] = int(value)

        self.train_environment, self.train_environment_gym = create_environment(env_kwargs, self.train_data)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()
