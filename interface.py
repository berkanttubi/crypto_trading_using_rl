import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from get_data import get_data
class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hisse Senedi Görselleştirici")

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

        # Grafik Paneli
        self.right_frame = tk.Frame(root, padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(8,5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_data(self, df):
        self.ax.clear()
        self.ax.plot(df['Close_time'], df['Close'], label="Close Fiyatı")
        self.ax.set_title("Kapanış Fiyatı Zaman Grafiği")
        self.ax.set_xlabel("Zaman")
        self.ax.set_ylabel("Fiyat")
        self.ax.tick_params(axis='x', labelrotation=45)
        self.ax.legend()
        self.canvas.draw()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV dosyaları", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
            self.plot_data(df)

    def load_data(self):
        symbol = self.symbol_combo.get()
        period = self.interval_combo.get()
        df = get_data(symbol = f'{symbol}USDT',interval = "4h",period=period)
        df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')

        self.plot_data(df)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()
