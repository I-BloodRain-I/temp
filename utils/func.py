import json
import time
import requests
import numpy as np
from colorama import Fore, Style
from utils.settings import BASE_URL
import pandas as pd
from enum import Enum
import orjson
import functools

import humanize
import psutil
import logging
import torch

import os
import pkg_resources

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/func.log'),  
        logging.StreamHandler()                    
    ]
)
logger = logging.getLogger(__name__)

def load_json(filename: str):
    with open(f"{filename.replace('.json', '')}.json", 'rb') as f:
        return orjson.loads(f.read())

def save_json(filename: str, dict: dict):
    # def enum_encoder(obj):
    #     if isinstance(obj, Enum):
    #         return obj.name
    #     raise TypeError("Type not serializable")

    class Int32Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int32):
                return int(obj)
            return json.JSONEncoder.default(self, obj)

    json_obj = json.dumps(dict, indent=4, cls=Int32Encoder)
    file = open(f'{filename.replace(".json", "")}.json', 'w')
    file.write(json_obj)
    file.close()

def get_historical_data(symbol: str, history_time: int, interval: str, limit: int = None):
    """
    :param symbol (str): BTCUSDC, ETHUSDC
    :param history_time (int): in hours
    :param interval (str): 1/3/5/15/30m; 1/2/4/6/8/12h; 1/3d; 1w; 1M
    :param limit (int): count of candles
    """

    def _getTimeFrameInMinutes(interval: str):
        _type = interval[-1]
        _time = int(interval[:len(interval) - 1])
        time_frame_in_minutes = 0
        if _type == 'm':        time_frame_in_minutes = _time
        elif _type == 'h':      time_frame_in_minutes = _time * 60 
        elif _type == 'd':      time_frame_in_minutes = _time * 1440
        elif _type == 'w':      time_frame_in_minutes = _time * 10080
        elif _type == 'M':      time_frame_in_minutes = _time * 43800
        return time_frame_in_minutes

    def _getCountKlines(history_time: int, interval: str):
        time_frame_in_minutes = _getTimeFrameInMinutes(interval)

        if history_time * 60 < time_frame_in_minutes:
            logger.error(f"[getCountKlines] History time ({history_time * 60}) < time frame ({time_frame_in_minutes})")
            raise "Error _getCountKlines"

        history_time_in_minutes = history_time * 60
        return history_time_in_minutes // time_frame_in_minutes if (history_time_in_minutes // time_frame_in_minutes) < 1001 else 1000 

    def _getCountIter(interval, limit):
        time_frame_in_secounds = _getTimeFrameInMinutes(interval) * 60
        requests_limit = []
        requests_end_time = []
        idx = limit

        while idx > 1000:
            idx -= 1000
            requests_limit.append(1000)
            if len(requests_end_time) == 0:
                requests_end_time.append(int(time.time()))
            else:
                requests_end_time.append(int(requests_end_time[-1] - (time_frame_in_secounds * 1000)))

        requests_limit.append(idx)
        if len(requests_end_time) == 0:
            requests_end_time.append(int(time.time()))
        else:
            requests_end_time.append(int(requests_end_time[-1] - (time_frame_in_secounds * idx)))

        return requests_limit, requests_end_time

    if not limit:
        limits = [_getCountKlines(history_time, interval)]
        times = [int(time.time())]
    else:
        limits, times = _getCountIter(interval, limit)

    total_data = []
    for i in range(len(limits) - 1, -1, -1):
        url = f'{BASE_URL}/klines'
        params = {
            'endTime': times[i] * 1000,
            'symbol': symbol,
            'interval': interval,
            'limit': limits[i]
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(response.text)
            raise Exception(f'Historical data -> Status code: {response.status_code}')
        data = response.json()

        historical_data = []
        for candle_info in data:
            historical_data.append({
                "open_time": int(str(candle_info[0])[:10]),
                "open": float(candle_info[1]),
                "high": float(candle_info[2]),
                "low": float(candle_info[3]),
                "close": float(candle_info[4]),
                "volume": float(candle_info[5])
            })
        total_data += historical_data
        save_json(f'datasets/{symbol}{i}', historical_data)

    df = pd.DataFrame(total_data)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
    return df

def get_future_symbols(quote_symbol = None):
    url = f"{BASE_URL}/exchangeInfo"
    response = requests.get(url)

    if response.status_code == 200:
        exchange_info = response.json()  
        symbols = exchange_info['symbols'] 

        futures_symbols = []
        for symbol_info in symbols:
            if quote_symbol:
                if quote_symbol != symbol_info['quoteAsset']:
                    continue

            if symbol_info['contractType'] == 'PERPETUAL': 
                if symbol_info['status'] == "TRADING":
                    futures_symbols.append(symbol_info['symbol'])

        # log(f"[getFuturesSymbols] Futures symbols: {futures_symbols}")
        return futures_symbols
    else:
        logger.info(f"{Fore.RED}[getFuturesSymbols] Error: {response.status_code} - {response.text}{Style.RESET_ALL}")

def count_decimal_crypto(df: pd.DataFrame):
    def count_in_df(price):
        price_str = str(price)
        if '.' in price_str:
            return len(price_str.split('.')[1])
        else:
            return 0
    decimal_count_df = df['close'].apply(count_in_df)
    return decimal_count_df.max()

def scalling_data(x: list, lower_percentile=0, upper_percentile=100, min_val=0, max_val=1.0, target_min=0.01, target_max=1.0):
    lower_bound = np.percentile(x, lower_percentile)
    upper_bound = np.percentile(x, upper_percentile)
    scaled_x = (x - lower_bound) / (upper_bound - lower_bound)
    scaled_x = scaled_x * (target_max - target_min) + target_min
    scaled_x = np.clip(scaled_x, min_val, max_val)
    return scaled_x

def get_memory_usage():
    """
    Get current RAM and VRAM usage
    
    Returns:
        Tuple[float, float]: RAM usage in GB, VRAM usage in GB
    """
    # Get RAM usage
    process = psutil.Process()
    ram_usage = process.memory_info().rss  # in bytes
    
    # Get VRAM usage if CUDA is available
    if torch.cuda.is_available():
        vram_usage = torch.cuda.memory_reserved()  # in bytes
    else:
        vram_usage = 0
        
    return ram_usage / (1024**3), vram_usage / (1024**3)  # Convert to GB

def log_memory_usage(message: str = ""):
    """
    Log current RAM and VRAM usage with optional message
    """
    ram_gb, vram_gb = get_memory_usage()
    
    logger.info(f"{message}")
    logger.info(f"RAM Usage: {humanize.naturalsize(ram_gb * (1024**3), binary=True)}")
    if torch.cuda.is_available():
        logger.info(f"VRAM Usage: {humanize.naturalsize(vram_gb * (1024**3), binary=True)}")
    logger.info("-" * 50)

def calc_time(func, *args, return_time = False):
    start_t = time.perf_counter()
    result = func(*args)
    end_t = time.perf_counter()
    print(f"Function: {func.__name__} completed in {end_t - start_t:.4f}s")
    if return_time:
        return result, end_t - start_t
    else:
        return result

def calc_time_decor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        result = func(*args, **kwargs)
        end_t = time.perf_counter()
        print(f"Function: {func.__name__} completed in {end_t - start_t:.4f}s")
        return result
    return wrapper

def get_imported_modules(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    modules = []
    for line in lines:
        if line.startswith("import ") or line.startswith("from "):
            module = line.split()[1].split('.')[0]
            modules.append(module)
    return set(modules)

def generate_requirements(file_path):
    modules = get_imported_modules(file_path)
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    required_packages = set(modules) & installed_packages

    with open("requirements.txt", "w") as req_file:
        for package in required_packages:
            version = pkg_resources.get_distribution(package).version
            req_file.write(f"{package}=={version}\n")
