import requests
from binance.um_futures import UMFutures
from .settings import *
from math import ceil, floor
import pandas as pd

def getBalance(client: UMFutures, quote_symbol: str):
    total_balance = client.balance()
    balance = float([symbol['availableBalance'] for symbol in total_balance if symbol['asset'] == quote_symbol][0])
    return balance

def calcLiquidationPrice(df: pd.DataFrame, symbol: str, bet: float, leverage: int):
    """
    df with buy/sell signals
    """
    def _liq_price(
        balance, main_margin, unrealized_profit, main_amount_both, main_amount_long, main_amount_short, side, qty_both, entry_price_both,
        qty_long, entry_price_long, qty_short, entry_price_short, main_margin_rate_both, main_margin_rate_long, main_margin_rate_short):
        
        numerator = balance - main_margin + unrealized_profit + main_amount_both + main_amount_long + \
                    main_amount_short - side * qty_both * entry_price_both - qty_long * entry_price_long + qty_short * entry_price_short
        denominator = qty_both * main_margin_rate_both + qty_long * main_margin_rate_long + qty_short * main_margin_rate_short - side * qty_both - qty_long + qty_short

        liq_price = numerator / denominator
        
        return liq_price
    
    def _maint_info(leverage_info: dict, total_bet: float):
            for bracket in leverage_info['brackets']:
                if total_bet > bracket['notionalFloor'] and total_bet <= bracket['notionalCap']:
                    return {"maintMarginRatio": bracket['maintMarginRatio'], "maintAmount": bracket['cum']}
        
    client = UMFutures(key=API_KEY, secret=API_SECRET)
    qty_length, min_qty = getCryptoPrecision(client, symbol, 'quote')
    leverage_info = client.leverage_brackets(symbol=symbol)[0]
    for i in range(len(df)):
        df_item = df.iloc[i]
        price = df_item['close']

        factor = 10 ** qty_length
        qty = floor((bet * leverage / price) * factor) / factor
        if qty < min_qty:
            df.loc[i, 'Liq Price'] = -1
            continue

        side = 1 if df_item['Signal'] == 'Buy' else -1
        maint_info = _maint_info(leverage_info, round(qty * price, 2))
        liq_price = _liq_price(bet, 0, 0, maint_info['maintAmount'], 0, 0, side, qty, price, 0, 0, 0, 0, maint_info['maintMarginRatio'], 0, 0)
        if liq_price < 0:
            liq_price = 0
        df.loc[i, 'Liq Price'] = liq_price
    return df

def getCryptoPrecision(client: UMFutures, symbol: str, type: str):
    """
    type: base/quote
    """
    url = f'{BASE_URL}/exchangeInfo'
    response = requests.get(url)
    data = response.json()
    
    tick_price = ""
    min_notional = 0
    for symbol_info in data['symbols']:

        if symbol_info['symbol'] == symbol:
            for _filter in symbol_info['filters']:
                if type == "base":
                    if _filter['filterType'] == 'PRICE_FILTER':
                        tick_price = float(_filter['tickSize'])
                else:
                    if _filter['filterType'] == 'LOT_SIZE':
                        tick_price = float(_filter['minQty'])
                    if _filter['filterType'] == 'MIN_NOTIONAL':
                        min_notional = float(_filter['notional'])

    if tick_price == "":
        raise "Error Tick Price"

    points = str(tick_price).split('.')
    precision = 0
    if points[-1] != "0":
        precision = len(points[-1])

    if type == "quote":
        # print(min_notional / float(client.mark_price(symbol)['markPrice']), precision)
        factor = 10 ** precision
        # min_qty = round(min_notional / float(client.mark_price(symbol)['markPrice']), precision)
        min_qty = ceil(min_notional / float(client.mark_price(symbol)['markPrice']) * factor) / factor
        return precision, min_qty
    return precision

def split_df_into_parts(df: pd.DataFrame, candle_step = 10):
    i_start = 0
    parts_df = []
    for i in range(int(len(df)/candle_step)):
        parts_df.append(df.iloc[i_start:i_start+candle_step])
        i_start += candle_step
    return parts_df

def format_number(number):
    number_str = str(number)
    if '.' in number_str:
        integer_part, decimal_part = number_str.split('.')
    else:
        integer_part, decimal_part = number_str, ''
    
    length = len(integer_part)
    if length <= 3:
        return f"{integer_part}{('.' + decimal_part) if decimal_part else ''}"

    parts = []
    while length > 3:
        parts.append(integer_part[-3:])  
        integer_part = integer_part[:-3]   
        length -= 3
        
    if integer_part:
        parts.append(integer_part)
    
    formatted_integer = ','.join(reversed(parts))
    return f"{formatted_integer}{('.' + decimal_part) if decimal_part else ''}"

def to_fixed(numObj, digits=0, type="str"):
    points = str(numObj).split('.')
    if len(points) == 1:
        points.append('00')

    if len(points[1]) > digits:
        points[1] = points[1][:digits]
    else: 
        while len(points[1]) < digits:
            points[1] += '0'

    if type == "str":
        return ".".join(points)
    else:
        return float(".".join(points))