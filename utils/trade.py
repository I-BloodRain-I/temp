from typing import Any
import numpy as np
# import mplfinance as mpf
import pandas as pd

class Trade:
    def __init__(self, entry_price: float, position: str, trading_fee: float, stop_loss_pct: float, take_profit_pct: float, leverage: int):
        self.entry_price = entry_price
        self.prev_price = entry_price
        self.close_price = None
        self.is_closed_by_stop_loss = False
        self.is_closed_by_take_profit = False

        self.trading_fee = trading_fee
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct < 0 else -stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.leverage = leverage

        self.pnl = self.trading_fee * self.leverage * -1
        self.pnl_change = 0
        self.pnl_history = [self.trading_fee * self.leverage * -1]
        self.max_draw_down = self.trading_fee * self.leverage * -1

        self.position = position
        self.duration = 0

    def close(self, close_price: float) -> dict:
        if self.is_closed():
            raise Exception("Trade already closed")

        self.pnl = self.pnl_history[-1]
        self.max_draw_down = min(self.pnl_history)
        self.close_price = close_price
        return self.get_stats()

    def step(self, current_price: float) -> float:
        if self.is_closed():
            raise Exception("Trade already closed")

        pnl = self._calc_pnl(current_price) - (self.trading_fee * self.leverage)
        pnl_change = self._calc_pnl_change(current_price)
        if self._is_triggered_stop_loss(pnl):
            pnl = self.stop_loss_pct - (self.trading_fee * self.leverage)
            self.pnl = pnl
            self.max_draw_down = pnl
            self.close_price = current_price

        elif self._is_triggered_take_profit(pnl):
            pnl = self.take_profit_pct - (self.trading_fee * self.leverage)
            self.pnl = pnl
            self.close_price = current_price

        self.pnl_history.append(pnl)
        self.pnl_change = pnl_change
        self.duration += 1
        self.prev_price = current_price
        return pnl
    
    def is_closed(self) -> bool:
        if self.close_price is None:
            return False
        else:
            return True
    
    def get_pnl_change(self) -> float:
        return self.pnl_change

    def get_stats(self) -> dict[str, Any]:
        return {
            "entry_price": self.entry_price,
            "close_price": self.close_price,
            "is_close_by_stop_loss": self.is_closed_by_stop_loss,
            "is_close_by_take_profit": self.is_closed_by_take_profit,
            "close_pnl": self.pnl,
            "draw_down": self.max_draw_down,
            "pnl_history": self.pnl_history,
            "duration": self.duration,
            "position": self.position 
        }
    
    def _calc_pnl(self, price: float) -> float:
        if self.position == 'long':
            return ((price - self.entry_price) / self.entry_price) * self.leverage
        else:
            return ((self.entry_price - price) / self.entry_price) * self.leverage

    def _calc_pnl_change(self, price: float) -> float:
        if self.position == 'long':
            return ((price - self.prev_price) / self.prev_price) * self.leverage
        else:
            return ((self.prev_price - price) / self.prev_price) * self.leverage

    def _is_triggered_stop_loss(self, pnl: float) -> bool:
        if pnl <= self.stop_loss_pct:
            self.is_closed_by_stop_loss = True
            return True
        else:
            return False
        
    def _is_triggered_take_profit(self, pnl: float) -> bool:
        if pnl >= self.take_profit_pct:
            self.is_closed_by_take_profit = True
            return True
        else:
            return False
        
class TradesManager:
    def __init__(self, trading_fee: float = 0.0005, stop_loss: float = -1.0, take_profit: float = 1.0, leverage: int = 1, max_loss = -1.0, df: pd.DataFrame = None):
        self.trades = []
        self.current_trade = None
        self.pnl_history = []
        self.stop_loss = stop_loss if stop_loss < 0 else -stop_loss
        self.take_profit = take_profit
        self.max_loss = max_loss if max_loss < 0 else -max_loss
        self.leverage = leverage
        self.trading_fee = trading_fee
        self.df = df

    def open_trade(self, entry_price: float, position: str):
        self.current_trade = Trade(entry_price, position, self.trading_fee * 2, self.stop_loss, self.take_profit, self.leverage)

    def hold_trade(self, current_price: float):
        self.current_trade.step(current_price)
        return self._close_trade_check()
    
    def close_trade(self, close_price: float):
        self.current_trade.step(close_price)
        if not self.current_trade.is_closed():
            self.current_trade.close(close_price)
        self._close_trade_check()

    def _update_pnl_history(self, pnl_history: list):
        if self.pnl_history:
            last_pnl = self.pnl_history[-1]
            pnl_history_np = last_pnl + np.array(pnl_history)
            self.pnl_history.extend(pnl_history_np)
        else:
            self.pnl_history.extend(pnl_history)

    def _close_trade_check(self):
        is_closed = self.current_trade.is_closed()
        if is_closed:
            self.trades.append(self.current_trade)
            self._update_pnl_history(self.current_trade.pnl_history)
            self.current_trade = None
        return is_closed

    def is_active_trade(self):
        return True if self.current_trade is not None else False
    
    def is_pnl_larger_max_loss(self):
        if self.current_trade is not None:
            pnl = self.current_trade.pnl
        elif self.pnl_history:
            pnl = self.pnl_history[-1]
        else:
            pnl = 0

        if pnl <= self.max_loss:
            return True
        return False

    def get_opened_trade(self):
        return self.current_trade

    def get_stats_from_all_trades(self):
        pnls = []
        draw_downs = []
        durations = []
        closed_stop_loss = []
        closed_take_profit = []
        for trade in self.trades:
            stats = trade.get_stats()
            pnls.append(stats['close_pnl'])
            draw_downs.append(stats['draw_down'])
            durations.append(stats['duration'])
            closed_stop_loss.append(stats['is_close_by_stop_loss'])
            closed_take_profit.append(stats['is_close_by_take_profit'])

        stats = {
            "pnls": np.array(pnls),
            "draw_downs": np.array(draw_downs), 
            "durations": np.array(durations),
            "closed_stop_loss_count": sum(closed_stop_loss),
            "closed_take_profit_count": sum(closed_take_profit) 
        }
        return stats
    
    def get_pnl_history(self):
        return np.array(self.pnl_history)

    def create_plot(self, buy_np: np.ndarray, sell_np: np.ndarray):
        pass
        # self.df['open_time'] = pd.to_datetime(self.df['open_time'])
        # self.df.set_index('open_time', inplace=True)

        # buy_series = pd.Series(buy_np[:, 1], index=buy_np[:, 0], dtype=np.float32)
        # sell_series = pd.Series(sell_np[:, 1], index=sell_np[:, 0], dtype=np.float32)

        # apds = [
        #     mpf.make_addplot(buy_series, type='scatter', marker='^', color='g', markersize=100),
        #     mpf.make_addplot(sell_series, type='scatter', marker='v', color='r', markersize=100)
        # ]
        # style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='', facecolor='white')
        # mpf.plot(self.df,
        #         type='candle',
        #         addplot=apds,
        #         style=style,
        #         volume=False,
        #         title='DOLBOEB MAKE FARM',
        #         figsize=(12, 8))

    def clear(self):
        self.trades.clear()
        self.pnl_history.clear()
        self.current_trade = None