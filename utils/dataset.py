import torch
import pandas as pd
import numpy as np
from utils.func import get_historical_data, scalling_data
from utils.settings import *

import technical.indicators as ind
# import technical.trend_neuro as tn
import technical.levels as lvl
import technical.patterns as patt

from .trade import TradesManager
from .testing import Testing
from .sequence import SequenceData

class TradingDataset:
    def __init__(self, folder = 'datasets'):
        self.folder = folder
        self.df = None

    def _check_diff_time(self):
        if self.df['open_time'].dtype.str.lower() == 'object':
            self.df['open_time'] = pd.to_datetime(self.df['open_time'])

        time_list = self.df['open_time'].values
        for i in range(1, len(time_list)):
            if time_list[i] - time_list[i - 1] != 3600000000000:
                print(time_list[i] - time_list[i - 1])
                raise Exception(f"Error {i}: {time_list[i]} - {time_list[i - 1]} != 3600000000000 ns")

    def load_data_from_history(self, symbol: str, interval: str, candle_count: int):
        self.df = get_historical_data(symbol, 1, interval, candle_count)
        self._check_diff_time()

    def load_data_from_csv(self, filename: str):
        self.df = pd.read_csv(f"{self.folder}/{filename}")
        self.df['open_time'] = pd.to_datetime(self.df['open_time']) 

    def add_trends(self, candle_count: int):
        t_model = Testing.load_model('trend_10.pth')
        s_model = Testing.load_model('sideways_10.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features = self.df[['adx', 'body_size', 'upper_shadow_size', 'lower_shadow_size', 'body_ratio', 'is_bullish', \
                            'volatility', 'rsi_top', 'rsi_bot', 'rsi_v_up', 'rsi_v_down', 'adx_up', 'adx_down', \
                            'close_pct_up', 'close_pct_down', 'volume_pct_up', 'volume_pct_down', 'rsi_prev_pct_up1', \
                            'rsi_prev_pct_down1', 'macd_pct_up', 'macd_pct_down', 'macd_prev_pct_up1', 'macd_prev_pct_down1',\
                            'macd_hist_v_up', 'macd_hist_v_down', 'macd_hist_prev_pct_up1', 'macd_hist_prev_pct_down1',\
                            'ema_4_pct_up', 'ema_4_pct_down', 'ema_4_prev_pct_up1', 'ema_4_prev_pct_down1', 'ema_10_pct_up',\
                            'ema_10_pct_down', 'ema_10_prev_pct_up1', 'ema_10_prev_pct_down1', 'adx_prev_pct_up1',\
                            'adx_prev_pct_down1', 'bbands_pct_up', 'bbands_pct_down', 'bbands_prev_pct_up1', 'bbands_prev_pct_down1',\
                            'n_body_size_prev_pct_up1', 'n_body_size_prev_pct_down1', 'n_upper_shadow_size_prev_pct_up1',\
                            'n_upper_shadow_size_prev_pct_down1', 'n_lower_shadow_size_prev_pct_up1',\
                            'n_lower_shadow_size_prev_pct_down1', 'n_body_ratio_prev_pct_up1', 'n_body_ratio_prev_pct_down1',\
                            'volatility_prev_pct_up1', 'volatility_prev_pct_down1']].values.astype(np.float32)

        seq_data = SequenceData(candle_count)
        seq_data.fill(features)

        trend_pred, trend_prob = Testing.binary_predict(t_model, torch.from_numpy(seq_data.get_data()).to(device))
        side_pred, side_prob = Testing.binary_predict(s_model, torch.from_numpy(seq_data.get_data()).to(device))

        assert trend_pred.shape == side_pred.shape
        assert trend_pred.shape[0] == seq_data.get_data().shape[0]

        nan_points = candle_count - 1
        trend_pred, trend_prob = np.concatenate([[np.nan] * nan_points, trend_pred]), np.concatenate([[np.nan] * nan_points, trend_prob])
        side_pred, side_prob = np.concatenate([[np.nan] * nan_points, side_pred]), np.concatenate([[np.nan] * nan_points, side_prob]) 

        self.df[f's_pred{candle_count}'] = side_pred
        self.df[f's_prob{candle_count}'] = side_prob
        self.df[f't_pred{candle_count}'] = trend_pred
        self.df[f't_prob{candle_count}'] = trend_prob

    def add_trends_as_binary(self, candle_count: int, trend_threshold: float, sideways_threshold: float):
        if f's_prob{candle_count}' not in self.df.columns.to_list():
            self.add_trends(candle_count)

        s_pred, s_prob = self.df[f's_pred{candle_count}'].values, self.df[f's_prob{candle_count}'].values
        t_pred, t_prob = self.df[f't_pred{candle_count}'].values, self.df[f't_prob{candle_count}'].values

        trends = [] # bull, bear, side
        for i in range(len(s_pred)):
            if s_prob[i] < sideways_threshold:
                trends.append([0, 0, 1])
                continue
            
            if s_pred[i] == 0:
                trends.append([0, 0, 1])
                continue

            if t_prob[i] < trend_threshold:
                trends.append([0, 0, 1])
                continue
            
            if t_pred[i] == 1:
                trends.append([1, 0, 0])
            else:
                trends.append([0, 1, 0])

        trends = np.array(trends)

        self.df[f'side{candle_count}'] = trends[:, 2]
        self.df[f'bear{candle_count}'] = trends[:, 1]
        self.df[f'bull{candle_count}'] = trends[:, 0]
        self.df.drop(columns=[f's_pred{candle_count}', f's_prob{candle_count}', f't_pred{candle_count}', f't_prob{candle_count}'], inplace=True)

    def add_indicators(self,
            wpr_index = WPR_INDEX,
            ao_index = AO_INDEX,
            ao_index_2 = AO_INDEX_2,
            stochrsi_index = STOCHRSI_INDEX,
            stochrsi_index_2 = STOCHRSI_INDEX_2,
            stochrsi_index_3 = STOCHRSI_INDEX_3,
            uo_index = UO_INDEX,
            uo_index_2 = UO_INDEX_2,
            uo_index_3 = UO_INDEX_3,
            cmf_index = CMF_INDEX,
            mfi_index = MFI_INDEX,
            rsi_index = RSI_INDEX,
            stoch_index = STOCH_INDEX,
            stoch_index_2 = STOCH_INDEX_2,
            cci_index = CCI_INDEX,
            mom_index = MOM_INDEX,
            adx_index = ADX_INDEX,
            adx_index2 = ADX_INDEX2,
            macd_index = MACD_INDEX,
            macd_index_2 = MACD_INDEX_2,
            macd_index_3 = MACD_INDEX_3,
            ma_index = MA_INDEX,
            ma_index_2 = MA_INDEX_2,
            ma_index_3 = MA_INDEX_3,
            ma_index_4 = MA_INDEX_4,
            ma_index_5 = MA_INDEX_5,
            is_sma = True,
            sar_index = SAR_INDEX,
            sar_index_2 = SAR_INDEX_2,
            ichimoku_index = ICHIMOKU_INDEX,
            ichimoku_index_2 = ICHIMOKU_INDEX_2,
            ichimoku_index_3 = ICHIMOKU_INDEX_3,
            bbp_index = BBP_INDEX,
            bbands_index = BBANDS_INDEX,
            bbands_index_2 = BBANDS_INDEX_2,
            atr_index = ATR_INDEX,
            obv_index = OBV_INDEX,
            fear_greed_index = FEAR_GREED_INDEX,
        ):
        self.df = ind.eval_indicators(
            df=self.df,
            wpr_index=wpr_index,
            ao_index=ao_index,
            ao_index_2=ao_index_2,
            stochrsi_index=stochrsi_index,
            stochrsi_index_2=stochrsi_index_2,
            stochrsi_index_3=stochrsi_index_3,
            uo_index=uo_index,
            uo_index_2=uo_index_2,
            uo_index_3=uo_index_3,
            cmf_index=cmf_index,
            mfi_index=mfi_index,
            rsi_index=rsi_index,
            stoch_index=stoch_index,
            stoch_index_2=stoch_index_2,
            cci_index=cci_index,
            mom_index=mom_index,
            adx_index=adx_index,
            adx_index2=adx_index2,
            macd_index=macd_index,
            macd_index_2=macd_index_2,
            macd_index_3=macd_index_3,
            ma_index=ma_index,
            ma_index_2=ma_index_2,
            ma_index_3=ma_index_3,
            ma_index_4=ma_index_4,
            ma_index_5=ma_index_5,
            is_sma=is_sma,
            sar_index=sar_index,
            sar_index_2=sar_index_2,
            ichimoku_index=ichimoku_index,
            ichimoku_index_2=ichimoku_index_2,
            ichimoku_index_3=ichimoku_index_3,
            bbp_index=bbp_index,
            bbands_index=bbands_index,
            bbands_index_2=bbands_index_2,
            atr_index=atr_index,
            obv_index=obv_index,
            fear_greed_index=fear_greed_index,
        )

    def add_patterns(self):
        self.df, columns = patt.search_patterns(self.df)
        cols_to_drop = []
        for col in columns:
            if col not in ['CDLDRAGONFLYDOJI', 'CDLGRAVESTONEDOJI', 'CDLLONGLEGGEDDOJI', 'CDLMATCHINGLOW', 'CDLRICKSHAWMAN', 'CDLTAKURI', 'hammer', 'INVhammer', 'doji', 'CDL3OUTSIDE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLENGULFING', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLLONGLINE', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'engulfing', 'threeSoldiers']:
                continue
            if col.startswith('CDL'):
                self.df[col] = np.where(self.df[col] > 0, 1, np.where(self.df[col] < 0, -1, 0))
            else:
                self.df[f"OWN{col.upper()}"] = np.where(self.df[col] > 0, 1, np.where(self.df[col] < 0, -1, 0))
                cols_to_drop.append(col)

        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)

    def add_levels(self,
        min_bounces: int = None,
        min_candles: int = None,
        min_diff_pct: float = None
    ):
        levels = lvl.Levels(self.df)
        prices = self.df['close'].values
        # для разогрева
        levels.get_nearest_levels(prices[300], 300)

        support_price = [np.nan] * self.df.shape[0]
        support_bounces = [0] * self.df.shape[0]
        support_candles = [0] * self.df.shape[0]

        resistance_price = [np.nan] * self.df.shape[0]
        resistance_bounces = [0] * self.df.shape[0]
        resistance_candles = [0] * self.df.shape[0]

        max_devision = 0.03

        prev_support_level, prev_resistance_level = None, None
        for idx in range(300, len(prices)):
            is_updated = False
            price = prices[idx]
        
            if prev_support_level is None and prev_resistance_level is not None:
                if (price > prev_resistance_level['price'] \
                or abs(prev_resistance_level['price'] - price) / prev_resistance_level['price'] > max_devision) \
                and not is_updated:
                    support_level, resistance_level = levels.get_nearest_levels(price, idx, min_bounces, min_candles, min_diff_pct)
                    prev_support_level, prev_resistance_level = support_level, resistance_level
                    is_updated = True
                else:
                    support_level, resistance_level = prev_support_level, prev_resistance_level

            elif prev_support_level is not None and prev_resistance_level is None:
                if (price < prev_support_level['price'] \
                or abs(prev_support_level['price'] - price) / prev_support_level['price'] > max_devision) \
                and not is_updated:
                    support_level, resistance_level = levels.get_nearest_levels(price, idx, min_bounces, min_candles, min_diff_pct)
                    prev_support_level, prev_resistance_level = support_level, resistance_level
                    is_updated = True
                else:
                    support_level, resistance_level = prev_support_level, prev_resistance_level

            if prev_support_level is not None:
                if price < prev_support_level['price'] and not is_updated:
                    support_level, resistance_level = levels.get_nearest_levels(price, idx, min_bounces, min_candles, min_diff_pct)
                    prev_support_level, prev_resistance_level = support_level, resistance_level
                    is_updated = True
                else:
                    support_level, resistance_level = prev_support_level, prev_resistance_level

            if prev_resistance_level is not None:
                if price > prev_resistance_level['price'] and not is_updated:
                    support_level, resistance_level = levels.get_nearest_levels(price, idx, min_bounces, min_candles, min_diff_pct)
                    prev_support_level, prev_resistance_level = support_level, resistance_level
                    is_updated = True
                else:
                    support_level, resistance_level = prev_support_level, prev_resistance_level

            if prev_support_level is None and prev_resistance_level is None and not is_updated:
                support_level, resistance_level = levels.get_nearest_levels(price, idx, min_bounces, min_candles, min_diff_pct)
                prev_support_level, prev_resistance_level = support_level, resistance_level

            if isinstance(support_level, dict):
                support_price[idx] = support_level['price']
                support_bounces[idx] = support_level.get('price_bounces', 0)
                support_candles[idx] = support_level.get('cons_candles', 0)

            if isinstance(resistance_level, dict):
                resistance_price[idx] = resistance_level['price']
                resistance_bounces[idx] = resistance_level.get('price_bounces', 0)
                resistance_candles[idx] = resistance_level.get('cons_candles', 0)

        self.df['support_price'] = support_price
        self.df['support_bounces'] = support_bounces
        self.df['support_candles'] = support_candles
        self.df['resistance_price'] = resistance_price
        self.df['resistance_bounces'] = resistance_bounces
        self.df['resistance_candles'] = resistance_candles

    @staticmethod
    def _calc_max_profit(prices: np.ndarray, stop_loss_pct: float, direction: str, current_step: int, check_period: int, price = None):
        current_price = prices[current_step]
        if price is not None:
            assert current_price == price
        max_len = min(current_step + check_period + 1, len(prices))
        prices_slice = prices[current_step + 1:max_len]
        if len(prices_slice) == 0:
            return 0, 0
        if direction == 'long':
            profit = (prices_slice - current_price) / current_price
        elif direction == 'short':
            profit = (current_price - prices_slice) / current_price
        else:
            raise Exception('Direction incorrected')

        stop_loss_pct = stop_loss_pct if stop_loss_pct < 0 else -stop_loss_pct
        if np.min(profit) <= stop_loss_pct:
            max_profit = -1
            max_profit_index = 0
        else:
            max_profit = float(np.max(profit))
            max_profit_index = int(np.argmax(profit))
        return max_profit, max_profit_index + 1

    def _add_rewards_by_max_profit(self, direction, check_period, min_profit, stop_loss_pct):
        self.df['rewards'] = np.nan
        prices = self.df['close'].values
        rewards = []
        
        for i in range(len(prices) - check_period):
            max_profit, _ = TradingDataset._calc_max_profit(prices, stop_loss_pct, direction, i, check_period)
            rewards.append(0 if max_profit <= min_profit else 1)
        
        rewards.extend([np.nan] * check_period)
        return np.array(rewards)

    def calibrate_min_profit(self, direction, check_period, target_ratio=0.35, stop_loss_pct=1.0, 
                            min_profit_range=(-0.05, 0.1), tolerance=0.001, max_iterations=50):
        """
        Подбирает min_profit для достижения целевого соотношения меток 0/1
        
        Args:
            df: DataFrame с ценовыми данными
            direction: направление торговли ('long' или 'short')
            check_period: период проверки
            target_ratio: целевое соотношение нулей (0.35 = 35% нулей)
            min_profit_range: диапазон поиска min_profit
            tolerance: допустимое отклонение от целевого соотношения
            max_iterations: максимальное число итераций
        """
        left, right = min_profit_range
        iteration = 0
        
        while iteration < max_iterations:
            mid = (left + right) / 2
            
            rewards = self._add_rewards_by_max_profit(direction, check_period, mid, stop_loss_pct)
            
            # Считаем текущее соотношение
            zeros = (rewards == 0).sum()
            total = len(rewards[~np.isnan(rewards)])
            current_ratio = zeros / total
            # print(iteration, mid, current_ratio)
            
            # Проверяем достижение цели
            if abs(current_ratio - target_ratio) < tolerance:
                return mid
                
            # Корректируем границы поиска
            if current_ratio > target_ratio:
                right = mid
            else:
                left = mid
                
            iteration += 1
        
        return (left + right) / 2

    def add_rewards_by_max_profit(self, direction: str, check_period: int, min_profit: float, stop_loss_pct = 1.0):
        self.direction = direction
        self.check_period = check_period
        self.min_profit = min_profit

        prices = self.df['close'].values
        rewards = []

        for i in range(len(prices) - check_period):
            max_profit, distance  = self._calc_max_profit(prices, stop_loss_pct, direction, i, check_period)
            if max_profit <= min_profit:
                rewards.append(0)
            else:
                rewards.append(1)
        rewards += [np.nan for _ in range(check_period)]

        self.df['rewards'] = np.array(rewards)
        df_rewards = self.df['rewards'].values
        print(f"0: {df_rewards[df_rewards == 0].shape[0]}, 1: {df_rewards[df_rewards == 1].shape[0]} ({df_rewards[df_rewards == 0].shape[0]/df_rewards.shape[0]:.2%}/{df_rewards[df_rewards == 1].shape[0]/df_rewards.shape[0]:.2%})")

    def test_rewards(self, min_pnl: float, start_i = 0, trading_fee = 0.0005, stop_loss_pct = 1.0, leverage = 1):
        manager = TradesManager(
            trading_fee=trading_fee, 
            stop_loss=stop_loss_pct, 
            leverage=leverage
        )

        prices = self.df['close'].iloc[start_i:].values
        print(prices.shape)
        close_idx = 0
        for idx, price in enumerate(prices):
            if idx + self.check_period == len(prices):
                break

            if not manager.is_active_trade():
                max_profit, _ = self._calc_max_profit(prices, stop_loss_pct / leverage, self.direction, idx, self.check_period, price)
                if max_profit < min_pnl:
                    continue
                manager.open_trade(price, self.direction)
                _, close_idx = self._calc_max_profit(prices, stop_loss_pct / leverage, self.direction, idx, self.check_period, price)
                close_idx += idx
                continue

            else:
                if idx - close_idx == 0:
                    manager.close_trade(price)
                    continue
                manager.hold_trade(price)
        
        stats = manager.get_stats_from_all_trades()
        pnls = stats['pnls']
        draw_downs = stats['draw_downs']
        durations = stats['durations']
        closed_stop_loss_count = stats['closed_stop_loss_count']

        if pnls.shape[0] == 0:
            print('Zero trades')
        else:
            print(f"Trades   -> {pnls[pnls > 0].shape[0]}/{pnls.shape[0]}, Winrate: {pnls[pnls > 0].shape[0]/pnls.shape[0]:.2%}, "
                  f"Stop Loss: {closed_stop_loss_count} ({closed_stop_loss_count/pnls.shape[0]:.2%})")
            print(f"PNL      -> Sum: {pnls.sum():.2%} ({pnls.sum() + (pnls.shape[0] * (trading_fee * 2) * leverage):.2%}), Min: {pnls.min():.2%}, Max: {pnls.max():.2%}, Mean: {pnls.mean():.2%}")
            print(f"DrawDown -> Max: {draw_downs.min():.2%}, Mean: {draw_downs.mean():.2%}")
            print(f"Duration -> Min: {durations.min()}, Max: {durations.max()}, Mean: {durations.mean():.2f}")

    def normalize(self, max_shift_period: int = None):
        # Candle
        body_size = (self.df['close'] - self.df['open']).abs()
        upper_shadow_size = self.df['high'] - np.maximum(self.df['open'], self.df['close']) 
        lower_shadow_size = np.minimum(self.df['open'], self.df['close']) - self.df['low']
        body_ratio = np.where(body_size / (upper_shadow_size + lower_shadow_size) != np.inf, body_size / (upper_shadow_size + lower_shadow_size), 0.5)

        self.df['n_body_size'] = body_size
        self.df['n_upper_shadow_size'] = upper_shadow_size
        self.df['n_lower_shadow_size'] = lower_shadow_size
        self.df['n_body_ratio'] = body_ratio

        self.df['body_size'] = scalling_data(body_size, 0, 99.5, target_min=0.0)
        self.df['upper_shadow_size'] = scalling_data(upper_shadow_size, 0, 99.67, target_min=0.0)
        self.df['lower_shadow_size'] = scalling_data(lower_shadow_size, 0, 99.67, target_min=0.0)
        self.df['body_ratio'] = scalling_data(body_ratio, 0, 99.25, target_min=0.0)
        self.df['is_bullish'] = np.where(self.df['close'] > self.df['open'], 1, 0)

        # Market
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['volatility'] = self.df['log_return'].rolling(window=24).std() * np.sqrt(252)  # за 24 свечи / годовая активность
        self.df.drop(columns=['log_return'], inplace=True)

        # [0-100]
        if 'rsi' in self.df.columns:
            self.df['rsi_top'] = np.where(self.df['rsi'] >= 70, 1, 0)
            self.df['rsi_bot'] = np.where(self.df['rsi'] <= 30, 1, 0)
            self.df['rsi_v_up'] = np.where(self.df['rsi'] >= 50, ((self.df['rsi'] - 50) / 100) * 2, 0)
            self.df['rsi_v_down'] = np.where(self.df['rsi'] < 50, ((50 - self.df['rsi']) / 100) * 2, 0)

        if 'uo' in self.df.columns:
            # self.df['uo_top'] = np.where(self.df['uo'] >= 70, 1, 0)
            # self.df['uo_bot'] = np.where(self.df['uo'] <= 30, 1, 0)
            self.df['uo_v_up'] = np.where(self.df['uo'] >= 50, ((self.df['uo'] - 50) / 100) * 2, 0)
            self.df['uo_v_down'] = np.where(self.df['uo'] < 50, ((50 - self.df['uo']) / 100) * 2, 0)

        if 'cmf' in self.df.columns:
            self.df['cmf_top'] = np.where(self.df['cmf'] >= 0.1, 1, 0)
            self.df['cmf_bot'] = np.where(self.df['cmf'] <= -0.1, 1, 0)
            self.df['cmf_v_up'] = np.where(self.df['cmf'] >= 0, self.df['cmf'], 0)
            self.df['cmf_v_down'] = np.where(self.df['cmf'] < 0, self.df['cmf'].abs(), 0)

        if 'mfi' in self.df.columns:
            self.df['mfi_top'] = np.where(self.df['mfi'] >= 80, 1, 0)
            self.df['mfi_bot'] = np.where(self.df['mfi'] <= 20, 1, 0)
            self.df['mfi_v_up'] = np.where(self.df['mfi'] >= 50, ((self.df['mfi'] - 50) / 100) * 2, 0)
            self.df['mfi_v_down'] = np.where(self.df['mfi'] < 50, ((50 - self.df['mfi']) / 100) * 2, 0)

        if 'adx' in self.df.columns:
            self.df['adx'] /= 100
            self.df['adx_up'] = np.where(self.df['adx_pos'] >= self.df['adx_neg'], 1, 0)
            self.df['adx_down'] = np.where(self.df['adx_pos'] < self.df['adx_neg'], 1, 0)

        if 'stoch_main' in self.df.columns:
            self.df['stoch'] = self.df['stoch_main'] / 100
            self.df['stoch_top'] = np.where((self.df['stoch_main'] >= 80) & ((self.df['stoch_side'] >= 80)), 1, 0)
            self.df['stoch_bot'] = np.where((self.df['stoch_main'] <= 20) & ((self.df['stoch_side'] <= 20)), 1, 0)
            self.df['stoch_up'] = np.where(self.df['stoch_side'] >= self.df['stoch_main'], 1, 0)
            self.df['stoch_down'] = np.where(self.df['stoch_side'] < self.df['stoch_main'], 1, 0)

        if 'wr' in self.df.columns:
            self.df['wr'] += 100
            self.df['wr_top'] = np.where(self.df['wr'] >= 80, 1, 0)
            self.df['wr_bot'] = np.where(self.df['wr'] <= 20, 1, 0)
            self.df['wr_v_up'] = np.where(self.df['wr'] >= 50, ((self.df['wr'] - 50) / 100) * 2, 0)
            self.df['wr_v_down'] = np.where(self.df['wr'] < 50, ((50 - self.df['wr']) / 100) * 2, 0)

        # Other
        # if 'bbands_lower' in self.df.columns:
        #     self.df['bbands_above'] = np.where(self.df['close'] < self.df['bbands_lower'], 1, 0)
        #     self.df['bbands_below'] = np.where(self.df['close'] > self.df['bbands_upper'], 1, 0)

        # cols_to_drop = ['open_time', 'open', 'low', 'high', 'rsi', 'uo', 'cmf', 'mfi', 'adx_pos', 'adx_neg', 'stoch_side', 'stoch_main', 'stoch_rsi', 'stoch_k', 'stoch_d', 'wr', 'fgi', 'bbands_upper', 'bbands_lower', 'ao_prev', 'ichimoku_base_line', 'ichimoku_conversion_line', 'ichimoku_a', 'ichimoku_b', 'bull_power', 'bear_power', 'n_body_size', 'n_upper_shadow_size', 'n_lower_shadow_size', 'n_body_ratio']
        cols_to_drop = ['rsi', 'uo', 'cmf', 'mfi', 'adx_pos', 'adx_neg', 'stoch_side', 'stoch_main', 'stoch_rsi', 'stoch_k', 'stoch_d', 'wr', 'fgi', 'bbands_upper', 'bbands_lower', 'ao_prev', 'ichimoku_base_line', 'ichimoku_conversion_line', 'ichimoku_a', 'ichimoku_b', 'bull_power', 'bear_power', 'n_body_size', 'n_upper_shadow_size', 'n_lower_shadow_size', 'n_body_ratio']
        self.df.rename(columns={"bbands_middle": 'bbands', 'bull_bear_power': 'bbp'}, inplace=True)
        df_columns = self.df.drop(columns=['open_time']).columns.to_list()
        for col in df_columns:
            # Price Based pct (indicators)
            if col.startswith('ema_') or col.startswith('sma_') \
            or col in ['sar', 'bbands', 'vwap', 'macd']:
                if col == 'macd':
                    pct = ((self.df[col] - self.df['macd_signal']) / self.df['macd_signal']) * 100
                else:
                    pct = ((self.df[col] - self.df['close']) / self.df['close']) * 100
                dir_up = np.where(pct > 0, pct, 0)
                dir_down = np.where(pct < 0, pct.abs(), 0)
                self.df[f'{col}_pct_up'] = scalling_data(dir_up, 0, 99.5, target_min=0.0)
                self.df[f'{col}_pct_down'] = scalling_data(dir_down, 0, 99.5, target_min=0.0)
                cols_to_drop.append(col)

            # Convert 1 direction to 2 directions (indicators)
            if col in ['macd_hist', 'cci', 'momentum', 'ao', 'bbp']:
                dir_up = np.where(self.df[col] > 0, self.df[col], 0)
                dir_down = np.where(self.df[col] < 0, self.df[col].abs(), 0)
                self.df[f'{col}_v_up'] = scalling_data(dir_up, 0, 99.5, target_min=0.0)
                self.df[f'{col}_v_down'] = scalling_data(dir_down, 0, 99.5, target_min=0.0)
                cols_to_drop.append(col)

            # Previous Value Based pct (indicators)
            if col in ['obv', 'ATR', 'close', 'volume']:
                pct = self.df[col].pct_change(1)
                dir_up = np.where(pct > 0, pct, 0)
                dir_down = np.where(pct < 0, pct.abs(), 0)
                self.df[f'{col}_pct_up'] = scalling_data(dir_up, 0, 99.5, target_min=0.0)
                self.df[f'{col}_pct_down'] = scalling_data(dir_down, 0, 99.5, target_min=0.0)
                if col != 'close':
                    cols_to_drop.append(col)

            # Previous Value Based pct while <= max_shift_period (indicators)
            if max_shift_period is not None:
                if col.startswith('ema_') or col.startswith('sma_') \
                or col in ['sar', 'bbands', 'vwap', 'macd', 'macd_hist', 'cci', 'momentum', 'ao', 'bbp',
                        'n_body_size', 'n_upper_shadow_size', 'n_lower_shadow_size', 'n_body_ratio',
                        'volatility', 'rsi', 'uo', 'cmf', 'mfi', 'adx', 'stoch_main', 'wr', 'close']:

                    if col in ['n_body_size', 'n_upper_shadow_size', 'n_lower_shadow_size', 'n_body_ratio']:
                        self.df.loc[self.df[col] == 0, col] = 1
                    else:
                        self.df.loc[self.df[col] == 0, col] = 0.01

                    for i in range(1, max_shift_period + 1):
                        if col == 'close' and i == 1:
                            continue
                        
                        pct = self.df[col].pct_change(i)
                        dir_up = np.where(pct > 0, pct, 0)
                        dir_down = np.where(pct < 0, pct.abs(), 0)

                        if not np.isnan(dir_up[i:]).any():
                            if col in ['n_body_size', 'n_upper_shadow_size', 'n_lower_shadow_size', 'n_body_ratio']:
                                self.df[f'{col}_prev_pct_up{i}'] = scalling_data(dir_up, 0, 95, target_min=0.0)
                            else:
                                self.df[f'{col}_prev_pct_up{i}'] = scalling_data(dir_up, 0, 99.7, target_min=0.0)
                        if not np.isnan(dir_down[i:]).any():
                            if col in ['n_body_size', 'n_upper_shadow_size', 'n_lower_shadow_size', 'n_body_ratio']:
                                self.df[f'{col}_prev_pct_down{i}'] = scalling_data(dir_down, 0, 95, target_min=0.0)
                            else:
                                self.df[f'{col}_prev_pct_down{i}'] = scalling_data(dir_down, 0, 99.7, target_min=0.0)

            # Convert 1 directory to 2 directory (patterns)
            if col.startswith('CDL') or col.startswith('OWN'):
                if col in ['CDLHAMMER', 'CDLDOJI']: # duplicate with own
                    cols_to_drop.append(col)

                # elif self.df[col].dropna()[self.df[col].dropna() != 0].shape[0] < int(len(self.df) / 100):
                #     cols_to_drop.append(col)

                elif self.df[col].min() == -1:
                    self.df[f'{col}_down'] = np.where(self.df[col] < 0, self.df[col].abs(), 0)
                    if self.df[col].max() == 1:
                        self.df[f'{col}_up'] = np.where(self.df[col] > 0, self.df[col], 0)
                    cols_to_drop.append(col)
                    
            # Clipped bounces/candles (levels)
            if col in ['support_bounces', 'support_candles', 'resistance_bounces', 'resistance_candles']:
                if col.endswith('bounces'):
                    pct = np.clip(self.df[col], 0, 5) / 5
                elif col.endswith('candles'):
                    pct = np.clip(self.df[col], 0, 40) / 40
                self.df[col] = pct

            # Price Based Pct (levels)
            if col in ['support_price', 'resistance_price']:
                # 10% price
                ten_percent_price = self.df['close'] * 0.1
                if col == 'support_price':
                    pct = ((self.df['close'] - self.df[col]) / ten_percent_price)
                else:
                    pct = ((self.df[col] - self.df['close']) / ten_percent_price)
                clipped = 1 - np.clip(pct, 0, 1)
                clipped[300:] = np.nan_to_num(clipped[300:], nan=0)
                self.df[col] = clipped

        # drop 0 from max period ma
        max_period_ma = 0
        for col in df_columns:
            if col.startswith('sma_') or col.startswith('ema_'):
                period = int(col.split('_')[1])
                if period > max_period_ma:
                    max_shift_period = period
        self.df = self.df.iloc[max_shift_period:]

        for col in cols_to_drop + ['macd_signal']:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)

    def move_training_labels_to_end(self, label: str):
        column = self.df.pop(label) 
        self.df[label] = column

    def render(self, except_cols = []):
        print(f"{'Feature':25} {'Not Zero':10} {'Min'.center(15)} {'Max'.center(15)} {'Mean'.center(15)} >0   [0-1]")
        print('-'*250)
        for col in self.df.columns.to_list():
            if col in except_cols:
                continue
            slice_df = self.df[col].dropna()
            not_zero = slice_df[slice_df != 0].shape[0]
            _min = slice_df.min()
            _max = slice_df.max()
            _mean = slice_df.mean()

            only_plus = False if _min < 0 else True
            in_range = True if _min >= 0 and _max <= 1 else False
            print(f"{col:25} {not_zero:7} {_min:15.4f} {_max:15.4f} {_mean:15.4f}   {only_plus}    {in_range}")

    def save_csv(self, name: str, cols_to_drop = []):
        self._check_diff_time()
        if cols_to_drop:
            self.df.dropna(how='any').drop(columns=cols_to_drop).to_csv(f'{self.folder}/{name.replace(".csv", "")}.csv', index=False)
        else:
            self.df.dropna(how='any').to_csv(f'{self.folder}/{name.replace(".csv", "")}.csv', index=False)