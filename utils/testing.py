import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from typing import Union

from utils.sequence import SequenceData
from utils.network import LSTMNetwork, LSTMNetworkWithMask
from utils.trade import TradesManager
from utils.func import calc_time_decor, calc_time, log_memory_usage

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        # logging.FileHandler('logs/testing.log'),  
        logging.StreamHandler()                    
    ]
)
logger = logging.getLogger(__name__)

class Testing:
    def __init__(self, 
        df: pd.DataFrame, 
        direction: str,
        sequence_length: int,
        check_period: int,
        open_model_path: str, 
        close_model_path: str, 
        trading_fee = 0.0005, 
        stop_loss_pct = 1.0, 
        take_profit_pct = 1.0,
        leverage = 1,
        max_loss = 1.0
    ):
        self.device = torch.device('cuda')
        SEED = 42
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.df = df
        self.direction = direction
        self.sequence_length = sequence_length
        self.check_period = check_period
        self.prices = df['close'].iloc[self.sequence_length - 1:].values
        self._fill_seq_data()

        self.trading_fee = trading_fee
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.leverage = leverage
        self.manager = TradesManager(
            trading_fee=self.trading_fee, 
            stop_loss=self.stop_loss_pct, 
            take_profit=self.take_profit_pct,
            leverage=self.leverage,
            max_loss=max_loss
        )

        self.open_model = Testing.load_model(open_model_path)
        self.close_model = Testing.load_model(close_model_path)

    def _fill_seq_data(self):
        data_test = self.df.drop(columns=['close', 'open', 'low', 'high', 'open_time', 'rewards']).values.astype(np.float32)
        self.seq_data = SequenceData(self.sequence_length)
        self.seq_data.fill(data_test)

    @torch.no_grad()
    @staticmethod
    def binary_predict(model, input_data: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        logits = model(input_data)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        
        confidence = torch.where(
            predictions == 0,
            (0.5 - probabilities) * 2,
            (probabilities - 0.5) * 2
        )
        return predictions.cpu().numpy().flatten(), confidence.cpu().numpy().flatten() 

    @staticmethod
    def calc_max_profit(
        prices: np.ndarray,
        stop_loss_pct: float,
        direction: str,
        current_step: int,
        check_period: int,
        price = None
    ) -> tuple[float, int]:
        
        current_price = prices[current_step]
        if price is not None:
            assert current_price == price
            
        max_len = min(current_step + check_period + 1, len(prices))
        prices_slice = prices[current_step + 1:max_len]
        if len(prices_slice) == 0:
            return 0.0, 0
        
        if direction == 'long':
            profit = (prices_slice - current_price) / current_price
        elif direction == 'short':
            profit = (current_price - prices_slice) / current_price
        else:
            raise Exception('Direction incorrected')

        # stop_loss_pct = stop_loss_pct if stop_loss_pct < 0 else -stop_loss_pct
        # if np.min(profit) <= stop_loss_pct:
        #     max_profit = stop_loss_pct
        #     max_profit_index = 0
        # else:
        max_profit = float(np.max(profit))
        max_profit_index = int(np.argmax(profit))
        return max_profit, max_profit_index + 1

    @torch.no_grad()
    def test_on_trades(self, 
        open_pred: np.ndarray,
        open_prob: np.ndarray,
        close_pred: np.ndarray,
        close_prob: np.ndarray,
        open_threshold: float,
        close_threshold: float,
        is_check_buy_when_close: bool
    ):
        self.manager.clear()
        for idx, price in enumerate(self.prices):
            if self.manager.is_pnl_larger_max_loss():
                break

            if not self.manager.is_active_trade():
                pred, prob = open_pred[idx], open_prob[idx]
                is_sell = True if close_pred[idx] == 1 and close_prob[idx] >= close_threshold else False 

                if pred == 1 and not is_sell:
                    if prob >= open_threshold:
                        self.manager.open_trade(price, self.direction)
                        continue
                self.manager._update_pnl_history([0])
            else:
                pred, prob = close_pred[idx], close_prob[idx]
                if is_check_buy_when_close:
                    is_buy = True if open_pred[idx] == 1 and open_prob[idx] >= open_threshold else False 
                else:
                    is_buy = False

                if pred == 1 and not is_buy:
                    if prob >= close_threshold:
                        self.manager.close_trade(price)
                        continue

                self.manager.hold_trade(price)

        if self.manager.is_active_trade():
            self.manager.close_trade(self.prices[-1])
            self.manager.pnl_history.pop()

        stats = self.manager.get_stats_from_all_trades()
        pnls = stats['pnls']
        draw_downs = stats['draw_downs']
        durations = stats['durations']
        closed_stop_loss_count = stats['closed_stop_loss_count']
        closed_take_profit_count = stats['closed_take_profit_count']

        if pnls.shape[0] == 0:
            logger.debug('Zero trades')
            return None
        else:
            pnls_sum = pnls.sum()
            trades = pnls.shape[0]
            win_trades = pnls[pnls > 0].shape[0]
            return {
                "Trades": trades,
                "Win Trades": win_trades,
                "Winrate": win_trades/trades,
                "Stop Loss": closed_stop_loss_count,
                "Max loss": self.manager.is_pnl_larger_max_loss(),
                "Take Profit": closed_take_profit_count,
                "PNL": pnls_sum,
                "PNL (COMS)": pnls_sum + (trades * (self.trading_fee * 2) * self.leverage),
                "PNL Min": pnls.min(),
                "PNL Max": pnls.max(),
                "PNL Mean": pnls.mean(),
                "Drawdown Min": draw_downs.min(),
                "Drawdown Mean": draw_downs.mean(),
                "Duration Min": durations.min(),
                "Duration Max": durations.max(),
                "Duration Mean": durations.mean()
            }

    @torch.no_grad()
    def test_on_trades_with_simulation(self,
        is_open_simulation: bool,
        min_pnl_for_open: float,
        preds: np.ndarray,
        probs: np.ndarray,
        threshold: float
    ):
        self.manager.clear()
        close_idx = 0

        for idx, price in enumerate(self.prices):
            if idx + self.check_period == len(self.prices):
                break

            if not self.manager.is_active_trade():
                # if open simulation == True
                if is_open_simulation:
                    max_profit, _ = Testing.calc_max_profit(self.prices, self.stop_loss_pct, self.direction, idx, self.check_period, price)
                    if max_profit < min_pnl_for_open:
                        continue
                    self.manager.open_trade(price, self.direction)
                    continue
                # if open simulation == False
                pred, prob = preds[idx], probs[idx]
                if pred == 1:
                    if prob >= threshold:
                        self.manager.open_trade(price, self.direction)
                        _, close_idx = Testing.calc_max_profit(self.prices, self.stop_loss_pct, self.direction, idx, self.check_period, price)
                        close_idx += idx
                        continue
            else:
                # if open simulation == False
                if not is_open_simulation:
                    if idx - close_idx == 0:
                        close_idx = 0
                        self.manager.close_trade(price)
                        continue
                else:
                    # if open simulation == True
                    pred, prob = preds[idx], probs[idx]
                    if pred == 1:
                        if prob >= threshold:
                            self.manager.close_trade(price)
                            continue

                self.manager.hold_trade(price)

        if self.manager.is_active_trade():
            self.manager.close_trade(self.prices[-1])
            self.manager.pnl_history.pop()

        stats = self.manager.get_stats_from_all_trades()
        pnls = stats['pnls']
        draw_downs = stats['draw_downs']
        durations = stats['durations']
        closed_stop_loss_count = stats['closed_stop_loss_count']
        closed_take_profit_count = stats['closed_take_profit_count']

        if pnls.shape[0] == 0:
            logger.debug('Zero trades')
            return None
        else:
            pnls_sum = pnls.sum()
            trades = pnls.shape[0]
            win_trades = pnls[pnls > 0].shape[0]
            return {
                "Trades": trades,
                "Win Trades": win_trades,
                "Winrate": win_trades/trades,
                "Stop Loss": closed_stop_loss_count,
                "Max loss": self.manager.is_pnl_larger_max_loss(),
                "Take Profit": closed_take_profit_count,
                "PNL": pnls_sum,
                "PNL (COMS)": pnls_sum + (trades * (self.trading_fee * 2) * self.leverage),
                "PNL Min": pnls.min(),
                "PNL Max": pnls.max(),
                "PNL Mean": pnls.mean(),
                "Drawdown Min": draw_downs.min(),
                "Drawdown Mean": draw_downs.mean(),
                "Duration Min": durations.min(),
                "Duration Max": durations.max(),
                "Duration Mean": durations.mean()
            }

    @torch.no_grad()
    def plot_pnl_graphic(self,
        open_threshold: float,
        close_threshold: float,
        is_check_buy: bool,
    ):
        
        stats = self.simple_testing(open_threshold, close_threshold, is_check_buy)
        if stats is not None:
            time_indices = pd.to_datetime(self.df['open_time'].iloc[self.sequence_length - 1:].values).values
            pnl_history = self.manager.get_pnl_history() * 100
            plt.figure(figsize=(10, 6))
            for i in range(len(pnl_history) - 1):
                if pnl_history[i + 1] > pnl_history[i]:
                    plt.plot([time_indices[i], time_indices[i+1]], [pnl_history[i], pnl_history[i+1]], 'g')
                elif pnl_history[i + 1] < pnl_history[i]:
                    plt.plot([time_indices[i], time_indices[i+1]], [pnl_history[i], pnl_history[i+1]], 'r')
                else:
                    plt.plot([time_indices[i], time_indices[i+1]], [pnl_history[i], pnl_history[i+1]], 'black')

            plt.grid(True)
            plt.ylabel('PNL %')
            plt.show()

    # @calc_time_decor
    @torch.no_grad()
    def simple_testing(self,
        open_threshold: float,
        close_threshold: float,
        is_check_buy_when_close: bool,
        open_pred: np.ndarray = None,
        open_prob: np.ndarray = None,
        close_pred: np.ndarray = None,
        close_prob: np.ndarray = None
    ):

        # if open_pred is None or open_prob is None:
        #     open_pred, open_prob = Testing.binary_predict(self.open_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
        # if close_pred is None or close_prob is None:
        #     close_pred, close_prob = Testing.binary_predict(self.close_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
        
        stats = self.test_on_trades(
            open_pred=open_pred,
            open_prob=open_prob,
            close_pred=close_pred,
            close_prob=close_prob,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            is_check_buy_when_close=is_check_buy_when_close
        )

        if stats:
            # result_df = pd.DataFrame([stats])
            # return result_df
            return stats

    @torch.no_grad()
    def testing_threshold(self,
        threshold_step: float,
        min_open_threshold: float = None,
        max_open_threshold: float = None,
        min_close_threshold: float = None,
        max_close_threshold: float = None,
        open_pred: np.ndarray = None,
        open_prob: np.ndarray = None,
        close_pred: np.ndarray = None,
        close_prob: np.ndarray = None,
        is_save = True
    ):
        if min_open_threshold is not None and max_open_threshold is not None:
            open_thresholds = np.arange(min_open_threshold, max_open_threshold, threshold_step)
        else:
            open_thresholds = [0.0]
        if min_close_threshold is not None and max_close_threshold is not None:
            close_thresholds = np.arange(min_close_threshold, max_close_threshold, threshold_step)
        else:
            close_thresholds = [0.0]

        logger.debug(f'Open Thresholds: {open_thresholds}')        
        logger.debug(f'Close Thresholds: {close_thresholds}')

        self.open_model.eval()
        self.close_model.eval()
        if open_pred is None or open_prob is None:
            open_pred, open_prob = Testing.binary_predict(self.open_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
        if close_pred is None or close_prob is None:
            close_pred, close_prob = Testing.binary_predict(self.close_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))

        # result_df = pd.DataFrame(columns=["Trades", "Win Trades", "Winrate", "Stop Loss", "PNL", "PNL (COMS)", \
        #                                   "PNL Min", "PNL Max", "PNL Mean", "Drawdown Min", "Drawdown Mean", \
        #                                   "Duration Min", "Duration Max", "Duration Mean", 'Check Buy', 'Open Thr',\
        #                                   'Close Thr'])
        # df_idx = 0
        result = []
        is_check_buy_vars = [True, False]
        for is_check_buy in is_check_buy_vars:
            for o_thr in open_thresholds:
                for c_thr in close_thresholds:
                    stats = self.test_on_trades(
                        open_pred=open_pred,
                        open_prob=open_prob,
                        close_pred=close_pred,
                        close_prob=close_prob,
                        open_threshold=o_thr,
                        close_threshold=c_thr,
                        is_check_buy_when_close=is_check_buy
                    )
                    if stats:
                        stats['Check Buy'] = is_check_buy
                        stats['Open Thr'] = o_thr
                        stats['Close Thr'] = c_thr
                        # result_df.loc[df_idx] = stats
                        result.append(stats)
                        # df_idx += 1
            logger.debug(f"Open Thr: {o_thr}, Close Thr: {c_thr}, Check Buy: {is_check_buy}")
        result_df = pd.DataFrame(result)
        if is_save:
            result_df.to_csv('logs/thresholds.csv', index=False)
        return result_df
        
    @torch.no_grad()
    def testing_stop_loss(self,
        min_stop_loss: float,
        max_stop_loss: float,
        stop_loss_step: float,
        open_threshold: float,
        close_threshold: float,
        is_check_buy: bool,
        open_pred: np.ndarray = None,
        open_prob: np.ndarray = None,
        close_pred: np.ndarray = None,
        close_prob: np.ndarray = None,
        is_save = True
    ):
        stop_losses = np.arange(min_stop_loss, max_stop_loss, stop_loss_step)

        self.open_model.eval()
        self.close_model.eval()
        if open_pred is None or open_prob is None:
            open_pred, open_prob = Testing.binary_predict(self.open_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
        if close_pred is None or close_prob is None:
            close_pred, close_prob = Testing.binary_predict(self.close_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))

        # result_df = pd.DataFrame(columns=["Trades", "Win Trades", "Winrate", "Stop Loss", "PNL", "PNL (COMS)", "PNL Min", "PNL Max", "PNL Mean", "Drawdown Min", "Drawdown Mean", "Duration Min", "Duration Max", "Duration Mean", 'Stop Loss Thr'])
        # df_idx = 0
        result = []
        for stop_loss in stop_losses:
            self.manager.stop_loss = -stop_loss
            stats = self.test_on_trades(
                open_pred=open_pred,
                open_prob=open_prob,
                close_pred=close_pred,
                close_prob=close_prob,
                open_threshold=open_threshold,
                close_threshold=close_threshold,
                is_check_buy_when_close=is_check_buy
            )
            if stats:
                stats['Stop Loss Thr'] = stop_loss
                result.append(stats)
                # result_df.loc[df_idx] = stats
                # df_idx += 1
            logger.debug(f"Stop loss: {stop_loss}")

        result_df = pd.DataFrame(result)
        if is_save:
            result_df.to_csv('logs/stop_loss.csv', index=False)
        return result_df

    @torch.no_grad()
    def testing_features(self,
        input_features: list[str],
        min_len_combination: int,
        max_len_combination: int,
        is_open_model: bool,
        open_threshold: float,
        close_threshold: float,
        is_check_buy: bool,
        disable_features: list[str] = None,
        is_disable_both_models = None,
        is_save = True
    ):
        
        def get_combinations(features: list[str], min_len: int, max_len: int) -> list[list[str]]:
            result = []
            for i in range(min_len, max_len + 1):
                result.extend(list(combinations(features, i)))
            return result
        
        def convert_to_masked_model(model: LSTMNetwork) -> LSTMNetworkWithMask:
            masked_model = LSTMNetworkWithMask(
                input_size=model.input_size,
                lstm_hidden_sizes=model.lstm_hidden_sizes,
                fc_sizes=model.fc_sizes,
                dropout_rate=model.dropout_rate,
                lstm_layers=model.lstm_layers_count,
                is_bidirectional=model.is_bidirectional,
                device=model.device
            )
            masked_model.load_state_dict(model.state_dict())
            return masked_model

        open_model = convert_to_masked_model(self.open_model)
        close_model = convert_to_masked_model(self.close_model)
        open_model.eval()
        close_model.eval()

        if disable_features is None:
            disable_feature_combinations = get_combinations(input_features, min_len_combination, max_len_combination)
        else:
            disable_feature_combinations = [disable_features]
        feature_map = {name: idx for idx, name in enumerate(input_features)}

        # result_df = pd.DataFrame(columns=["Trades", "Win Trades", "Winrate", "Stop Loss", "PNL", "PNL (COMS)", "PNL Min", "PNL Max", "PNL Mean", "Drawdown Min", "Drawdown Mean", "Duration Min", "Duration Max", "Duration Mean", 'Dis Features'])
        # df_idx = 0
        result = []
        for idx, combo in enumerate(disable_feature_combinations):
            disabled_indices = [feature_map[feature] for feature in combo]

            if is_disable_both_models:
                open_model.set_mask(disabled_indices)
                close_model.set_mask(disabled_indices)
            elif is_open_model:
                open_model.set_mask(disabled_indices)
            else:
                close_model.set_mask(disabled_indices)

            if is_open_model:
                open_pred, open_prob = Testing.binary_predict(open_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
                close_pred, close_prob = Testing.binary_predict(close_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
            else:
                open_pred, open_prob = Testing.binary_predict(open_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
                close_pred, close_prob = Testing.binary_predict(close_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
            
            stats = self.test_on_trades(
                open_pred=open_pred,
                open_prob=open_prob,
                close_pred=close_pred,
                close_prob=close_prob,
                open_threshold=open_threshold,
                close_threshold=close_threshold,
                is_check_buy_when_close=is_check_buy
            )

            if stats:
                stats['Dis Features'] = combo
                result.append(stats)
                # result_df.loc[df_idx] = stats
                # df_idx += 1
            logger.debug(f"{idx+1}/{len(disable_feature_combinations)} {combo}")

        result_df = pd.DataFrame(stats)
        if is_save:
            result_df.to_csv('logs/disable_features.csv', index=False)

        return result_df

    @torch.no_grad()
    def testing_ansamble(self,
        another_open_model_paths: list[str],
        another_close_model_paths: list[str],
        open_threshold: float,
        close_threshold: float,
        is_check_buy_when_close: bool
    ):
        open_models = [self.open_model]
        close_models = [self.close_model]
        if another_open_model_paths:
            open_models.extend([Testing.load_model(model_path) for model_path in another_open_model_paths])
        if another_close_model_paths:
            close_models.extend([Testing.load_model(model_path) for model_path in another_close_model_paths])
        for model in open_models:
            model.eval()
        for model in close_models:
            model.eval()
    
        open_pred, open_prob = Testing.binary_predict(self.open_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
        close_pred, close_prob = Testing.binary_predict(self.close_model, torch.from_numpy(self.seq_data.get_data()).to(self.device))
        
        stats = self.test_on_trades(
            open_pred=open_pred,
            open_prob=open_prob,
            close_pred=close_pred,
            close_prob=close_prob,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            is_check_buy_when_close=is_check_buy_when_close
        )
        if stats:
            result_df = pd.DataFrame([stats])
            return result_df

    @staticmethod
    def load_model(model_path: str) -> Union[LSTMNetwork, LSTMNetworkWithMask]:
        checkpoint = torch.load(os.path.join(os.getcwd(), 'models', model_path))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model

class MultiModelTesting:
    def __init__(self,
        open_model_names: list[str],
        close_model_names: list[str],
        df: pd.DataFrame, 
        direction: str,
        sequence_length: int,
        check_period: int,
        start_episode: int = None,
        end_episode: int = None,
        open_model_path: str = None, 
        close_model_path: str = None, 
        trading_fee = 0.0005, 
        stop_loss_pct = 1.0, 
        take_profit_pct = 1.0,
        leverage = 1
    ):
        self.open_model_names = open_model_names
        self.close_model_names = close_model_names
        self.df = df
        self.direction = direction
        self.sequence_length = sequence_length
        self.check_period = check_period
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.open_model_path = open_model_path
        self.close_model_path = close_model_path
        self.trading_fee = trading_fee
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.leverage = leverage
        self.device = torch.device('cuda')

        self.open_model_paths = self._get_model_paths(open_model_names, open_model_path)
        self.close_model_paths = self._get_model_paths(close_model_names, close_model_path)

        # key -> [model_instance, pred, prob]
        self.saved_models_info = {}

        # self.tester = Testing(
        #     df=self.df,
        #     direction=self.direction,
        #     sequence_length=self.sequence_length,
        #     check_period=self.check_period,
        #     open_model_path=self.open_model_paths[0],
        #     close_model_path=self.close_model_paths[0],
        #     trading_fee=self.trading_fee,
        #     stop_loss_pct=self.stop_loss_pct,
        #     leverage=self.leverage
        # )
        # self.tester.open_model = None
        # self.tester.close_model = None

    def _get_model_paths(self, model_names: list[str], model_path: str = None):
        model_paths = []
        if model_path is None:
            models = os.listdir('models')
            for model in models:
                for model_name in model_names:
                    if not model.startswith(model_name):
                        continue

                    episode = int(model.replace('.pth', '').split('_')[-1])
                    if self.start_episode is not None:
                        if episode < self.start_episode:
                            continue
                    if self.end_episode is not None:
                        if episode > self.end_episode:
                            continue

                    model_paths.append(model)
        else:
            model_paths.append(model_path)
        return model_paths

    @calc_time_decor
    @torch.no_grad()
    def testing_models(self,
        open_threshold: float,
        close_threshold: float,
        is_check_buy_when_close: bool,
        open_model_paths: list[str] = None,
        saved_models_info: dict = None,
        close_models_paths: list[str] = None
    ):
        tester = Testing(
            df=self.df,
            direction=self.direction,
            sequence_length=self.sequence_length,
            check_period=self.check_period,
            open_model_path=self.open_model_paths[0],
            close_model_path=self.close_model_paths[0],
            trading_fee=self.trading_fee,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            leverage=self.leverage
        )
        del tester.open_model
        del tester.close_model

        if open_model_paths is None:
            open_model_paths = self.open_model_paths

        if saved_models_info is None:
            saved_models_info = self.saved_models_info

        result = []
        for open_model_path in open_model_paths:
            if open_model_path not in saved_models_info:
                open_model = self.load_model(open_model_path)
                open_pred, open_prob = Testing.binary_predict(open_model, torch.from_numpy(tester.seq_data.get_data()).to(tester.device))
                del open_model
                torch.cuda.empty_cache()
                saved_models_info[open_model_path] = [open_pred.copy(), open_prob.copy()]
            else:
                saved_model_info = saved_models_info[open_model_path]
                open_pred, open_prob = saved_model_info[0], saved_model_info[1]

            for close_model_path in close_models_paths:
                if close_model_path not in saved_models_info:
                    close_model = self.load_model(close_model_path)
                    close_pred, close_prob = Testing.binary_predict(close_model, torch.from_numpy(tester.seq_data.get_data()).to(tester.device))
                    del close_model
                    torch.cuda.empty_cache()
                    saved_models_info[close_model_path] = [close_pred.copy(), close_prob.copy()]
                else:
                    saved_model_info = saved_models_info[close_model_path]
                    close_pred, close_prob = saved_model_info[0], saved_model_info[1]

                df = tester.simple_testing(
                    open_threshold=open_threshold, 
                    close_threshold=close_threshold, 
                    is_check_buy_when_close=is_check_buy_when_close,
                    open_pred=open_pred,
                    open_prob=open_prob,
                    close_pred=close_pred,
                    close_prob=close_prob,
                )
                if df is not None:
                    df['open_model'] = open_model_path
                    df['close_model'] = close_model_path
                    result.append(df)

        if result:
            # df = pd.concat(result)
            # df = pd.DataFrame(result)
            # df.to_csv('logs/testing_models.csv', index=False)
            # return df, open_model_path, saved_models_info[open_model_path]
            return result
        else:
            # return None, open_model_path, saved_models_info[open_model_path]
            return None

    @calc_time_decor
    @torch.no_grad()
    def testing_models_treades(self,
        open_threshold: float,
        close_threshold: float,
        is_check_buy_when_close: bool,
    ):
        result = []
        manager = Manager()
        shared_models_info = manager.dict()
        # прогрев
        stats = self.testing_models(open_threshold, close_threshold, is_check_buy_when_close, [self.open_model_paths[0]], shared_models_info, self.close_model_paths)
        if stats is not None:
            result.extend(stats)


        # model_paths = [self.open_model_paths[i::len(self.open_model_names) * 200] for i in range(len(self.open_model_names) * 200)]
        print(os.cpu_count())
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(self.testing_models, open_threshold, close_threshold, is_check_buy_when_close, [open_paths], shared_models_info, self.close_model_paths): open_paths for open_paths in self.open_model_paths[1:]}
            with tqdm(total=len(futures), desc="Processing models") as pbar:
                for future in as_completed(futures):
                    try:
                        # stats, open_model_path, open_model_info = future.result()
                        # self.saved_models_info[open_model_path] = open_model_info
                        stats = future.result()
                        if stats is not None:
                            result.extend(stats)
                    except Exception as e:
                        print(e)
                    finally:
                        pbar.update(1)

        # pd.concat(result).to_csv('testing_models_treades.csv', index=False)
        pd.DataFrame(result).to_csv('logs/testing_models_treades.csv', index=False)

    def testing_models_with_simulation(self,
        model_name,
        is_open_simulation: bool,
        min_pnl_for_open: float,
        threshold: float,
    ):
        tester = Testing(
            df=self.df,
            direction=self.direction,
            sequence_length=self.sequence_length,
            check_period=self.check_period,
            open_model_path=self.open_model_paths[0],
            close_model_path=self.close_model_paths[0],
            trading_fee=self.trading_fee,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            leverage=self.leverage
        )
        del tester.open_model
        del tester.close_model

        result = []
        # if is_open_simulation:
        #     model_paths = self.close_model_paths
        # else:
        #     model_paths = self.open_model_paths
        model_paths = self._get_model_paths(model_name)
        for model_path in model_paths:
            model = self.load_model(model_path)
            if is_open_simulation:
                tester.close_model = model
            else:
                tester.open_model = model

            preds, probs = Testing.binary_predict(model, torch.from_numpy(tester.seq_data.get_data()).to(self.device))
            stats = tester.test_on_trades_with_simulation(
                is_open_simulation=is_open_simulation,
                min_pnl_for_open=min_pnl_for_open,
                preds=preds,
                probs=probs,
                threshold=threshold
            )
            if stats:
                spt = model_path.split('_')
                stats['Name'] = spt[0]
                stats['Dataset'] = spt[1]
                stats['Reward_'] = spt[2]
                stats['Episode_'] = spt[3].replace('.pth', '')
                result.append(stats)

        if result:
            df = pd.DataFrame(result)
            df.to_csv('logs/testing_models_with_simulation.csv', index=False)
            return df 
        else:
            return None

    @calc_time_decor
    def testing_all_models_with_sumilation(self):
        files = os.listdir('models')
        files = [f for f in files if f.endswith('.pth')]
        uniq_names = []
        for file in files:
            name = file.split('_')[0]
            if name not in uniq_names:
                uniq_names.append(name)

        dfs = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self.testing_models_with_simulation, name, True, 0.015, 0.3): name for name in uniq_names}
            with tqdm(total=len(futures), desc="Processing models") as pbar:
                for future in as_completed(futures):
                    try:
                        df = future.result()
                        if df is not None and df.shape[0] != 0:
                            dfs.append(df)
                    except:
                        pass
                    finally:
                        pbar.update(1)
        pd.concat(dfs, ignore_index=True).to_csv('logs/testing_all_models_with_sumilation.csv', index=False)

    @calc_time_decor
    def testing_models_threshold(self,
        threshold_step: float,
        min_open_threshold: float = None,
        max_open_threshold: float = None,
        min_close_threshold: float = None,
        max_close_threshold: float = None
    ):
        result = []
        for open_model_path in self.open_model_paths:
            open_model = self.load_model(open_model_path)
            self.tester.open_model = open_model

            for close_model_path in self.close_model_paths:
                close_model = self.load_model(close_model_path)
                self.tester.close_model = close_model

                stats = self.tester.testing_threshold(
                    threshold_step=threshold_step,
                    min_open_threshold=min_open_threshold,
                    max_open_threshold=max_open_threshold,
                    min_close_threshold=min_close_threshold,
                    max_close_threshold=max_close_threshold,
                    is_save=False
                )
                if stats:
                    result.append(stats)

        pd.DataFrame(result).to_csv('logs/testing_models_threshold.csv', index=False)

    @calc_time_decor
    def testing_models_stop_loss(self,
        min_stop_loss: float,
        max_stop_loss: float,
        stop_loss_step: float,
        open_threshold: float,
        close_threshold: float,
        is_check_buy: bool,
    ):
        result = []
        for open_model_path in self.open_model_paths:
            open_model = self.load_model(open_model_path)
            self.tester.open_model = open_model

            for close_model_path in self.close_model_paths:
                close_model = self.load_model(close_model_path)
                self.tester.close_model = close_model

                stats = self.tester.testing_stop_loss(
                    min_stop_loss=min_stop_loss,
                    max_stop_loss=max_stop_loss,
                    stop_loss_step=stop_loss_step,
                    open_threshold=open_threshold,
                    close_threshold=close_threshold,
                    is_check_buy=is_check_buy,
                    is_save=False,
                )
                if stats:
                    result.append(stats)

        pd.DataFrame(result).to_csv('logs/testing_models_stop_loss.csv', index=False)

    @calc_time_decor
    def testing_models_features(self,
        input_features: list[str],
        min_len_combination: int,
        max_len_combination: int,
        is_open_model: bool,
        open_threshold: float,
        close_threshold: float,
        is_check_buy: bool,
        disable_features: list[str] = None,
        is_disable_both_models = None,
    ):
        result = []
        for open_model_path in self.open_model_paths:
            open_model = self.load_model(open_model_path)
            self.tester.open_model = open_model

            for close_model_path in self.close_model_paths:
                close_model = self.load_model(close_model_path)
                self.tester.close_model = close_model

                stats = self.tester.testing_features(
                    input_features=input_features,
                    min_len_combination=min_len_combination,
                    max_len_combination=max_len_combination,
                    is_open_model=is_open_model,
                    open_threshold=open_threshold,
                    close_threshold=close_threshold,
                    is_check_buy=is_check_buy,
                    disable_features=disable_features,
                    is_disable_both_models=is_disable_both_models,
                    is_save = False
                )
                if stats:
                    result.append(stats)

        pd.DataFrame(result).to_csv('logs/testing_models_features.csv', index=False)

    def load_model(self, model_path: str):
        return Testing.load_model(model_path).to(self.device)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    df = pd.read_csv('datasets/BTCUSDT_processed_open_long_trend_levels_24.csv')
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df[df['open_time'] >= pd.to_datetime('2023-09-03 15:00:00')]
    # print(df.shape)

    # testing = Testing(
    #     df=df,
    #     direction='long',
    #     sequence_length=12,
    #     check_period=None,
    #     # open_model_path='1733529036_16_24_101.pth',
    #     open_model_path='1733491156_51_24_77.pth',
    #     close_model_path='1733654868_16_6_141.pth',
    #     stop_loss_pct=0.05
    # )

    # testing.plot_pnl_graphic(0.95, 0.75, True)

    multitesting = MultiModelTesting(
        open_model_names=['1733491355'],
        close_model_names=['1733649988'],
        df=df,
        direction='long',
        sequence_length=12,
        check_period=None
    ).testing_models_treades(0.3, 0.3, True)

    # open_pred, open_prob = Testing.binary_predict(testing.open_model, torch.from_numpy(testing.seq_data.get_data()).to(testing.device))
    # stats = testing.test_on_trades_with_simulation(
    #     is_open_simulation=False,
    #     min_pnl_for_open=None,
    #     preds=open_pred,
    #     probs=open_prob,
    #     threshold=0.3
    # )
    # print(stats)
    # exit(0)

    # Function: testing_threshold completed in 37.2789s
    # testing.testing_threshold(
    #     threshold_step=0.05,
    #     min_open_threshold=0.0,
    #     max_open_threshold=1.0,
    #     min_close_threshold=0.0,
    #     max_close_threshold=1.0
    # )

    # Function: testing_stop_loss completed in 1.6035s
    # testing.testing_stop_loss(
    #     stop_loss_step=0.05,
    #     min_stop_loss=0.0,
    #     max_stop_loss=1.0,
    #     open_threshold=0.95,
    #     close_threshold=0.75,
    #     is_check_buy=True,
    # )

    # testing.testing_features(
    #     input_features=df.drop(columns=['close', 'open', 'high', 'low', 'open_time', 'rewards']).columns.to_list(),
    #     is_disable_both_models=False,
    #     min_len_combination=1,
    #     max_len_combination=5,
    #     is_open_model=True,
    #     open_threshold=0.3,
    #     close_threshold=0.0,
    #     is_check_buy=True,
    #     disable_features=None # [rsi, macd]
    # )

    # 260.7273s