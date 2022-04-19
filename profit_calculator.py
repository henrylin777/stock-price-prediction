import sys
import argparse
from enum import IntEnum
from typing import List

import pandas as pd


class Action(IntEnum):
    BUY = 1
    NO_ACTION = 0
    SOLD = -1


class InvalidActionError(Exception):
    pass


class StockNumExceedError(Exception):
    pass


class InvalidActionNumError(Exception):
    pass


class StockTrader:
    def __init__(self):
        self.accumulated_profit = 0
        self.holding_price = None
        self.sell_short_price = None

    @property
    def is_holding_stock(self):
        return self.holding_price is not None

    @property
    def is_shorting_stock(self):
        return self.sell_short_price is not None

    def perform_action(self, action_code: int, stock_price: float):
        if action_code == Action.BUY:
            self.buy(stock_price)
        elif action_code == Action.SOLD:
            self.sell(stock_price)
        elif action_code == Action.NO_ACTION:
            pass
        else:
            raise InvalidActionError('Invalid Action')

    def buy(self, stock_price: float):
        if self.is_holding_stock:
            raise StockNumExceedError('You cannot buy stocks when you hold one')
        elif self.is_shorting_stock:
            self.accumulated_profit += self.sell_short_price - stock_price
            self.sell_short_price = None
        else:
            self.holding_price = stock_price

    def sell(self, stock_price: float):
        if self.is_shorting_stock:
            raise StockNumExceedError("You cannot sell short stocks when you've already sell short one")
        elif self.is_holding_stock:
            self.accumulated_profit += stock_price - self.holding_price
            self.holding_price = None
        else:
            self.sell_short_price = stock_price


def check_stock_actions_length(stocks_df: pd.DataFrame, actions: List[int]) -> bool:
    if len(stocks_df) != (len(actions) + 1):
        return False
    return True


def calculate_profit(stocks_df: pd.DataFrame, actions: List[int]) -> float:
    stock_trader = StockTrader()

    stock = None
    stocks_df.drop(0, inplace=True)
    for (_, stock), action in zip(stocks_df.iterrows(), actions):
        stock_trader.perform_action(action, stock['open'])

    if stock is not None:
        if stock_trader.is_holding_stock:
            stock_trader.sell(stock['close'])
        elif stock_trader.is_shorting_stock:
            stock_trader.buy(stock['close'])

    return stock_trader.accumulated_profit



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('stock',
                        help='input stock file name')
    parser.add_argument('action',
                        help='input action file name')
    args = parser.parse_args()

    # Load stock data
    FEATURE_NAMES = ('open', 'high', 'low', 'close')
    stocks_df = pd.read_csv(args.stock, names=FEATURE_NAMES)

    # Load actions
    with open(args.action, 'r') as action_file:
        actions = list()
        for line in action_file.readlines():
            action = int(line.strip())
            actions.append(action)

    if not check_stock_actions_length(stocks_df, actions):
        raise InvalidActionNumError('Invalid number of actions')

    profit = calculate_profit(stocks_df, actions)
    print(profit)
