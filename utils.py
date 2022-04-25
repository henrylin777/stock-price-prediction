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


def calculate_profit(prev_stocks_df: pd.DataFrame, actions: List[int]) -> float:
    stock_trader = StockTrader()

    stock = None
    stocks_df = prev_stocks_df.drop(0, inplace=False)
    for (_, stock), action in zip(stocks_df.iterrows(), actions):
        stock_trader.perform_action(action, stock['open'])

    if stock is not None:
        if stock_trader.is_holding_stock:
            stock_trader.sell(stock['close'])
        elif stock_trader.is_shorting_stock:
            stock_trader.buy(stock['close'])

    return stock_trader.accumulated_profit




def trend2action(trend_list: list) -> list:
    result = []
    slot = 0
    
    for index in range(0, len(trend_list)-1):
        curr = trend_list[index]

        if curr == -2:
            if slot == 0:
                result.append(-1)
                slot = -1
            elif slot == 1:
                result.append(-1)
                slot = 0
            else:
                result.append(0)

        elif curr == -1:
            if slot == 0:
                result.append(1)
                slot = 1
            elif slot == 1:
                result.append(0)
            else:
                result.append(1)
                slot = 0

        elif curr == 0:
            result.append(0)
            slot = 0

        elif curr == 1:
            if slot == 0:
                result.append(0)
            elif slot == 1:
                result.append(-1)
                slot = 0
            else:
                result.append(0)
              

        elif curr == 2:
            if slot == 0:
                result.append(1)
                slot = 1
            elif slot == 1:
                result.append(0)
            else:
                result.append(0) 

            

    return result
