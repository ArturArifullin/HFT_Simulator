import numpy as np
import pandas as pd
from typing import List, Union

from simulator import update_best_positions, MdUpdate, OwnTrade


def get_pnl(updates_list:List[ Union[MdUpdate, OwnTrade] ]) -> pd.DataFrame:
    '''
        This function calculates PnL from list of updates
    '''
    btc_pos, usd_pos = 0.0, 0.0
    
    worth = 0.0
    
    worth_list = []
    btc_pos_list = []
    usd_pos_list = []
    mid_price_list = []
    
    best_bid = -np.inf
    best_ask = np.inf

    for update in updates_list:
        
        if isinstance(update, MdUpdate):
            best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
        
        mid_price = 0.5 * ( best_ask + best_bid )
        
        if isinstance(update, OwnTrade):
            trade = update    
            #update positions
            if trade.side == 'BID':
                btc_pos += trade.size
                usd_pos -= trade.price * trade.size
            elif trade.side == 'ASK':
                btc_pos -= trade.size
                usd_pos += trade.price * trade.size
        
        worth = usd_pos + mid_price * btc_pos
        worth_list.append(worth)
        btc_pos_list.append(btc_pos)
        usd_pos_list.append(usd_pos)
        mid_price_list.append(mid_price)
    receive_ts = [update.receive_ts for update in updates_list]
    exchange_ts = [update.exchange_ts for update in updates_list]
    df = pd.DataFrame({"exchange_ts": exchange_ts, "receive_ts":receive_ts, "total":worth_list, "BTC":btc_pos_list, 
                       "USD":usd_pos_list, "mid_price":mid_price_list})
    return df


def trade_to_dataframe(trades_list:List[OwnTrade]) -> pd.DataFrame:
    exchange_ts = [ trade.exchange_ts for trade in trades_list ]
    receive_ts = [ trade.receive_ts for trade in trades_list ]
    
    size = [ trade.size for trade in trades_list ]
    price = [ trade.price for trade in trades_list ]
    side  = [trade.side for trade in trades_list ]
    
    dct = {
        "exchange_ts" : exchange_ts,
        "receive_ts"  : receive_ts,
         "size" : size,
        "price" : price,
        "side"  : side
    }
    
    df = pd.DataFrame(dct)
    
    return df


def md_to_dataframe(md_list: List[MdUpdate]) -> pd.DataFrame:
    
    best_bid = -np.inf
    best_ask = np.inf
    best_bids = []
    best_asks = []
    for md in md_list:
        best_bid, best_ask = update_best_positions(best_bid, best_ask, md)
        
        best_bids.append(best_bid)
        best_asks.append(best_ask)
        
    exchange_ts = [ md.exchange_ts for md in md_list ]
    receive_ts = [ md.receive_ts for md in md_list ]
    dct = {
        "exchange_ts" : exchange_ts,
        "receive_ts"  :receive_ts,
        "bid_price" : best_bids,
        "ask_price" : best_asks
    }
    
    df = pd.DataFrame(dct)    
    return df