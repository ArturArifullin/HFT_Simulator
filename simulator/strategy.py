from cmath import log
import math
from turtle import up
from typing import List, Optional, Tuple, Union, Dict
import scipy.stats as sps
import numpy as np
import pandas as pd

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions

class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, 
                 gamma:float = 0.001, lookback:int = 2000) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        
        self.gamma = gamma
        self.lookback = lookback
        #self.alpha = alpha
        #self.K = K


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        price_bid = -np.inf
        price_ask = np.inf
        
        #Standart dev 
        sd = None 
        
        #Alpha coefficient
        alpha = None
        
        #coefficient K 
        K = None
        
        #reserve price
        r = None
        
        #spread for reserve price
        spread = None
        
        
        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    #price_bid, price_ask = update_best_positions(price_bid, price_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'
            
            #If we have not enought data 
            if ( len(md_list)  < self.lookback ):
                continue 
            
                
            lookback_md = md_list[-self.lookback:]
            list_of_mid_prices = []
            first_iteration = True 
            exch_ts = 0 #temp variable of timestampx
            size_list = []
            delta_p_list = []
            temp_price_list_asks = []
            temp_price_list_bids = []
            sum_of_bids = 0
            sum_of_asks = 0 
            for md in lookback_md:
                orderbook = md.orderbook
                #print(orderbook)
                #print(orderbook)
                if ( orderbook is not None ):
                    mid_price = (orderbook.asks[0][0] + orderbook.bids[0][0])/2
                    list_of_mid_prices.append(mid_price)
                
                anontrade = md.trade
                #print(type(anontrade))
                if ( anontrade is not None):
                    if first_iteration:
                        exch_ts = anontrade.exchange_ts
                        first_iteration = False    
                        
                    if (anontrade.exchange_ts != exch_ts):  
                        if ( sum_of_asks > 0 ):
                            size_list.append(sum_of_asks)
                            delta_p_list.append(abs(max(temp_price_list_asks)-
                                                    min(temp_price_list_asks)))
                        if ( sum_of_bids > 0 ):
                            size_list.append(sum_of_bids)
                            delta_p_list.append(abs(max(temp_price_list_bids)-
                                                    min(temp_price_list_bids)))
                        temp_price_list_asks = []
                        temp_price_list_bids = []
                        sum_of_bids = 0
                        sum_of_asks = 0 
                        exch_ts = anontrade.exchange_ts
                    
                    if (anontrade.side == 'ASK'):
                        sum_of_asks += anontrade.size
                        temp_price_list_asks.append(anontrade.price)
                            
                    if (anontrade.side == 'BID'):
                        sum_of_bids += anontrade.size
                        temp_price_list_bids.append(anontrade.price)

                
            sd = np.array(list_of_mid_prices).std() 
            q = 0
            for trade in trades_list:
                if ( trade.side == 'BID' ):
                    q += trade.size
                else:
                    q -= trade.size
            
            #Reserve price
            r = list_of_mid_prices[-1] - q * self.gamma * sd ** 2
            '''
            print('_______________')
            print(list_of_mid_prices)
            print(size_list)
            print('_______________')
            '''
            
            #Finding alpha 
            if  (len(size_list) > 0):
                sizes = np.array(size_list)
                y, x = np.histogram(sizes, bins = len(size_list))
                x = (x[1:]+x[:-1])/2
                x = x[y>0]
                y = y[y>0]
                res = sps.linregress(np.log(x), np.log(y)) #delete nulls
                alpha = -1 - res.slope
                
                #Findind K
                res = sps.linregress( np.log(sizes), np.array(delta_p_list))
                K = 1 / res.slope
                
                #Solving spread 
                k = K * alpha
                spread = self.gamma * sd ** 2 + ( 2 / self.gamma ) * log( 1 + self.gamma / k ) 
                ''' 
                print('_______________')
                print(spread, alpha, K, r)
                print('_______________')
                '''
                #Making new price of orders
                price_ask = r + spread / 2
                price_bid = r - spread / 2
                            
                if ( price_ask is not None and price_bid is not None):
                    if receive_ts - prev_time >= self.delay:
                        prev_time = receive_ts
                        #place order
                        bid_order = sim.place_order( receive_ts, 0.001, 'BID', price_bid )
                        ask_order = sim.place_order( receive_ts, 0.001, 'ASK', price_ask )
                        ongoing_orders[bid_order.order_id] = bid_order
                        ongoing_orders[ask_order.order_id] = ask_order

                        all_orders += [bid_order, ask_order]
                
                
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders

class ReserverveStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, param_1:float = 1.69 , param_2: float = -1.2) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.param_1 = param_1
        self.param_2 = param_2
        self.q = 0 
        #self.alpha = alpha
        #self.K = K


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        price_bid = -np.inf
        price_ask = np.inf
        
        
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        
        ask_list = []
        bid_list = []
        r_list = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            
            current_mid_price = None
            q = 0
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    #price_bid, price_ask = update_best_positions(price_bid, price_ask, update)
                    md_list.append(update)
                    orderbook = update.orderbook
                    if (orderbook is not None):
                        current_mid_price = (orderbook.asks[0][0] + orderbook.bids[0][0])/2
                    
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    if ( update.side == 'BID' ):
                        q += update.size
                    else:
                        q -= update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                    
                
                else: 
                    assert False, 'invalid type of update!'
             
            self.q += q
            
            if ( current_mid_price is not None):
                
                reservation_price = current_mid_price - self.q * self.param_1
                spread = self.param_1 + self.param_2
                
                price_bid = reservation_price - spread / 2
                price_ask = reservation_price + spread / 2
                
                if receive_ts - prev_time >= self.delay:
                        prev_time = receive_ts
                        #place order
                        bid_order = sim.place_order( receive_ts, 0.001, 'BID', price_bid )
                        ask_order = sim.place_order( receive_ts, 0.001, 'ASK', price_ask )
                        
                        ongoing_orders[bid_order.order_id] = bid_order
                        ongoing_orders[ask_order.order_id] = ask_order

                        all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
            ask_list.append(price_ask)
            bid_list.append(price_bid)
            r_list.append(reservation_price)
                
        return trades_list, md_list, updates_list, all_orders, ask_list, bid_list, r_list

class MidPriceSlowStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, 
                 gamma:float = 0.001, lookback:int = 2000) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        
        self.gamma = gamma
        self.lookback = lookback
        #self.alpha = alpha
        #self.K = K


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        price_bid = -np.inf
        price_ask = np.inf
        
        #Standart dev 
        sd = None 
        
        #Alpha coefficient
        alpha = None
        
        #coefficient K 
        K = None
        
        #reserve price
        r = None
        
        #spread for reserve price
        spread = None
        
        
        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    #price_bid, price_ask = update_best_positions(price_bid, price_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'
            
            #If we have not enought data 
            if ( len(md_list)  < self.lookback ):
                continue 
            
                
            lookback_md = md_list[-self.lookback:]
            list_of_mid_prices = []
            first_iteration = True 
            exch_ts = 0 #temp variable of timestampx
            size_list = []
            delta_p_list = []
            temp_price_list_asks = []
            temp_price_list_bids = []
            sum_of_bids = 0
            sum_of_asks = 0 
            for md in lookback_md:
                orderbook = md.orderbook
                #print(orderbook)
                #print(orderbook)
                if ( orderbook is not None ):
                    mid_price = (orderbook.asks[0][0] + orderbook.bids[0][0])/2
                    list_of_mid_prices.append(mid_price)
                
                anontrade = md.trade
                #print(type(anontrade))
                if ( anontrade is not None):
                    if first_iteration:
                        exch_ts = anontrade.exchange_ts
                        first_iteration = False    
                        
                    if (anontrade.exchange_ts != exch_ts):  
                        if ( sum_of_asks > 0 ):
                            size_list.append(sum_of_asks)
                            delta_p_list.append(abs(max(temp_price_list_asks)-
                                                    min(temp_price_list_asks)))
                        if ( sum_of_bids > 0 ):
                            size_list.append(sum_of_bids)
                            delta_p_list.append(abs(max(temp_price_list_bids)-
                                                    min(temp_price_list_bids)))
                        temp_price_list_asks = []
                        temp_price_list_bids = []
                        sum_of_bids = 0
                        sum_of_asks = 0 
                        exch_ts = anontrade.exchange_ts
                    
                    if (anontrade.side == 'ASK'):
                        sum_of_asks += anontrade.size
                        temp_price_list_asks.append(anontrade.price)
                            
                    if (anontrade.side == 'BID'):
                        sum_of_bids += anontrade.size
                        temp_price_list_bids.append(anontrade.price)

                
            sd = np.array(list_of_mid_prices).std() 
            q = 0
            for trade in trades_list:
                if ( trade.side == 'BID' ):
                    q += trade.size
                else:
                    q -= trade.size
            
            #Reserve price
            r = list_of_mid_prices[-1] #I simply deleted element of formula :)
            '''
            print('_______________')
            print(list_of_mid_prices)
            print(size_list)
            print('_______________')
            '''
            
            #Finding alpha 
            if  (len(size_list) > 0):
                sizes = np.array(size_list)
                y, x = np.histogram(sizes, bins = len(size_list))
                x = (x[1:]+x[:-1])/2
                x = x[y>0]
                y = y[y>0]
                res = sps.linregress(np.log(x), np.log(y)) #delete nulls
                alpha = -1 - res.slope
                
                #Findind K
                res = sps.linregress( np.log(sizes), np.array(delta_p_list))
                K = 1 / res.slope
                
                #Solving spread 
                k = K * alpha
                spread = self.gamma * sd ** 2 + ( 2 / self.gamma ) * log( 1 + self.gamma / k ) 
                ''' 
                print('_______________')
                print(spread, alpha, K, r)
                print('_______________')
                '''
                #Making new price of orders
                price_ask = r + spread / 2
                price_bid = r - spread / 2
                            
                if ( price_ask is not None and price_bid is not None):
                    if receive_ts - prev_time >= self.delay:
                        prev_time = receive_ts
                        #place order
                        bid_order = sim.place_order( receive_ts, 0.001, 'BID', price_bid )
                        ask_order = sim.place_order( receive_ts, 0.001, 'ASK', price_ask )
                        ongoing_orders[bid_order.order_id] = bid_order
                        ongoing_orders[ask_order.order_id] = ask_order

                        all_orders += [bid_order, ask_order]
                
                
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders

class MidPriceFastStrategy: 

    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, param_1:float = 1.69 , param_2: float = -1.2) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.param_1 = param_1
        self.param_2 = param_2
        self.q = 0 
        #self.alpha = alpha
        #self.K = K


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        price_bid = -np.inf
        price_ask = np.inf
        
        
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        
        ask_list = []
        bid_list = []
        
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            
            current_mid_price = None
            q = 0
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    #price_bid, price_ask = update_best_positions(price_bid, price_ask, update)
                    md_list.append(update)
                    orderbook = update.orderbook
                    if (orderbook is not None):
                        current_mid_price = (orderbook.asks[0][0] + orderbook.bids[0][0])/2
                    
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    if ( update.side == 'BID' ):
                        q += update.size
                    else:
                        q -= update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                    
                
                else: 
                    assert False, 'invalid type of update!'
             
            self.q += q
            
            if ( current_mid_price is not None):
                
                reservation_price = current_mid_price # Changes in here!!!
                spread = self.param_1 + self.param_2
                
                price_bid = reservation_price - spread / 2
                price_ask = reservation_price + spread / 2
                
                if receive_ts - prev_time >= self.delay:
                        prev_time = receive_ts
                        #place order
                        bid_order = sim.place_order( receive_ts, 0.001, 'BID', price_bid )
                        ask_order = sim.place_order( receive_ts, 0.001, 'ASK', price_ask )
                        
                        ongoing_orders[bid_order.order_id] = bid_order
                        ongoing_orders[ask_order.order_id] = ask_order

                        all_orders += [bid_order, ask_order]
                        
            if( price_ask == np.inf and price_bid == -np.inf): 
                ask_list.append(0)
                bid_list.append(0)
            else:         
                ask_list.append(price_ask)
                bid_list.append(price_bid)
            

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders, ask_list, bid_list
    
    
class FuturePriceStrategy: 

    def __init__(self, delay: float, hold_time:Optional[float] = None, param_1:float = 1.69 , param_2: float = -1.2, 
                 future_price_data = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.param_1 = param_1
        self.param_2 = param_2
        self.q = 0 
        self.future_price_data = future_price_data
        #self.alpha = alpha
        #self.K = K
    
    def get_future_md(self, ts):
        ans = None
        last = None
        for i in range (len(self.future_price_data)):
            if ( self.future_price_data[i].exchange_ts > ts and self.future_price_data[i].orderbook is not None ):
                ans = (self.future_price_data[i].orderbook.asks[0][0] +
                       self.future_price_data[i].orderbook.bids[0][0])/2
                last = i
                break
            
        self.future_price_data = self.future_price_data[last:]
        return ans


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        price_bid = -np.inf
        price_ask = np.inf
        
        
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        
        ask_list = []
        bid_list = []
        
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            
            current_mid_price = None
            q = 0
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    #price_bid, price_ask = update_best_positions(price_bid, price_ask, update)
                    md_list.append(update)
                    orderbook = update.orderbook
                    if (orderbook is not None):
                        current_mid_price = (orderbook.asks[0][0] + orderbook.bids[0][0])/2
                    
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    if ( update.side == 'BID' ):
                        q += update.size
                    else:
                        q -= update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                    
                
                else: 
                    assert False, 'invalid type of update!'
             
            self.q += q
            
            future_mid_price = self.get_future_md(update.exchange_ts)
            if (future_mid_price is None):
                break
            ask_order = None
            bid_order = None
            
            if ( current_mid_price is not None):
                
                spread = self.param_1 + self.param_2
                price_ask = current_mid_price + spread/2
                price_bid = current_mid_price - spread/2
                
                if ( future_mid_price - current_mid_price >= 0 ):      
                      if receive_ts - prev_time >= self.delay:
                        prev_time = receive_ts
                        #place order
                        bid_order = sim.place_order( receive_ts, 0.001, 'BID', price_bid )              
                        ongoing_orders[bid_order.order_id] = bid_order
                        all_orders += [bid_order]

                if ( future_mid_price - current_mid_price <= 0 ):      
                      if receive_ts - prev_time >= self.delay:
                        prev_time = receive_ts
                        #place order
                        ask_order = sim.place_order( receive_ts, 0.001, 'ASK', price_ask )              
                        ongoing_orders[ask_order.order_id] = ask_order
                        all_orders += [ask_order]

                        
            if( price_ask == np.inf and price_bid == -np.inf): 
                ask_list.append(0)
                bid_list.append(0)
            elif ( ask_order is None ):
                ask_list.append(np.nan)
            elif ( bid_order is None ):
                bid_list.append(np.nan)
            else:         
                ask_list.append(price_ask)
                bid_list.append(price_bid)
            

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders, ask_list, bid_list