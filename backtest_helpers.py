from datetime import date
from numpy.linalg import norm
from SA_OU_helpers import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class TradingSimulator:
    def __init__(self,returns, start_capital=1000, trans_cost=0.001):
        #simulator parameters
        self.returns = returns
        self.start_capital = start_capital
        self.transaction_cost = trans_cost

        #portfolio keeps track of three things:
        self.positions = pd.DataFrame(index=returns.index, columns = returns.columns).fillna(0)
        self.trade_history = pd.DataFrame(columns=['open_time','close_time','main_asset','direction','positions', 'open_for', 'n_adj', 'pnl', 'status','descr'])
        self.transaction_cost_history = pd.DataFrame(index=returns.index, columns = ['tc']).fillna(0)

        #used internally
        self.count = 0
        self.capital = pd.DataFrame(index=returns.index, columns = ['cap'])
        self.capital_returns = pd.DataFrame(index=returns.index, columns = ['ret'])
        self.capital.iloc[0,0] = self.start_capital

    #needs to be run after all trades were done every time increment
    def time_increment(self):
        if self.count > 0:
            #calculate todays returns on yesterdays positions and subtract the transaction costs of these returns
            self.capital_returns.iloc[self.count] = self.returns.iloc[self.count].dot(self.positions.iloc[self.count-1].values.reshape(-1,1)) - self.transaction_cost_history.iloc[self.count,0]
            #todays capital
            self.capital.iloc[self.count,0] = self.capital.iloc[self.count-1,0] * (1+self.capital_returns.iloc[self.count].item())
            #update todays positions
            self.positions.iloc[self.count] += self.positions.iloc[self.count-1]
            self.open_trades_increment()
        
        self.count += 1

    def open_trade(self, asset, direction, neutral_portfolio):
        date_time = self.returns.index[self.count]

        if asset in self.get_open_positions().main_asset.values:
            print('''can't open {} because position is already open'''.format(asset))
        else:
            asset_loc = self.returns.columns.get_loc(asset)

            if direction == 'long':
                neutral_portfolio[asset_loc] += 1 #get capital allocation
            else:
                neutral_portfolio[asset_loc] += -1

            positions = neutral_portfolio #/ len(self.returns.columns) #TODO: fix this leverage adjustment
            self.positions.loc[date_time] += positions #add capital structure to existing structure

            trans_cost = norm(positions,1) * self.transaction_cost #cost incurred when opening this trade
            self.transaction_cost_history.loc[date_time, 'tc'] += trans_cost
            #save the trade
            self.trade_history = self.trade_history.append({'open_time':date_time,'close_time':None,'main_asset':asset,'direction':direction,'positions':positions, 'open_for':0, 'n_adj':1, 'pnl':-trans_cost * self.capital.iloc[self.count -1][0], 'status':True, 'descr':None},ignore_index=True)


    def close_trade(self, asset,descr):
        date_time = self.returns.index[self.count]
        if asset in self.get_open_positions().main_asset.values:
            #look for the trade
            old_trade = self.trade_history[(self.trade_history.status == True)&(self.trade_history.main_asset == asset)]

            #adjust open positions for time period
            self.positions.loc[date_time] -= old_trade.positions.values[0]
            trans_cost = norm(old_trade.positions.values[0],1) * self.transaction_cost
            self.transaction_cost_history.loc[date_time, 'tc'] += trans_cost
            
            pnl = old_trade.pnl.iloc[0]  - trans_cost * self.capital.iloc[self.count -1][0] #subtract the transaction costs for the current trade
            new_trade = {'open_time':old_trade.open_time.iloc[0],'close_time':date_time,'main_asset':asset,'direction':old_trade.direction.iloc[0],'positions':old_trade.positions.iloc[0], 'open_for':old_trade.open_for.iloc[0], 'n_adj':old_trade.n_adj.iloc[0], 'pnl':pnl, 'status':False,'descr':descr}

            self.update_trade(old_trade,new_trade) #close the transaction
        else:
            print('cant close position {} because no position is currently open'.format(asset))

    def readjust_trade(self, asset, neutral_portfolio):
        date_time = self.returns.index[self.count]
        old_trade = self.trade_history.loc[(self.trade_history.status == True)&(self.trade_history.main_asset == asset)]

        n_adj = old_trade.n_adj.iloc[0] + 1
        
        #find out what needs to change about allocation in the current trade
        asset_loc = self.returns.columns.get_loc(asset)
        direction = old_trade.direction.iloc[0]
        if direction == 'long':
            neutral_portfolio[asset_loc] += 1 #get capital allocation
        else:
            neutral_portfolio[asset_loc] += -1
        #neutral_portfolio /= len(self.returns.columns) #TODO: fix this leverage adjustment
        adjustment = neutral_portfolio - old_trade.positions[0]
        #adjust open positions for time period 
        self.positions.loc[date_time] += adjustment
        trans_cost = norm(adjustment,1) * self.transaction_cost 
        
        self.transaction_cost_history.loc[date_time, 'tc'] += trans_cost
        pnl = old_trade.pnl.iloc[0] - trans_cost * self.capital.iloc[self.count -1][0] #subtract the transaction costs for the current trade pnl

        new_trade = {'open_time':old_trade.open_time.iloc[0],'close_time':None,'main_asset':asset,'direction':old_trade.direction.iloc[0],'positions':neutral_portfolio, 'open_for':old_trade.open_for.iloc[0], 'n_adj':n_adj, 'pnl':pnl, 'status':True, 'descr':old_trade.descr.iloc[0]}
        self.update_trade(old_trade,new_trade)

    def open_trades_increment(self):
        history = self.trade_history[self.trade_history.status == True].copy()
        for idx, trade in history.iterrows():
            ret_increment = self.returns.iloc[self.count+1].dot(trade.positions) * self.capital.iloc[self.count] 
            pnl = trade.pnl + ret_increment.iloc[0]

            new_trade = {'open_time':trade.open_time,'close_time':None,'main_asset':trade.main_asset,'direction':trade.direction,'positions':trade.positions, 'open_for':trade.open_for + 1, 'n_adj':trade.n_adj, 'pnl':pnl, 'status':True,'descr':trade.descr}
            self.update_trade(trade,new_trade)

    def get_open_positions(self):
        return self.trade_history[self.trade_history.status == True]

    def get_trade(self,asset):
        return self.trade_history[(self.trade_history.status == True)&(self.trade_history.main_asset == asset)].iloc[0]
    
    def update_trade(self, old, new):
        #drop the old
        index_to_drop = self.trade_history[(self.trade_history.status == True)&(self.trade_history.main_asset == new['main_asset'])].index
        self.trade_history = self.trade_history.drop(index_to_drop)
        #update the new
        self.trade_history = self.trade_history.append(new,ignore_index=True)
    
    def get_date(self):
        return  self.returns.index[self.count]

    def is_position_open(self,asset):
        return asset in self.get_open_positions().main_asset.values


def backtest(returns,s_score_forward,R2_training_all_epochs, OU_params,betas_all_epochs,eig_portf, R2_thresh, kappa_thresh, transaction_costs,lookback, entry, exit, stoploss):
    #initialise the trade simulator
    sim = TradingSimulator(returns,1000,transaction_costs)
    previous_epoch = 0
    sl_cooldown = dict.fromkeys(returns.columns, True)
    count = -1
    for idx, row in tqdm(returns.iterrows()):
        count += 1
        current_epoch = int(np.floor(count/lookback))
        positions = sim.get_open_positions()
        #only after first epoch can trading start
        if current_epoch > 0:
            new_epoch = previous_epoch < current_epoch
            for asset in returns.columns:
                s_score = s_score_forward.iloc[count, returns.columns.get_loc(asset)]
                #check if there is a trade already open or not
                if sim.is_position_open(asset):
                    trade = sim.get_trade(asset)

                    ##take profits
                    if (trade.direction == 'long') & (s_score > -exit):
                        #close  long position
                        sim.close_trade(asset,'tp')
                    elif (trade.direction == 'short') & (s_score < exit):
                        #close short position
                        sim.close_trade(asset,'tp')
                    
                    ##stop losses
                    if (trade.direction == 'long') & (s_score < -stoploss):
                        #close  long position
                        sim.close_trade(asset,'sl')
                        sl_cooldown[asset] = False
                    elif (trade.direction == 'short') & (s_score > stoploss):
                        #close short position
                        sim.close_trade(asset,'sl')
                        sl_cooldown[asset] = False
                    
                    '''##rebalance if new epoch started check if the position hasnt been closed alread
                    if new_epoch & sim.is_position_open(asset):
                        neutral_portfolio = np.zeros(len(returns.columns))
                        betas = betas_all_epochs.loc[current_epoch,asset][0]
                        for k, beta in enumerate(betas):
                            weights = eig_portf.loc[current_epoch,'portfolios'][:,k]
                            neutral_portfolio += weights * beta

                        if trade.direction == 'long':
                            sim.readjust_trade(asset,-neutral_portfolio) #take negative of neutral portfolio to hedge
                        else:
                            sim.readjust_trade(asset,neutral_portfolio)'''
                else:
                    ##open position if R^2,kappa and s_score are ok also cannot have been stop lossed in current epoch
                    R2 = R2_training_all_epochs.loc[current_epoch, asset]
                    kappa = OU_params.loc[current_epoch,asset][0]
                    ok_to_trade = (R2 > R2_thresh) & (kappa > kappa_thresh) &  (sl_cooldown[asset])

                    ##open short spread position
                    if (s_score > entry) & ok_to_trade:
                        neutral_portfolio = np.zeros(len(returns.columns))
                        betas = betas_all_epochs.loc[current_epoch,asset][0]
                        for k, beta in enumerate(betas):
                            weights = eig_portf.loc[current_epoch,'portfolios'][:,k]
                            neutral_portfolio += weights * beta
                        sim.open_trade(asset,'short', neutral_portfolio)

                    ##open long spread position 
                    if (s_score < -entry) & ok_to_trade:
                        neutral_portfolio = np.zeros(len(returns.columns))
                        betas = betas_all_epochs.loc[current_epoch,asset][0]
                        for k, beta in enumerate(betas):
                            weights = eig_portf.loc[current_epoch,'portfolios'][:,k]
                            neutral_portfolio += weights * beta
                        sim.open_trade(asset,'long', -neutral_portfolio)

            #keep track of new epochs
            if new_epoch:
                sl_cooldown = dict.fromkeys(returns.columns, True)
            previous_epoch = current_epoch
        
        if count == len(returns -2):
            return sim
        #count the day
        sim.time_increment()
        
    return sim