from simulator.couriers import *
from simulator.nodes import *
from simulator.orders import *
import numpy as np
import pandas as pd


class DispatchPair:
    def __init__(self,courier,orderList):
        self.courier = courier
        self.orderList = orderList


class DispatchSolution:
    def __init__(self):
        self.dispatchPairList = []   # 骑手用户的派单列表
        self.state = None
        self.next_state = None
        self.action_state = None
        self.action_prob = None # 用于softmax求导时
        self.next_action_state = None
        self.reward = 0

    def get_dispatchPairList(self,dispatchDict):
        for key,value in dispatchDict.items():  # key为courier,value为order_list
            DP = DispatchPair(key,value)
            self.dispatchPairList.append(DP)

    def set_reward(self,reward):  # 将奖励赋值给骑手
        self.reward = reward

    def set_state(self,state):
        self.state = state

    def set_action_state(self,action_state):
        self.action_state = action_state

    def set_action_prob(self,action_prob):
        self.action_prob = action_prob

    def set_next_state(self,next_state):
        self.next_state = next_state

    def set_next_action_state(self,next_action_state):
        self.next_action_state = next_action_state