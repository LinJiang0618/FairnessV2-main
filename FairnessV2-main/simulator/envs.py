from simulator.nodes import *
from simulator.orders import *
from simulator.couriers import *
from algorithm.AC import *
from algorithm.RL_utils import *
import numpy as np
import pandas as pd




class Region:

    def __init__(self,courier_init,real_orders,M,N,max_day,max_time):
        self.courier_init = courier_init  # 骑手信息初始化

        # 全体订单信息：
        self.real_orders = real_orders  # 订单信息

        # 全体时间信息
        self.city_day = 0  # 对应系统发展的第几天
        self.city_time = 0 # 对应这一天的什么时候
        self.max_city_time = max_time # 从早上8点到晚上10点，共计统计14个小时
        self.max_city_day = max_day # 最大的城市天数

        # 全体骑手信息
        self.courier_list = []  # 骑手列表
        self.courier_dict = {}  # 骑手ID对应的字典
        self.courier_num = 0  # 对应的总骑手数目
        self.node_couriers = [] # 所有区域的骑手信息,按照0~99进行排列

        # 全体网格信息
        self.M = M  # 结点横轴数
        self.N = N  # 结点纵轴数
        self.node_num = self.M * self.N # 总的结点数
        self.node_list = [Node(i) for i in range(self.M * self.N)]  # 区域的结点列表
        self.node_dict = {}
        self.construct_map_simulation()

        # 时间步信息
        self.day_orders = []  # 这一天所有的订单信息，按0~167进行排列

        # 环境参数信息
        self.region_mean_efficiency = 0 # 所有骑手的效率


    def construct_node_dict(self):
        for node in self.node_list:
            self.node_dict[node.node_index] = node

    def construct_map_simulation(self):  #循环结点设置邻居结点ID
        for node in self.node_list:
            node.set_neighbors()  # 给node赋上了邻居结点的ID

    def bootstrap_one_day_order(self): # 读取一天的订单
        day_orders = [[] for _ in np.arange(self.max_city_time)]   # 168个
        for day_real_orders in self.real_orders[self.city_day]:  # 循环这一天
            for real_order in day_real_orders: # 循环这一天每个时间的订单
                # 订单信息包含：订单ID,订单天数，订单时间，订单结点位置，订单商家经度，维度，订单用户经度，纬度，订单预计送达时间，订单价格
                start_time =int(real_order[2])  # 订单时间
                day_orders[start_time].append(Order(real_order[0],real_order[1],real_order[2],real_order[3],real_order[4],
                                                    real_order[5],real_order[6],real_order[7],real_order[8],real_order[9]))
        self.day_orders = day_orders # 表示day_orders



    def clear_order_list(self): # 清空上一步order的订单信息
        for node in self.node_list:
            node.clear_node_order_list()

    def step_bootstrap_order(self):  # 将订单对应到结点
        # 加载每一个时间步的订单
        step_orders = self.day_orders[self.city_time]
        if len(step_orders) != 0:
            for iorder in step_orders:  # order有订单时间(对应到self.city_time里),包含商家经纬度,用户经纬度,结点位置
                order_node = self.node_list[iorder.order_node]
                order_node.step_add_orders(iorder)


    def bootstrap_node_courier(self):  # 将骑手分配给结点
        node_couriers = [[] for _ in np.arange(self.node_num)]
        # 环境中的骑手
        for courier_node_list in self.courier_init:
            for courier in courier_node_list:
            #骑手ID, 骑手结点, 骑手经度, 骑手纬度
                courier_node = int(courier[1])
                courier_var = Courier(courier[0])
                courier_var.set_courier_init_loc(courier[2],courier[3])
                node_couriers[courier_node].append(courier_var)   # 包含了这个node包含哪些骑手
        self.node_couriers =  node_couriers
        for node_courier in self.node_couriers:
            for courier in node_courier:
                self.courier_list.append(courier)
                self.courier_dict[courier.courier_id] = courier

        # 将这些骑手对应到每个结点中
        for i in range(self.node_num):
            self.node_list[i].set_courier_info(self.node_couriers[i])


    def reset_courier_info(self):  # 每天开始设置骑手的初始位置
        for courier in self.courier_list:
            courier.set_loc(courier.init_longitude,courier.init_latitude)   # 设置骑手的位置
            courier.clean_order_info()



    def get_day_info(self,day): # 给环境,骑手,结点都赋上值
        self.city_day = day
        for courier in self.courier_list:
            courier.get_day_index(day)
        for node in self.node_list:
            node.get_day_index(day)

    def reset_clean(self):  # 每天初始化
        self.city_time = 0  # 设置今天的时间
        self.reset_courier_info()  #  重设骑手初始位置,重置骑手信息
        self.bootstrap_one_day_order()  # 导入这一天的订单
        self.clear_order_list()
        self.step_bootstrap_order()  # 按时间导入订单


    def add_system_time(self):
        for node in self.node_list:
            node.city_time += 1  #结点时间向前推进一次
            for courier in node.courier_list:
                courier.city_time += 1 # 骑手时间向前推进一次

    def step(self,d_dict,agent,state_dim,action_dim): #1) 增加所有的时间信息 2) 导入新的订单 3) 获得新的next_state和next_action_state
        self.city_time += 1  # 增加系统的时间、
        self.add_system_time()   # 增加结点的时间信息和骑手的时间信息
        self.clear_order_list()   # 清空node中的订单,node中的订单都派给了骑手
        self.step_bootstrap_order()  # 按时间导入订单(获得新的orderList)
        # 获得新的next_state和next_action_state
        self.get_next_state(d_dict,agent,state_dim,action_dim)


    def get_next_state(self,d_dict,agent,state_dim,action_dim):
        for node in self.node_list:  # node订单已经确定,骑手已经更新,这时候已经是下一次了
            if node.node_index in d_dict:  # 只有在dict中，构成训练集合才需要更新
                # 需要state状态(直接统计当前骑手信息即可)
                # 挑一个动作赋值动作状态(take_action直接选择最大的那一个)
                supply_and_demand_state = node.count_supply_and_demand_state()  # 确定结点内订单需求和供应的比例,返回numpy类型 50
                if len(node.order_list) > 0:
                    orderCourierList = node.get_solution(node.order_list, node.courier_list)  # list中每个dict都是一个骑手分配方案
                    stateArray = node.get_state_info(orderCourierList)
                    stateArray = np.hstack(
                        (np.array([supply_and_demand_state] * len(orderCourierList)), np.array(stateArray)))
                    solutionNum = agent.take_next_action(stateArray)
                    next_state = stateArray[solutionNum]
                else:
                    stateArray = node.get_zero_state_info(node.courier_list)
                    next_state = np.hstack((np.array(supply_and_demand_state),np.array(stateArray)))
                d_dict[node.node_index].set_next_state(next_state[:state_dim])
                d_dict[node.node_index].set_next_action_state(next_state[state_dim:state_dim + action_dim])



    def process_momery(self,d):
        state = np.hstack((d.state,d.action_state))    #  连接state和action进行匹配
        reward = d.reward
        next_state = np.hstack((d.next_state,d.next_action_state))
        policy = d.action_prob
        return state,reward,next_state,policy




























