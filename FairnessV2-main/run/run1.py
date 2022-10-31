from simulator.envs import *
import numpy as np
import pandas as pd
import torch
#from dispatch.dispatch_method import *
from algorithm.AC import *
from algorithm.RL_utils import *
from dispatch.dispatch_solution import *
import warnings
warnings.filterwarnings('ignore')


import os
import pickle

day_index = 0  # 当前第几轮
max_day = 5 # 最大循环多少轮
max_time = 168 # 派单的轮数
matrix_longitude = 10
matrix_latitude = 10
memory_size = 1000
batch_size = 50

# 读取数据部分
real_orders = pd.read_pickle('../data/real_order_list.pickle')   # 30天订单信息
# 订单信息包含：订单ID,订单天数，订单时间，订单结点位置，订单商家经度，维度，订单用户经度，纬度，订单预计送达时间，订单价格

courier_init = pd.read_pickle('../data/courier_init.pickle')  # 骑手初始化信息
# 骑手信息包含：骑手ID,骑手结点,骑手经度,骑手纬度

env = Region(courier_init, real_orders, matrix_longitude, matrix_latitude, max_day, max_time)
env.construct_map_simulation()   # 初始化结点的node信息
env.construct_node_dict()   # 生成node_dict结点
env.bootstrap_node_courier()  # 将骑手信息赋值到结点中


action_dim = 7
state_dim = 55
T = 0 # 一天的时间计数

actor_lr = 0.001
critic_lr = 0.001
gamma = 0.9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = ActorCritic(state_dim,action_dim,actor_lr,critic_lr,gamma,device)
replay_buffer = ReplayBuffer(memory_size,batch_size)
# 每天有168 * 100 = 16800,共有300000次

day_recorder = open('../info/day_reward.txt','a+')   # a+不会被覆写

while day_index < max_day:  # 利用30天的订单
    env.get_day_info(day_index)  # 将day_index赋值到环境,骑手和结点中
    env.reset_clean()  # 导入这一天的骑手信息、订单信息
    day_reward = []
    step_recorder = open(f'../info/step_info/day{day_index}.txt','a+')

    while T < max_time:
        print(f'Day {T}:')
        d_dict = {}  #  每次都新建一个d_dict
        step_recorder.write("step" + str(T) + ':'+'\n')

        for node in env.node_list:
            # ① 每轮的时间向前进步
            # ② 每轮的订单被分配
            # ③每轮骑手的信息确认
            # node.courier_list # 表示node中的骑手信息
            # node.order_list # 表示node中的订单信息
            if len(node.order_list) > 0:  # 如果这个周期有单可派
                supply_and_demand_state = node.count_supply_and_demand_state()  # 确定结点内订单需求和供应的比例,返回numpy类型 50
                orderCourierList = node.get_solution(node.order_list, node.courier_list)  # list中每个dict都是一个骑手分配方案
                stateArray = node.get_state_info(orderCourierList)
                stateArray = np.hstack(
                (np.array([supply_and_demand_state] * len(orderCourierList)), np.array(stateArray)))
                solutionNum,action_prob = agent.take_action(stateArray)  # state_dim = 50 + 10 array_dim = 14
                orderCourierSolution = orderCourierList[solutionNum]  # 对应是dict

                # 设定两个step,一个是node-step,该step中也会包含强化学习学习的多元组,该多元组在env.step派单后还会补上next_state和next_action
                d = DispatchSolution()  # 该结点产生了一个派单方案
                d.get_dispatchPairList(orderCourierSolution)
                d.set_state(stateArray[solutionNum][:state_dim])
                d.set_action_state(stateArray[solutionNum][state_dim:state_dim + action_dim])   # state为np形式
                d.set_action_prob(action_prob[solutionNum])  # action_prob为tensor形式
                d_dict[node.node_index] = d   # 将结点和结点对应的解决方案对应起来

            #   一个时间段全部完成
        for node in env.node_list:
            if node.node_index in d_dict:
                node.node_step(d_dict[node.node_index])
            else:
                for courier in node.courier_list:
                    courier.execute_order()
                    courier.update_route_information()
        # 在所有结点中骑手订单信息都更新完后,开始1) 增加所有的时间信息 2) 导入新的订单 3) 获得新的next_state和next_action_state
        env.step(d_dict,agent,state_dim,action_dim)    # 输入并更新d_dict的信息
        T = T + 1  # 循环时间+1


        step_reward = []
        # 得到了d_list,接下来就是更新网络了
        for node_index,d in d_dict.items():
            state,reward,next_state,policy = env.process_momery(d)
            replay_buffer.add(state,reward,next_state,policy)
            step_reward.append(reward)   # 每一步的信息记录
            day_reward.append(reward)   # 每一个episode信息记录
        mean_step_reward = round(float(np.mean(np.array(step_reward))),4)
        step_recorder.write("average_reward_per_step:" + str(mean_step_reward) + "\n")
        print(f'time_step{T}: {mean_step_reward}.')


        batch_state, batch_reward, batch_next_s, batch_policy = replay_buffer.sample() # 进行更新(每次增加100,抽50进行更新)
        agent.update(batch_state, batch_reward, batch_next_s, batch_policy)


    step_recorder.close()
    mean_day_reward = round(float(np.mean(np.array(day_reward))),4)
    day_index += 1
    day_recorder.write('day' + str(day_index) + ':' + str(mean_day_reward) + '\n')




