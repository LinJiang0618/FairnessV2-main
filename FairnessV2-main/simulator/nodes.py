from simulator.env_utility import *
import numpy as np
import copy
import random

class Node(object):
    def __init__(self,index):
        self.node_index = index # 这个结点的index

        # 时间信息
        self.city_day = 0
        self.city_time = 0

        # 骑手信息
        self.courier_list = []  # 区域骑手列表(每个step都需要更新)
        self.courier_dict = {}  # 区域字典

        #订单信息
        self.order_list = []  # 一个step内区域订单信息(每个step都需要更新)

        # 结点供需信息
        self.node_supply = 0  # 骑手的空间
        self.node_demand = 0 # 订单量

        # 邻居网格
        self.n_side = 0
        self.neighbors = []  # 周围八个邻居网格

        # 结点公平信息
        self.node_mean_efficiency = 0


    def set_neighbors(self):
        # num1是个位数，num2是十位数
        num1 = int(self.node_index % 10)
        num2 = int(self.node_index / 10)
        if self.node_index == 0:
            nodes_list = [10,11,1]
        elif self.node_index == 90:
            nodes_list = [80,81,91]
        elif self.node_index == 9:
            nodes_list = [8,18,19]
        elif self.node_index == 99:
            nodes_list = [98,88,89]
        elif num2 == 0:
            nodes_list = [num1-1,10+num1-1,10+num1,10+num1+1,num1+1]
        elif num2 == 9:
            nodes_list = [90+num1-1,80+num1-1,80+num1,80+num1+1,90+num1+1]
        elif num1 == 0:
            nodes_list = [num2*10-10,num2*10-9,num2*10+1,num2*10+11,num2*10+10]
        elif num1 == 9:
            nodes_list = [num2*10-1,num2*10-2,num2*10+8,num2*10+18,num2*10+19]
        else:
            nodes_list = [num2*10+num1-1,num2*10+num1-11,num2*10+num1-10,num2*10+num1-9,
                         num2*10+num1+1,num2*10+num1+11,num2*10+num1+10,num2*10+num1+9]
        self.neighbors = nodes_list
        self.n_side = len(nodes_list)


    def clear_node_order_list(self):
        self.order_list = []   # 将结点中order_list进行清空


    def step_add_orders(self,order):
        self.order_list.append(order)

    def set_courier_info(self,courier_list):   # 确定区域的骑手
        for courier in courier_list:
            self.courier_list.append(courier)
            self.courier_dict[courier.courier_id] = courier
            courier.set_node(self.node_index)

    def get_day_index(self,day):
        self.city_day = day

    def count_supply_and_demand_state(self):
        supply_and_demand_state = [0 for i in range(50)]
        lower_longitude,upper_longitude,lower_latitude,upper_latitude = set_lon_lat_range(self.node_index)
        node_supply,node_demand = 0,0
        # 差距为0.006
        # 计算供应
        for courier in self.courier_list:
            longitude_index = int((courier.longitude - lower_longitude) / 0.006)
            if longitude_index == 5:
                longitude_index = 4
            if longitude_index == -1:
                longitude_index = 0
            latitude_index = int((courier.latitude - lower_latitude) / 0.006)
            if latitude_index == 5:
                latitude_index = 4
            if latitude_index == -1:
                latitude_index = 0
            supply_index = (4 - latitude_index) * 5 + (longitude_index - 0)
            supply_and_demand_state[supply_index * 2] += (courier.capacity - courier.order_num)  # 增加骑手供应的量
            node_supply += (courier.capacity - courier.order_num)
        # 计算需求,订单为需求
        for order in self.order_list:
            longitude_index = int((order.merchant_longitude - lower_longitude) / 0.006)
            if longitude_index == 5:
                longitude_index = 4
            if longitude_index == -1:
                longitude_index = 0
            latitude_index = int((order.merchant_latitude - lower_latitude) / 0.006)
            if latitude_index == 5:
                latitude_index = 4
            if latitude_index == -1:
                latitude_index = 0
            demand_index = (4 - latitude_index) * 5 + (longitude_index - 0)
            supply_and_demand_state[demand_index * 2 + 1] += 1    # 增加订单量
            node_demand += 1

        self.node_supply = node_supply
        self.node_demand = node_demand
        return np.array(supply_and_demand_state)


    def get_solution(self,nodeOrderList,nodeCourierList):
        courierList = []
        for courier in nodeCourierList:
            courierList.append(courier)
        orderList = []
        for order in nodeOrderList:
            orderList.append(order)

        for courier in courierList:
            courier.update_simulate_information()
        removalIndexList = self.check_courier_capacity(courierList)
        courierList = [courierList[i] for i in range(0, len(courierList), 1) if i not in removalIndexList]   # 去掉订单满了的骑手
        rewardMatrix = np.zeros((len(orderList),len(courierList)))
        for i in range(len(orderList)):
            for j in range(len(courierList)):
                rewardMatrix[i, j] = self.get_initial_reward(orderList[i], courierList[j])
        courierOrderList = []


        # 第一步 进行贪心遍历
        for courier_index in range(len(courierList)): # 遍历所有骑手,基于每个骑手选择最优的那个，后续继续选择最优
            nullList = []
            for j in range(len(courierList)):
                nullList.append([])
            courierOrderDict = dict(zip(courierList,nullList))  # 关于骑手和对应的订单列表
            tempRewardMatrix = copy.deepcopy(rewardMatrix)
            tempOrderList = copy.deepcopy(orderList)
            tempCourierList = copy.deepcopy(courierList)
            orderIndex = np.argmax(tempRewardMatrix[:,courier_index])
            waitingCourier = tempCourierList[courier_index]  # 待分配的骑手
            waitingOrder = tempOrderList[int(orderIndex)]  # 待分配的订单
            courierID = waitingCourier.courier_id
            orderID = waitingOrder.order_id
            courier_i, order_i = 0, 0
            for j in range(len(nodeCourierList)):
                if courierID == nodeCourierList[j].courier_id:
                    courier_i = j
                    break
            for k in range(len(nodeOrderList)):
                if orderID == nodeOrderList[k].order_id:
                    order_i = k
                    break
            courierOrderDict[nodeCourierList[courier_i]].append(nodeOrderList[order_i])  # 加到dict当中  是骑手 + 订单列表的组合

            tempOrderList.pop(int(orderIndex))  # 将已经分配的订单去除
            tempRewardMatrix = np.delete(tempRewardMatrix, orderIndex, axis=0)
            waitingCourier.add_simulate_order_num()  # 骑手当前的单量加一
            waitingCourier.add_simulate_order_list(waitingOrder)  # 将待派订单派过去
            waitingCourier.add_simulate_order_flag(waitingOrder)  # 增加待派订单改变订单的flag
            removalIndexList = self.check_courier_capacity(tempCourierList)  # 检查骑手容量

            if len(removalIndexList) != 0:  # 去除长度为0
                tempCourierList = [tempCourierList[i] for i in range(0, len(tempCourierList), 1) if
                                   i not in removalIndexList]  # 去除多的courier
                tempRewardMatrix = np.delete(tempRewardMatrix, courier_index, axis=1)  # 将过容的骑手从reward中剔除

            else:
                for j in range(tempRewardMatrix.shape[0]):  # 骑手已经接了新的订单，需要更新他与其他订单的reward参数
                    tempRewardMatrix[j, courier_index] = self.get_new_reward(tempOrderList[j], tempCourierList[courier_index])



            while len(tempOrderList) > 0:  # 当订单还没有分配完
                orderIndex = np.where(tempRewardMatrix == np.max(tempRewardMatrix))[0][0]
                courierIndex = np.where(tempRewardMatrix == np.max(tempRewardMatrix))[1][0]
                waitingOrder = tempOrderList[orderIndex]  # 待分配订单
                waitingCourier = tempCourierList[courierIndex]  # 待分配骑手
                courierID = waitingCourier.courier_id
                orderID = waitingOrder.order_id
                courier_i, order_i = 0, 0
                for j in range(len(nodeCourierList)):
                    if courierID == nodeCourierList[j].courier_id:
                        courier_i = j
                        break
                for k in range(len(nodeOrderList)):
                    if orderID == nodeOrderList[k].order_id:
                        order_i = k
                        break
                courierOrderDict[nodeCourierList[courier_i]].append(nodeOrderList[order_i])  # 加到dict当中  是骑手 + 订单列表的组合

                tempOrderList.pop(orderIndex)  # 将已经分配的订单去除
                tempRewardMatrix = np.delete(tempRewardMatrix, orderIndex, axis=0)  # 对于对应的matrix列表进行更新,去除被分配订单那一列
                waitingCourier.add_simulate_order_num()  # 骑手当前的单量加一
                waitingCourier.add_simulate_order_list(waitingOrder)  # 将待派订单派过去
                waitingCourier.add_simulate_order_flag(waitingOrder)  # 增加待派订单改变订单的flag
                removalIndexList = self.check_courier_capacity(tempCourierList)  # 检查骑手容量

                if len(removalIndexList) != 0:  # 去除长度为0
                    tempCourierList = [tempCourierList[i] for i in range(0, len(tempCourierList), 1) if
                                       i not in removalIndexList]
                    tempRewardMatrix = np.delete(tempRewardMatrix, courierIndex, axis=1)  # 将过容的骑手从reward中剔除
                    # 这里已经将骑手删去了,此时courier_index

                else:
                    for j in range(tempRewardMatrix.shape[0]):  # 骑手已经接了新的订单，需要更新他与其他订单的reward参数
                        tempRewardMatrix[j, courierIndex] = self.get_new_reward(tempOrderList[j], tempCourierList[courierIndex])

            courierOrderList.append(courierOrderDict)  # 每个dict即是一种分配方案
        courierOrderListFirstStepLen = len(courierOrderList)

        # 第二步  参与一些非贪心的路径增大系统选择范围(增加10个选择方案)
        for add_index in range(10):  # 一定要使得分配方案达到10
            nullList = []
            for j in range(len(courierList)):
                nullList.append([])
            courierOrderDict = dict(zip((courierList), nullList))
            tempRewardMatrix = copy.deepcopy(rewardMatrix)  # 深复制一个rewardMatrix,此时所有的骑手都可以被派单
            tempOrderList = copy.deepcopy(orderList)
            tempCourierList = copy.deepcopy(courierList)
            while len(tempOrderList) > 0:  # 当订单还没有分配完
                orderIndex = random.sample(list(range(len(tempOrderList))), 1)[0]  # 随机选择了order
                courierIndex = random.sample(list(range(len(tempCourierList))), 1)[0]  # 随机选择了courier

                waitingOrder = tempOrderList[orderIndex]  # 待分配订单
                waitingCourier = tempCourierList[courierIndex]  # 待分配骑手
                courierID = waitingCourier.courier_id
                orderID = waitingOrder.order_id
                courier_i, order_i = 0, 0
                for j in range(len(nodeCourierList)):
                    if courierID == nodeCourierList[j].courier_id:
                        courier_i = j
                        break
                for k in range(len(nodeOrderList)):
                    if orderID == nodeOrderList[k].order_id:
                        order_i = k
                        break
                courierOrderDict[nodeCourierList[courier_i]].append(nodeOrderList[order_i])  # 加到dict当中  是骑手 + 订单列表的组合

                tempOrderList.pop(orderIndex)  # 将已经分配的订单去除
                tempRewardMatrix = np.delete(tempRewardMatrix, orderIndex, axis=0)  # 对于对应的matrix列表进行更新,去除被分配订单那一列
                waitingCourier.add_simulate_order_num()  # 骑手当前的单量加一
                waitingCourier.add_simulate_order_list(waitingOrder)  # 将待派订单派过去
                waitingCourier.add_simulate_order_flag(waitingOrder)  # 增加待派订单改变订单的flag
                removalIndexList = self.check_courier_capacity(tempCourierList)  # 检查骑手容量

                if len(removalIndexList) != 0:  # 去除长度为0
                    tempCourierList = [tempCourierList[i] for i in range(0, len(tempCourierList), 1) if
                                       i not in removalIndexList]
                    tempRewardMatrix = np.delete(tempRewardMatrix, courierIndex, axis=1)  # 将过容的骑手从reward中剔除

                else:
                    for j in range(tempRewardMatrix.shape[0]):  # 骑手已经接了新的订单，需要更新他与其他订单的reward参数
                        tempRewardMatrix[j, courierIndex] = self.get_new_reward(tempOrderList[j], tempCourierList[courierIndex])

            courierOrderList.append(courierOrderDict)  # 每个dict即是一种分配方案

        return courierOrderList





    def check_courier_capacity(self,courierList):
            removalIndexList = []
            for i in range(len(courierList)):
                if courierList[i].simulate_num_order >= courierList[i].capacity:
                    removalIndexList.append(i)
            return removalIndexList



    def get_initial_reward(self,order,courier):
        courier.update_simulate_information()
        add_route_time, add_route_money, add_overdue_time, add_overdue_whole_time = courier.simulate_get_route_information(
            [order])
        reward_one = pow(reward.gamma, add_route_time) * pow(reward.mu, (
                    courier.capacity - courier.order_num)) * add_route_money  # reward的第一部分
        reward_two = abs(courier.acc_efficiency - self.node_mean_efficiency)
        allreward = reward.alpha * reward_one #- (1 - reward.alpha) * reward_two
        return allreward


    def get_new_reward(self,order,courier):
        add_route_time, add_route_money, add_overdue_time, add_overdue_whole_time = courier.simulate_get_route_information(
            [order])
        reward_one = pow(reward.gamma, add_route_time) * pow(reward.mu, (
                courier.capacity - courier.order_num)) * add_route_money  # reward的第一部分
        reward_two = abs(courier.acc_efficiency - self.node_mean_efficiency)
        allreward = reward.alpha * reward_one  # - (1 - reward.alpha) * reward_two
        return allreward


    def get_state_info(self,orderCourierList):  # 这一版是平均版，先跑起来看一看效果
        orderCourierStateArray = []
        # 统计了以下特征：
        # 1、骑手现有本身的订单量
        # 2、骑手所在区域的订单供需
        # 3、骑手平均效率
        # 4、骑手距离系统总效率的差距
        # 5、骑手被分配的订单量
        # 6、骑手与订单间的平均距离
        # 7、骑手接单后增加的平均时间
        # 8、骑手接单后增加的平均金钱
        # 9、骑手接单后增加的平均误单数目
        # 10、骑手接单后增加的平均误单时间
        # 11、骑手接单后剩下的骑手容量
        for orderCourierDict in orderCourierList:
            # 这四个为骑手的状态信息(5个)
            courier_order_num_list = [] # 骑手订单量
            courier_route_time_list = []  # 骑手当前的配送时间
            courier_route_money_list = []  # 骑手当前的配送金钱
            courier_efficiency_list = [] # 骑手效率
            courier_efficiency_gap_list = []   # 骑手效率与平均效率之间的绝对值
            # 这些为对应的动作信息（7个）
            courier_pending_order_num_list = [] # 骑手被分配的订单量
            courier_order_mean_distance_list = [] # 骑手订单间的平均距离
            add_route_time_list = []
            add_route_money_list = []
            add_overdue_time_list = []
            add_overdue_whole_time_list = []
            add_courier_remaining_capacity_list = []
            for key,value in orderCourierDict.items():  # key为骑手，value为订单量
                courier_order_num_list.append(key.order_num)   # 骑手现有的订单量
                courier_route_time_list.append(key.route_time) # 骑手空间供应
                courier_route_money_list.append(key.route_money)  # 订单空间供应
                courier_efficiency_list.append(key.acc_efficiency)  # 记录骑手的效率
                courier_efficiency_gap_list.append(abs(key.acc_efficiency - self.node_mean_efficiency))  # 记录骑手效率与平均效率的绝对值
                courier_pending_order_num_list.append(len(value)) # 骑手被分配的订单量
                mean_distance = []
                for order in value:
                    mean_distance.append(get_distance_hav(key.loc.latitude,key.loc.longitude,order.merchant_loc.latitude,order.merchant_loc.longitude))
                if len(mean_distance) == 0:
                    courier_order_mean_distance_list.append(0)
                else:
                    courier_order_mean_distance_list.append(np.mean(np.array(mean_distance)))  # 骑手和订单间的平均距离
                key.update_simulate_information()  # 计算平均距离
                add_route_time, add_route_money, add_overdue_time, add_overdue_whole_time = key.simulate_get_route_information(
                    value)
                add_route_time_list.append(add_route_time)  # 增加时间
                add_route_money_list.append(add_route_money)  # 增加金钱
                add_overdue_time_list.append(add_overdue_time)  # 增加的误单数目
                add_overdue_whole_time_list.append(add_overdue_whole_time)  # 增加的误单时间
                add_courier_remaining_capacity_list.append(key.capacity - key.order_num - len(value))   # 计算剩下的骑手容量
            # 计算平均值和方差
            orderCourierState = []

            # 有十个特征（state）
            orderCourierState.append(np.mean(np.array(courier_order_num_list)))
            # orderCourierState.append(np.var(np.array(courier_order_num_list)))
            orderCourierState.append(np.mean(np.array(courier_route_time_list)))
            # orderCourierState.append(np.var(np.array(courier_route_time_list)))
            orderCourierState.append(np.mean(np.array(courier_route_money_list)))
            # orderCourierState.append(np.var(np.array(courier_route_money_list)))
            orderCourierState.append(np.mean(np.array(courier_efficiency_list)))
            # orderCourierState.append(np.var(np.array(courier_efficiency_list)))
            orderCourierState.append(np.mean(np.array(courier_efficiency_gap_list)))
            # orderCourierState.append(np.var(np.array(courier_efficiency_gap_list)))

            # 有十四个特征（action）
            orderCourierState.append(np.mean(np.array(courier_pending_order_num_list)))
            # orderCourierState.append(np.var(np.array(courier_pending_order_num_list)))
            orderCourierState.append(np.mean(np.array(courier_order_mean_distance_list)))
            # orderCourierState.append(np.var(np.array(courier_order_mean_distance_list)))
            orderCourierState.append(np.mean(np.array(add_route_time_list)))
            # orderCourierState.append(np.var(np.array(add_route_time_list)))
            orderCourierState.append(np.mean(np.array(add_route_money_list )))
            # orderCourierState.append(np.var(np.array(add_route_money_list )))
            orderCourierState.append(np.mean(np.array(add_overdue_time_list)))
            # orderCourierState.append(np.var(np.array(add_overdue_time_list)))
            orderCourierState.append(np.mean(np.array(add_overdue_whole_time_list)))
            # orderCourierState.append(np.var(np.array(add_overdue_whole_time_list)))
            orderCourierState.append(np.mean(np.array(add_courier_remaining_capacity_list)))
            # orderCourierState.append(np.var(np.array(add_courier_remaining_capacity_list)))
            orderCourierStateArray.append(orderCourierState)

        return orderCourierStateArray


    def get_zero_state_info(self,courier_list):
        # 这四个为骑手的状态信息(5个)
        courier_order_num_list = []  # 骑手订单量
        courier_route_time_list = []  # 骑手当前的配送时间
        courier_route_money_list = []  # 骑手当前的配送金钱
        courier_efficiency_list = []  # 骑手效率
        courier_efficiency_gap_list = []  # 骑手效率与平均效率之间的绝对值
        # 这些为对应的动作信息（7个）
        courier_pending_order_num_list = []  # 骑手被分配的订单量
        courier_order_mean_distance_list = []  # 骑手订单间的平均距离
        add_route_time_list = []
        add_route_money_list = []
        add_overdue_time_list = []
        add_overdue_whole_time_list = []
        add_courier_remaining_capacity_list = []
        for courier in courier_list:
            courier_order_num_list.append(courier.order_num)  # 骑手现有的订单量
            courier_route_time_list.append(courier.route_time)  # 骑手空间供应
            courier_route_money_list.append(courier.route_money)  # 订单空间供应
            courier_efficiency_list.append(courier.acc_efficiency)  # 记录骑手的效率
            courier_efficiency_gap_list.append(abs(courier.acc_efficiency - self.node_mean_efficiency))  # 记录骑手效率与平均效率的绝对值
            courier_pending_order_num_list.append(0)  # 骑手被分配的订单量
            courier_order_mean_distance_list.append(0) # 骑手订单间的平均距离
            add_route_time_list.append(0)
            add_route_money_list.append(0)
            add_overdue_time_list.append(0)
            add_overdue_whole_time_list.append(0)
            add_courier_remaining_capacity_list.append(0)
        orderCourierState = []

        # 有十个特征（state）
        orderCourierState.append(np.mean(np.array(courier_order_num_list)))
        orderCourierState.append(np.var(np.array(courier_order_num_list)))
        orderCourierState.append(np.mean(np.array(courier_route_time_list)))
        orderCourierState.append(np.var(np.array(courier_route_time_list)))
        orderCourierState.append(np.mean(np.array(courier_route_money_list)))
        orderCourierState.append(np.var(np.array(courier_route_money_list)))
        orderCourierState.append(np.mean(np.array(courier_efficiency_list)))
        orderCourierState.append(np.var(np.array(courier_efficiency_list)))
        orderCourierState.append(np.mean(np.array(courier_efficiency_gap_list)))
        orderCourierState.append(np.var(np.array(courier_efficiency_gap_list)))

        # 有十四个特征（action）
        orderCourierState.append(np.mean(np.array(courier_pending_order_num_list)))
        orderCourierState.append(np.var(np.array(courier_pending_order_num_list)))
        orderCourierState.append(np.mean(np.array(courier_order_mean_distance_list)))
        orderCourierState.append(np.var(np.array(courier_order_mean_distance_list)))
        orderCourierState.append(np.mean(np.array(add_route_time_list)))
        orderCourierState.append(np.var(np.array(add_route_time_list)))
        orderCourierState.append(np.mean(np.array(add_route_money_list)))
        orderCourierState.append(np.var(np.array(add_route_money_list)))
        orderCourierState.append(np.mean(np.array(add_overdue_time_list)))
        orderCourierState.append(np.var(np.array(add_overdue_time_list)))
        orderCourierState.append(np.mean(np.array(add_overdue_whole_time_list)))
        orderCourierState.append(np.var(np.array(add_overdue_whole_time_list)))
        orderCourierState.append(np.mean(np.array(add_courier_remaining_capacity_list)))
        orderCourierState.append(np.var(np.array(add_courier_remaining_capacity_list)))

        return orderCourierState


    def node_step(self,DB):   # 结点输入了内部的解决方案
        DB_reward = 0
        for dispatchPair in DB.dispatchPairList:  # 对解决方案中的配对方案进行遍历
            test_courier = dispatchPair.courier  # 匹配对中的骑手
            test_orderList = dispatchPair.orderList  # 匹配对中的订单列表
            test_route,add_route_time,add_route_money,add_overdue_time,add_overdue_whole_time = test_courier.get_route_information(test_orderList)
            test_courier.add_route_money = add_route_money  # 记录这两条临时信息
            test_courier.add_route_time = add_route_time
            # 并没有派给骑手,而是生成了骑手的route(仅仅是基于order去做的)
            test_courier.get_route_info(test_route)  # 将骑手的接单后新的route信息赋值给骑手
            test_courier.get_order_assignment(test_orderList)  # 骑手获得对应的订单
        for test_courier in self.courier_list:  # 结点中所有骑手都要执行订单
            test_courier.execute_order()  # 骑手执行新的route,更新了骑手的位置、订单状态和收益
            test_courier.update_route_information()
        self.node_mean_efficiency = 0 #结点平均效率
        for courier in self.courier_list:  # 在所有骑手派完单后,计算结点的平均效率
            self.node_mean_efficiency += courier.acc_efficiency
        self.node_mean_efficiency = (self.node_mean_efficiency) / len(self.courier_list)
        for courier in self.courier_list:
            courier_reward = self.set_reward(courier)   # 获得这个匹配对下的reward
            DB_reward += courier_reward   # 得到这个区域累计的reward
        DB.set_reward(DB_reward)




    def qiset_reward(self,test_courier):
        reward_one = pow(reward.gamma, test_courier.add_route_time) * pow(reward.mu, (
                test_courier.capacity - test_courier.order_num)) * test_courier.add_route_money  # reward的第一部分
        reward_two = abs(test_courier.acc_efficiency - self.node_mean_efficiency)
        courier_reward = reward.alpha * reward_one - (1 - reward.alpha) * reward_two
        return courier_reward






