from simulator.env_utility import *
from route_planning.route_plan import *
import copy

class Courier(object):

    def __init__(self,courier_id):

        # 骑手基本属性
        self.courier_id = courier_id
        self.capacity = 6
        self.speed = 20 # 骑手速度20km/h
        self.acc_time = 0 # 骑手累计工作时间
        self.acc_money = 0 # 骑手累计收入
        self.acc_efficiency = 0 # 骑手效率
        self.init_longitude = 0  # 骑手初始经度
        self.init_latitude = 0  # 骑手初始纬度

        #环境时间信息
        self.city_day = 0   # 骑手当前所处城市天数
        self.city_time =0   # 骑手当前所处城市时间

        #  骑手订单情况
        self.order_list = []  # 骑手拥有的订单信息
        self.order_flag = {} # 骑手所有订单的配送状态
        self.order_num = 0 # 骑手当前订单量

        # 骑手位置
        self.latitude = 0
        self.longitude = 0
        self.loc = None # 骑手当前位置
        self.node = None # 骑手当前所在结点编号

        # 骑手配送路径
        self.route = None  # 赋值骑手路径信息
        self.route_plan = []  # 骑手的配送计划,元素是Loc,表示接下来应该怎么走
        self.route_order_plan = []  # 骑手的配送计划,元素是(订单,配送状态),如果是0表示还未取餐,1表示已经取餐但还未送达
        self.route_time = 0  # 骑手当前route的时间
        self.route_money = 0  # 骑手当然route的收入
        self.overdue_whole_time = 0  # 骑手当前route总的误单时间
        self.overdue_time = 0 # 订单超时次数


        # 派单时模拟路径采用的模拟模块
        self.simulate_num_order = 0 # 现有订单数
        self.simulate_order_list = []  # 订单列表
        self.simulate_order_flag = {} # 订单当前的配送状态

        # 临时信息
        self.add_route_time = 0  # 一次执行后增加的route_time
        self.add_route_money = 0  # 一次执行后增加的route_money


    def set_courier_init_loc(self,longitude,latitude):  # 在导入courier_init设置骑手初始位置
        self.init_longitude = longitude
        self.init_latitude = latitude
        self.init_loc = Loc(longitude,latitude)

    def clean_order_info(self):  # 将骑手一些天属性进行修改
        self.order_list = []
        self.order_flag = {}
        self.order_num = 0
        self.route = None
        self.route_plan = []
        self.route_order_plan = []
        self.route_time = 0  # 骑手当前route的时间
        self.route_money = 0  # 骑手当然route的收入
        self.overdue_whole_time = 0  # 骑手当前route总的误单时间
        self.overdue_time = 0 # 订单超时次数
        self.simulate_num_order = 0  # 现有订单数
        self.simulate_order_list = []  # 订单列表
        self.simulate_order_flag = {}  # 订单当前的配送状态


    def set_loc(self,longitude,latitude):  # 设置骑手现在的位置
        self.longitude = longitude
        self.latitude = latitude
        self.loc = Loc(longitude,latitude)

    def get_day_index(self,day):
        self.city_day = day


    def update_simulate_information(self):
        self.simulate_num_order = self.order_num  # 现有订单数
        self.simulate_order_list = []
        for order in self.order_list:
            self.simulate_order_list.append(order)
        self.simulate_order_flag = {}
        for order,flag in self.order_flag.items():
            self.simulate_order_flag[order] = flag

    def set_node(self,node_index):
        self.node = node_index

    def add_simulate_order_num(self):  # 增加虚拟订单量
        self.simulate_num_order += 1

    def add_simulate_order_list(self, order):  # 扩展订单列表
        self.simulate_order_list.append(order)

    def add_simulate_order_flag(self, order):  # 修改订单flag
        self.simulate_order_flag[order] = 0  # 这个订单是未送的状态


    def simulate_get_route_information(self,pending_order_list):   # 相比当前订单增加了多少
        temp_order_list = self.simulate_order_list   # 当前的订单
        order_flag_dict = self.simulate_order_flag   # 当前订单的状态
        order_list = []
        num = len(temp_order_list)
        for order in temp_order_list:
            order_list.append((order,order_flag_dict[order]))
        for order in pending_order_list:
            order_list.append((order,0))

        route = Route(order_list[:num],self.loc,order_list[num:],self.city_time,self.speed)
        route.get_route_plan()
        route_time, route_money, overdue_time, overdue_whole_time = route.count_route_money_and_time_and_overdue()
        add_route_time = route_time - self.route_time
        add_route_money = route_money - self.route_money
        add_overdue_time = overdue_time - self.overdue_time
        add_overdue_whole_time = overdue_whole_time - self.overdue_whole_time

        return add_route_time, add_route_money, add_overdue_time, add_overdue_whole_time

    def get_route_information(self,pending_order_list):  # 将待选订单派给骑手,这次是真正派送了
        order_list = []
        num = len(self.order_list)
        for order in self.order_list:
            order_list.append((order,self.order_flag[order]))
        for order in pending_order_list:
            order_list.append((order,0))   # 在route的order_list中是包含flag的

        # self.available_flag.append(self.order_list[i].flag)
        route = Route(order_list[:num], self.loc, order_list[num:], self.city_time,self.speed)
        #这里还达不到self.route_plan的程度,self.route_plan直接用来更新骑手的轨迹
        route.get_route_plan()
        route_time,route_money,overdue_time,overdue_whole_time = route.count_route_money_and_time_and_overdue()
        add_route_time = route_time - self.route_time  #   返回骑手增加的信息
        add_route_money = route_money - self.route_money
        add_overdue_time = overdue_time - self.overdue_time
        add_overdue_whole_time = overdue_whole_time - self.overdue_whole_time
        return  route,add_route_time,add_route_money,add_overdue_time,add_overdue_whole_time

    def get_route_info(self,route):    # 每次轮回都需要调用该函数
        self.route = route  # 赋值骑手路径信息
        self.route_plan = route.route_plan  # 骑手的配送计划,元素是Loc,表示接下来应该怎么走
        self.route_order_plan = route.route_order_plan  # 骑手的配送计划,元素是(订单,配送状态),如果是0表示还未取餐,1表示已经取餐但还未送达
        self.route_time = route.route_time  # 骑手当前route的时间
        self.route_money = route.route_money  # 骑手当然route的收入
        self.overdue_whole_time = route.overdue_whole_time  # 骑手当前route总的误单时间
        self.overdue_time = route.overdue_time  # 订单超时次数

    def get_order_assignment(self,pending_order_list): #  骑手被分配这些订单
        for order in pending_order_list:
            order.courier_id = self.courier_id
            self.order_list.append(order)  # 增加当前的order_list
            self.order_flag[order] = 0  # 将order设置到order_flag中并赋值为0
        self.order_num = len(self.order_list)  # 更新order_num


    def execute_order(self):  # 骑手在5分钟内执行手中的订单,改变骑手的位置,订单和效益
        # 订单已经分配给骑手,骑手执行手中的order和oder_flag
        courier_loc = self.loc  # 记录骑手当前的位置
        if len(self.route_plan) == 0:  # 如果骑手没有单,效率等会发生变化
            self.acc_time += 5  # 时间增加了5分钟
            self.acc_efficiency = self.acc_money / self.acc_time  # 更新骑手的效率
        else:
            whole_time = 5
            acc_time = 0   # 花在这趟路程中的时间
            for i in range(len(self.route_plan)):  # 表示配送计划
                next_loc = self.route_plan[i]
                next_dis = get_distance_hav(next_loc.latitude,next_loc.longitude,courier_loc.latitude,courier_loc.longitude)
                next_time = (next_dis / self.speed) * 60  # 下一单需要花费的时间
                acc_time += next_time  # 累计的
                next_order = self.route_order_plan[i][0]  # 表示订单顺序(order,flag)
                next_order_flag = self.route_order_plan[i][1]  # 表示订单对应的派单状态,0为未取餐,1为已取餐
                if (whole_time - acc_time) >= 0:  # 到了这个地还有时间
                    self.loc = next_loc   # 骑手位置改变
                    self.longitude = self.loc.longitude
                    self.latitude = self.loc.latitude
                    courier_loc = next_loc
                    if next_order_flag == 0:
                        self.order_flag[next_order] = 1  # 该订单的状态被赋值为1
                        next_order.flag = 1  # 该订单的状态被赋值
                    else:
                        self.order_list.remove(next_order) # 订单信息从骑手处移除
                        self.order_flag.pop(next_order)
                        self.order_num -= 1
                        self.acc_money += next_order.price # 骑手累计收入
                else:   # 在半路事件已经到了,只改变骑手的位置,不涉及订单的送达与否
                    add_pre = 1 - ((acc_time - whole_time) / next_time)   # 已经行进的比例
                    next_longitude = round(add_pre * (next_loc.longitude - self.loc.longitude) + self.loc.longitude,4)
                    next_latitude = round(add_pre * (next_loc.latitude - self.loc.latitude) + self.loc.latitude,4)
                    self.loc = Loc(next_longitude,next_latitude)  # 确定骑手的位置
                    self.longitude = next_longitude
                    self.latitude = next_latitude
                    break
            self.acc_time += 5 # 更新骑手的工作时间
            self.acc_efficiency = self.acc_money / self.acc_time  # 更新骑手的效率

    def update_route_information(self):
        order_list = []
        for order,flag in self.order_flag.items():
            order_list.append((order,flag))
        num = 0
        # self.available_flag.append(self.order_list[i].flag)
        route = Route(order_list[:num], self.loc, order_list[num:], self.city_time,self.speed)
        #这里还达不到self.route_plan的程度,self.route_plan直接用来更新骑手的轨迹
        route.get_route_plan()
        _,_,_,_ = route.count_route_money_and_time_and_overdue()
        self.get_route_info(route)













