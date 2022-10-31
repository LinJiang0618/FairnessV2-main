from math import sin,asin,cos,radians,fabs,sqrt
from simulator.env_utility import *

class Route:

    def __init__(self,order_list,loc,pending_order_list,city_time,speed):

        self.order_list = order_list + pending_order_list
        self.order_flag_dict = {}  #
        self.loc = loc
        self.route_plan = []  # 元素是loc,接下来应该怎么走
        self.route_order_plan = []  # 元素是(order,flag),接下来要配送哪个订单,记录订单和订单的配送状态
        self.next_loc = 0  # 骑手下一战要去的地方
        self.speed = speed
        self.current_time = city_time  # 计算该路径时当前的时间

        self.route_money = 0 # 骑手这一趟赚的钱
        self.route_time = 0 # 骑手这一趟花时间
        self.overdue_time = 0 # 骑手这一趟误单的次数
        self.overdue_whole_time = 0 # 骑手这一趟误单的总时间


    def get_order_flag(self):
        order_id_list = [order.order_id for order,flag in self.order_list]
        order_flag_list = [flag for order,flag in self.order_list]
        self.order_flag_dict = dict(zip(order_id_list,order_flag_list))


    def get_route_plan(self):
        self.get_order_flag()
        temp_shop_list = {}  # 记录商家的位置  (shop_loc,order_id)
        temp_user_list = {}  # 记录用户的位置   (user_loc,order_id)
        loc_list = []  # 包含未来订单的信息，不包括本身的位置  (loc,order_id)  如果订单送达了需要更新
        count_time = 0
        self.route_loc = self.loc
        for order,flag in self.order_list:  # 包含订单和对应的flag
            if flag == 1:  # 如果已经取餐
                loc_list.append((order.user_loc,order.order_id))
            elif flag == 0:
                temp_shop_list[order.order_id] = (order.merchant_loc,order.order_id)
                loc_list.append((order.merchant_loc,order.order_id))
                temp_user_list[order.order_id] = (order.user_loc,order.order_id)

        while len(loc_list):  # (位置,订单ID)  还没有送完需要更新
            loc_distance_list = [get_distance_hav
                                 (self.route_loc.latitude,self.route_loc.longitude,loc[0].latitude,loc[0].longitude)
                                 for loc in loc_list]  # 当前的位置送到各个loc的距离
            temp_loc_list = list(zip(loc_distance_list,loc_list))  #[距离，[位置,ID]]
            temp_loc_list.sort(key = lambda x:x[0])  # 利用贪心配送距离最近的位置
            if count_time == 0:  # 只用来记录下一个位置
                self.next_loc = temp_loc_list[0][1][0]   # 找到next loc
            loc_list.remove(temp_loc_list[0][1])  # 去除这个loc和id
            self.route_plan.append(temp_loc_list[0][1][0])  # 增加loc
            if self.order_flag_dict[temp_loc_list[0][1][1]] == 0:
                self.order_flag_dict[temp_loc_list[0][1][1]] = 1
                loc_list.append(temp_user_list[temp_loc_list[0][1][1]])  # 将用户信息放进去
                self.route_order_plan.append((temp_loc_list[0][1][1],0))   # 0表示待取餐,意思是去这个点取餐
            elif self.order_flag_dict[temp_loc_list[0][1][1]] == 1:
                self.order_flag_dict[temp_loc_list[0][1][1]] = 2    # 这里已经送到了
                self.route_order_plan.append((temp_loc_list[0][1][1],1))   # 1表示待送达，意思是去这个点送达
            count_time += 1

        temp_route_order_plan = []
        for order_id,flag in self.route_order_plan:
            for order,_ in self.order_list:   # order是基于order_list中拿出的
                if order.order_id == order_id:
                    temp_route_order_plan.append((order,flag))
        self.route_order_plan = temp_route_order_plan

    def count_route_money_and_time_and_overdue(self):  #记录增加的钱和时间
        loc_list = []
        loc_list.append(self.loc)  # 当前位置
        for loc in self.route_plan:
            loc_list.append(loc)  #未来增加的位置
        loc_distance_list = [(loc_list[i],loc_list[i+1]) for i in range(0,len(loc_list)-1)]
        distance_list = [get_distance_hav(x[0].latitude,x[0].longitude,x[1].latitude,x[1].longitude)
                         for x in loc_distance_list]
        for dis in distance_list:
            self.route_time += (dis / self.speed) * 60
        for order,_ in self.order_list:
            self.route_money += order.price
        self.check_overdue_situation()  # 检查该route中是否存在超时等情况发生
        return self.route_time,self.route_money,self.overdue_time,self.overdue_whole_time  #返回配送时间，配送价格，超时次数,总超时时间

    def check_overdue_situation(self): #需要有骑手的配送订单顺序，同时考虑订单的限定时间
        loc = self.loc
        time_list = []
        order_route_time_list = []
        order_time_dict = {}
        for order,flag in self.route_order_plan:  #route_order_plan 返回的是订单和其配送状态
            if flag == 0:
                next_loc = order.merchant_loc
                distance = get_distance_hav(loc.latitude,loc.longitude,next_loc.latitude,next_loc.longitude)
                time_list.append((distance / self.speed) * 60)   # 应该记录的是时间节点
                loc = Loc(next_loc.longitude,next_loc.latitude)
            if flag == 1:
                next_loc = order.user_loc
                distance = get_distance_hav(loc.latitude,loc.longitude,next_loc.latitude,next_loc.longitude)
                time_list.append((distance / self.speed) * 60)
                order_time_dict[order] = (sum(time_list))
                loc = Loc(next_loc.longitude,next_loc.latitude)
        for order,_ in self.route_order_plan:  #判断每个订单在该route下是否存在超时(涉及到action_collect方面的选择)
            order_create_time = order.get_order_create_time()
            delivery_time = 5 * (self.current_time - order_create_time) + order_time_dict[order]   #（当前时间 - 订单接单时间） + 订单在路径中所花费的时间
            overdue_time = order.promise_delivery_time - delivery_time
            if overdue_time < 0:
                self.overdue_time += 1
                self.overdue_whole_time += abs(overdue_time)
            if overdue_time < 0 and overdue_time >= -3:
                self.route_money -= 0.5 * order.price
            elif overdue_time < -3 and overdue_time >= -8:
                self.route_money -= order.price
            elif overdue_time < -8:
                self.route_money -= (1.5 * order.price)






