import numpy as np
from route_planning.route_plan import *
# 订单信息包含：订单ID,订单天数，订单时间，订单结点位置，订单商家经度，维度，订单用户经度，纬度，订单预计送达时间，订单价格

class Order(object):
    def __init__(self,order_id,order_day,order_time,order_node,merchant_longitude,merchant_latitude,
                 user_longitude,user_latitude,promise_delivery_time,order_price):
        self.order_id = order_id
        self.order_day = order_day
        self.order_create_time = order_time  # 哪个时间被下发的
        self.order_node = order_node
        self.merchant_longitude = merchant_longitude
        self.merchant_latitude = merchant_latitude
        self.merchant_loc = Loc(merchant_longitude,merchant_latitude)
        self.user_latitude = user_latitude
        self.user_longitude = user_longitude
        self.user_loc = Loc(user_longitude, user_latitude)
        self.promise_delivery_time = promise_delivery_time
        self.price = order_price
        self.flag = 0 # flag表示还未取餐，1表示已经取餐但还未送达，后续在step中更新
        self.courier_id = 0 # 订单被分配给哪个骑手



    def get_order_create_time(self):  # 返回订单创建时间
        return self.order_create_time

