import math
from math import sin,asin,cos,radians,fabs,sqrt

class Loc:

    def __init__(self,longitude,latitude):
        self.longitude = longitude
        self.latitude = latitude


def set_node(lat,lon):
    return int(math.floor((31.0 - lat/0.03)*10 + math.floor((lon-121.1)/0.03)))

def set_lon_lat_range(node):  #返回经度下界,经度上界,纬度下界,纬度上界
    return round(121.1 + int(node%10) * 0.03,2),round(121.1 + int(node%10) * 0.03 + 0.03,2),\
           round(31.0 - int(node/10)* 0.03 - 0.03,2),round(31.0 - int(node/10) * 0.03,2)

'''计算两点之间的取餐距离'''

EARTH_RADIUS=6371

def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1):
    """
     用haversine公式计算球面两点间的距离
    """
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance


class Reward: # 关于rewad当中的一些参数用法
    def __init__(self):
        self.alpha = 0.6
        self.gamma = 0.9
        self.mu = 1.1

reward = Reward()