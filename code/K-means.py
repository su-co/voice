import numpy as np
import pandas as pd
import random
import sys
import time


class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray  # 点集
        self.cluster_num = cluster_num  # 簇数
        self.points = self.__pick_start_point(ndarray, cluster_num)  # points是cluster_num个点的坐标列表

    def cluster(self):
        result = []
        for i in range(self.cluster_num):  # 假设cluster_num=3，result=[[],[],[]]
            result.append([])
        for item in self.ndarray:  # 将所有点按照中心点points进行分簇（3簇）
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):  # 寻找三个中心点中的最近中心点
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]  # 加入最近中心点对应簇
        new_center = []
        for item in result:  # 计算每一簇的中心点
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            return result
        # 未达到稳态，继续递归
        self.points = np.array(new_center)  # 更新中心点
        return self.cluster()

    def __center(self, list):
        '''计算一组坐标的中心点
        '''
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        '''计算两点间距
        '''
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):  # 随机选取cluster_num个中心点！

        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")

        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(),
                                cluster_num)  # 从ndarray随机抽取cluster_num个点，取下标
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())  # 将随机选取的点坐标加入到points
        return np.array(points)
