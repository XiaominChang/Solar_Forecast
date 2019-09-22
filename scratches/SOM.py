# -*- coding:utf-8 -*-
# som 实例算法 by 自由爸爸
import random
import math

input_layer = [[39, 281], [18, 307], [24, 242], [54, 333], [322, 35], [352, 17], [278, 22], [382, 48]]  # 输入节点
category = 2


class Som_simple_zybb():
    def __init__(self, category):
        self.input_layer = input_layer  # 输入样本
        self.output_layer = []  # 输出数据
        self.step_alpha = 0.5  # 步长 初始化为0.5
        self.step_alpha_del_rate = 0.95  # 步长衰变率
        self.category = category  # 类别个数
        self.output_layer_length = len(self.input_layer[0])  # 输出节点个数 2
        self.d = [0.0] * self.category

    # 初始化 output_layer
    def initial_output_layer(self):
        for i in range(self.category):
            self.output_layer.append([])
            for _ in range(self.output_layer_length):
                self.output_layer[i].append(random.randint(0, 400))

    # som 算法的主要逻辑
    # 计算某个输入样本 与 所有的输出节点之间的距离,存储于 self.d 之中
    def calc_distance(self, a_input):
        self.d = [0.0] * self.category
        for i in range(self.category):
            w = self.output_layer[i]
            # self.d[i] =
            for j in range(len(a_input)):
                self.d[i] += math.pow((a_input[j] - w[j]), 2)  # 就不开根号了

    # 计算一个列表中的最小值 ，并将最小值的索引返回
    def get_min(self, a_list):
        min_index = a_list.index(min(a_list))
        return min_index

    # 将输出节点朝着当前的节点逼近(对node节点的所有维度，这里是x,y分别都进行更新)
    def move(self, a_input, min_output_index):
        for i in range(len(self.output_layer[min_output_index])):
            # 这里不考虑距离winner神经元越远，更新率的衰减问题
            self.output_layer[min_output_index][i] = self.output_layer[min_output_index][i] + self.step_alpha * (a_input[i] - self.output_layer[min_output_index][i])

    # som 逻辑 (一次循环)
    def train(self):
        for a_input in self.input_layer:
            self.calc_distance(a_input)
            min_output_index = self.get_min(self.d)
            self.move(a_input, min_output_index)

    # 循环执行som_train 直到稳定
    def som_looper(self):
        generate = 0
        while self.step_alpha >= 0.0001:  # 这样子会执行167代
            self.train()
            generate += 1
            print("代数:{0} 此时步长:{1} 输出节点:{2}".format(generate, self.step_alpha, self.output_layer))
            self.step_alpha *= self.step_alpha_del_rate  # 步长衰减


if __name__ == '__main__':
    som_zybb = Som_simple_zybb(category)
    som_zybb.initial_output_layer()
    som_zybb.som_looper()