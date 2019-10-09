#-*- coding: utf-8 -*-

def GM11(x0): #自定义灰色预测函数
  import numpy as np
  x1 = x0.cumsum() #1-AGO序列
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std()
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率

'''
斜杠：/，用来表示除法；
反斜杠：\ ，与中文中的顿号是一个键；
在Windows中表示文件目录时应该用反斜杠\，但是不同的编译器对正反斜杠/\（写在一起像一个“八”）的解释不同，在Windows下的jupyter notebook仍然要用斜杠/.

cumsum():计算数组各行的累加值
ones_like(a, dtype=None, order='K', subok=True) [source]，返回与指定数组a具有相同形状和数据类型的数组，并且数组中的值都为1。
append()：添加一个记录或者多个新记录，所以-z1, np.ones_like(z1)都是要往B列表中添加的内容
dot(a, b, out=None)，计算两个数组的乘积。对于二维数组来说，dot()计算的结果就相当于矩阵乘法；对于一维数组，它计算的是两个向量的点积；对于N维数组，它是a的最后一维和b的倒数第二维积的和：dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
点积，即元素对应相乘。

np.linalg.inv()：矩阵求逆
np.linalg.det()：矩阵求行列式（标量）
'''