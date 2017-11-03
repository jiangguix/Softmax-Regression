# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.preprocessing import PolynomialFeatures

def soft_max(X, Y, K, alpha, lamda):
	n = len(X[0])
	w = np.zeros((K,n))
	wnew = np.zeros((K,n))
	for times in range(1000):
		for i in range(len(X)):
			x = X[i]
			for k in range(K):
				y = 0
				if Y[i] == k:
					y = 1
				p = predict(w,x,k)
				g = (y-p)*x
				wnew[k] = w[k] + (alpha*g + lamda*w[k])
			w = wnew.copy()
	return w

def predict(w,x,k):
	numerator = np.exp(np.dot(w[k],x))
	denominator = sum(np.exp(np.dot(w,x)))
	return numerator/denominator

def model(w,x,K):
	cat = []
	p = [0,0,0]
	for i in range(len(x[:,0])):
		for k in range(K):
			p[k] = predict(w,x[i,:],k)
		cat.append(p.index(max(p)))
	return cat

def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a

data = pd.read_csv('iris.data', header=None)
columns = np.array([u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度', u'类型'])
data.rename(columns=dict(zip(np.arange(5), columns)), inplace=True)
data.drop(columns[:2],axis=1,inplace=True)
data[u'类型'] = pd.Categorical(data[u'类型']).codes

x = data[columns[2:-1]].values
y = data[columns[-1]].values
poly = PolynomialFeatures(2)
x_p = poly.fit_transform(x)

N, M = 200, 200     # 横纵各采样多少个值
x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())   # 第0列的范围
x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())   # 第1列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)

x_show = np.stack((x1.flat, x2.flat), axis=1)   # 测试点
x_show_p = poly.fit_transform(x_show)

K = 3
w = soft_max(x_p,y,K,0.0005,0.0000005)
print w
y_hat = np.array(model(w,x_show_p,K))

y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
print y_hat

cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], s=30, c=y, edgecolors='k', cmap=cm_dark)  # 样本的显示
x1_label, x2_label = columns[2],columns[3]
plt.xlabel(x1_label, fontsize=14)
plt.ylabel(x2_label, fontsize=14)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(b=True, ls=':')
# 画各种图
# a = mpl.patches.Wedge(((x1_min+x1_max)/2, (x2_min+x2_max)/2), 1.5, 0, 360, width=0.5, alpha=0.5, color='r')
# plt.gca().add_patch(a)
patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
          mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
          mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
plt.legend(handles=patchs, fancybox=True, framealpha=0.8, loc='lower right')
plt.title(u'鸢尾花softmax回归分类', fontsize=17)
plt.show()