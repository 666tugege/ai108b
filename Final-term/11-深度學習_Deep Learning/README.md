# 深度学习的应用
语音识别
图像应用
大规模图片识别（分类/聚类）
基于图片的搜索服务
图片内容识别（具体图片内容信息）
NP（自然语言）处理	
游戏 机器人等
常见的机器学习领域应用（聚类 分类 回归问题）

# 神经网络之深度神经网络
增多中间层（隐层）的神经网络就叫做深度神经网络（DNN），可以认为深度学习是神经网络的一个发展

# 解决神经网络过拟合的方法
1.交叉验证
训练集（子集）
验证集（评估模型的性能和指标）
测试集（预测）
2.剪枝
每次训练的epoch结束时，将计算的accuracy跟上一次进行比较，如果连续几次accuracy都不再变化，则停止训练
3.正则化	
就是在目标函数上加上一个参数，用来惩罚那些权重很大的向量，称之为一个正则化处理
两种正则化规则：L1（想知道哪一个特征对于最后结果产生较大影响）
L2（如果不在意对于特征分析）

# 用深度学习来做线性回归问题
```python
In[1]:	import numpy as np
		import tensorflow as tf
		import matplotlib.pyplot as plt
In[2]:	# 随机生成一千个点，围绕在y=0.1x+0.3的直线范围内
		num_points = 1000
		vectors_set = []
		for i in range(num_points):
			# 生成一个均值为0.0，方差为0.55的高斯分布
			x1 = np.random.normal(0.0, 0.55)
			# 声明一个y=0.1x+0.3的函数，增加一些抖动(wx+b)
			y1 = x1*0.1 + 0.3 + np.random.normal(0.0, 0.03)
			# 放入vectors_set中
			vectors_set.append([x1,y1])
In[3]:	# 生成一些样本
		x_data = [v[0] for v in vectors_set]
		y_data = [v[1] for v in vectors_set]
		plt.scatter(x_data, y_data)
		plt.show()
In[4]:	# 模拟训练
		# 生成一个一维的w矩阵，取值范围在[-1,1]之间
		W = tf.Variable(tf.random_uniform([1], -1, 1), name='W')
		print(W)
In[5]:	# 生成一个一维的矩阵b，初始值为0
		b = tf.Variable(tf.zeros([1]), name='b')
In[6]:	# 经过计算得出预估值y
		y = W * x_data + b
In[7]:	# 以预估值y和实际值y_data之间的协方差作为损失
		loss = tf.reduce_mean(tf.square(y-y_data), name='loss')
		# 使用梯度下降法来优化参数
		op = tf.train.GradientDescentOptimizer(0.5) # 每次更新的幅度为0.5
		# 训练的过程就是最小化这个误差值
		train = op.minimize(loss, name='train')
		# 建立会话
		sess = tf.Session()
		# 全局变量的初始化
		init = tf.global_variables_initializer()
		# 初始化训练变量
		sess.run(init)
		# 查看初始化的W和b的值
		print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))
In[8]:	# 进行训练，训练30次
		for step in range(30):
			sess.run(train)
		# 输出训练完的W和b的值，看一下能否接近0.1和0.3
		print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))
In[9]:	# 完成线性回归
		plt.scatter(x_data, y_data)
		plt.plot(x_data, sess.run(W)*x_data + sess.run(b), c='r')
		plt.show()
```
