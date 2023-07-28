import cv2
import numpy as np
import matplotlib.pyplot as plt


# 点集由50个点组成
data = np.random.randint(0, 100, (50, 2)).astype(np.float32)
# 为1每个点创建标签 (0:红色, 1:蓝色)
labels = np.random.randint(0, 2, (50, 1)).astype(np.float32)
# 创建要分类的样本点
sample = np.random.randint(0, 100, (1, 2)).astype(np.float32)



# 创建 kNN 分类器
knn = cv2.ml.KNearest_create()
# 训练 kNN 分类器
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
# 找到要分类样本点的 k 个最近邻居
k = 3
ret, results, neighbours, dist = knn.findNearest(sample, k)
# 打印结果
print("result: {}".format(results))
print("neighbours: {}".format(neighbours))
print("distance: {}".format(dist))

# 可视化
fig = plt.figure(figsize=(8, 6))
red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')
blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')
plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')
plt.show()
