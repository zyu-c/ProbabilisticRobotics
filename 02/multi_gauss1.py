import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../sensor_data/sensor_data_700.txt", delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

d = data[(data["time"] < 160000) & (data["time"] >= 120000)]
d = d.loc[:, ["ir", "lidar"]]

#sns.jointplot(d["ir"], d["lidar"], d, kind = "kde")
#plt.show()

#print("光センサの計測値の分散 : ", d.ir.var())
#print("LiDARの計測値の分散 : ", d.lidar.var())

diff_ir = d.ir - d.ir.mean()
diff_lidar = d.lidar - d.lidar.mean()
a = diff_ir * diff_lidar
#print("共分散 : ", sum(a) / (len(d) - 1))
#print(d.mean())

#print(d.cov())

from scipy.stats import multivariate_normal
irlidar = multivariate_normal(mean = d.mean().values.T, cov = d.cov().values)

import numpy as np
x, y = np.mgrid[0:40, 710:750]
pos = np.empty(x.shape + (2, ))
pos[:, :, 0] = x
pos[:, :, 1] = y
cont = plt.contour(x, y, irlidar.pdf(pos))
cont.clabel(fmt = '%1.1e')
plt.show()