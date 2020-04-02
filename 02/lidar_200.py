import pandas as pd

data = pd.read_csv("../sensor_data/sensor_data_200.txt", delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))
#print(data)

#print(data["lidar"][0:5])

import matplotlib.pyplot as plt

#data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align = 'left')
#plt.show()

mean1 = sum(data["lidar"].values) / len(data["lidar"].values)
mean2 = data["lidar"].mean()
#print(mean1, mean2)

data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), color = "orange", align = 'left')
plt.vlines(mean1, ymin = 0, ymax = 5000, color = "red")
plt.show()