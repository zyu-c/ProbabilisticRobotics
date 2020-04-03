import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../sensor_data/sensor_data_600.txt", delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

#data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align = 'left')
#plt.show()

#data.lidar.plot()
#plt.show()

data["hour"] = [e//10000 for e in data.time]
d = data.groupby("hour")
#d.lidar.mean().plot()
#plt.show()

#d.lidar.get_group(6).hist()
#d.lidar.get_group(14).hist()
#plt.show()

each_hour = {i : d.lidar.get_group(i).value_counts().sort_index() for i in range(24)}
freqs = pd.concat(each_hour, axis = 1)
freqs = freqs.fillna(0)
probs = freqs / len(data)
#print(probs)

import seaborn as sns
#sns.heatmap(probs)
sns.jointplot(data["hour"], data["lidar"], data, kind = "kde")
plt.show()