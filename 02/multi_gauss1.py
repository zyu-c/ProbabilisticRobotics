import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../sensor_data/sensor_data_700.txt", delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

d = data[(data["time"] < 160000) & (data["time"] >= 120000)]
d = d.loc[:, ["ir", "lidar"]]

sns.jointplot(d["ir"], d["lidar"], d, kind = "kde")
plt.show()