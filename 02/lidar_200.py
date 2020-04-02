import pandas as pd

data = pd.read_csv("../sensor_data/sensor_data_200.txt", delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))
print(data)

print(data["lidar"][0:5])