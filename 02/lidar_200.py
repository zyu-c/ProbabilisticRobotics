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

#data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), color = "orange", align = 'left')
#plt.vlines(mean1, ymin = 0, ymax = 5000, color = "red")
#plt.show()

zs = data["lidar"].values
mean = sum(zs) / len(zs)
diff_square = [(z - mean) ** 2 for z in zs]

sampling_var = sum(diff_square) / (len(zs))
unbiased_var = sum(diff_square) / (len(zs) - 1)
#print(sampling_var)
#print(unbiased_var)

pandas_sampling_var = data["lidar"].var(ddof=False)
pandas_default_var = data["lidar"].var()
#print(pandas_sampling_var)
#print(pandas_default_var)

import numpy as np

numpy_default_var = np.var(data["lidar"])
numpy_unbiased_var = np.var(data["lidar"], ddof = 1)
#print(numpy_default_var)
#print(numpy_unbiased_var)

import math

stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)
pandas_stddev = data["lidar"].std()

#print(stddev1)
#print(stddev2)
#print(pandas_stddev)

freqs = pd.DataFrame(data["lidar"].value_counts())
freqs["probs"] = freqs["lidar"] / len(data["lidar"])
print(freqs.transpose())
print(sum(freqs["probs"]))

freqs["probs"].sort_index().plot.bar()
plt.show()

def drawing():
    return freqs.sample(n = 1, weights = "probs").index[0]

print(drawing())

samples = [drawing() for i in range(1000)]

simulated = pd.DataFrame(samples, columns = ["lidar"])
p = simulated["lidar"]
p.hist(bins = max(p) - min(p), color = "orange", align = 'left')
plt.show()