import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes

mat = np.load("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/GW_CTCF/5K/chr16.totalMats.npy", allow_pickle=True)
mat_df = pd.DataFrame(mat)
mat_df.to_csv('/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/GW_CTCF/5K/chr16.totalMats.txt', sep = "\t", header = False, index = False)

fig = plt.figure()
#定义画布为1*1个划分，并在第1个位置上进行作图
ax = fig.add_subplot(111)
#作图并选择热图的颜色填充风格，这里选择hot
im = ax.imshow(mat_df, cmap=plt.cm.hot_r)
#增加右侧的颜色刻度条
plt.colorbar(im)
#增加标题
plt.title("chr16")
#save
plt.savefig('/public1/xinyu/CohesinProject/SRHiC/predict/chr16_inhouse.png')
#show
plt.show()


plt.figure(dpi=120)
sns.heatmap(data=mat_df)
plt.title('chr16')
plt.savefig('/public1/xinyu/CohesinProject/SRHiC/predict/chr16_inhouse.png')
plt.show()

fig = sns.heatmap(mat_df, annot = False)
heatmap = fig.get_figure()
heatmap.savefig("/public1/xinyu/CohesinProject/SRHiC/predict/chr16_inhouse_heatmap.png", dpi = 400)

