import wrapper
import pandas as pd

# dt_in_S_AB_small = pd.read_csv("D:/TASK2/Data/S_AB/temp/dt_in_S_AB_small.csv")

file_name = "ppi.sif"
network = wrapper.get_network(file_name, only_lcc = True)




# 计算单对
d1 = pd.read_csv("./Data/case_study/T_B/Telmisartan2.csv")
d2 = pd.read_csv("./Data/case_study/T_B/Hypertension2.csv")

t1 = d1["gene_name"].values.tolist();
t2 = d2["gene_name"].values.tolist();
nodes_from = []
nodes_to = []
for t in t1:
    t = str(t)
    if network.has_node(t):
        nodes_from.append(str(t))
for tt in t2:
    tt = str(tt)
    if network.has_node(tt):
        nodes_to.append(str(tt))

print(nodes_from)
print(nodes_to)

#计算 S 和 Z
if len(nodes_from) == 0 or len(nodes_to) == 0:
    z = 99999999999
else:
    d, z, (mean, sd) = wrapper.calculate_proximity(network, nodes_from, nodes_to, min_bin_size = 2)

print(z);





# 计算文件中的

# for k in range(1):
#     print("File"+str(k+7))
#     S_AB_small = pd.read_csv("D:/TASK2/Data/S_AB/S_AB_small/S_AB_small_" + str(k+7) + ".csv")
#     S_AB_score = []
#     for i in range(S_AB_small.shape[0]):
#         d1 = S_AB_small.iloc[i][0]
#         d2 = S_AB_small.iloc[i][1]
#         for j in range(dt_in_S_AB_small.shape[0]):
#             if d1 == dt_in_S_AB_small.iloc[j][0]:
#                 t1 = dt_in_S_AB_small.iloc[j][1][:-1]
#             if d2 == dt_in_S_AB_small.iloc[j][0]:
#                 t2 = dt_in_S_AB_small.iloc[j][1][:-1]
#         t1 = t1.split(";")
#         t2 = t2.split(";")
#         nodes_from = []
#         nodes_to = []
#         for t in t1:
#             if network.has_node(t):
#                 nodes_from.append(str(t))
#         for tt in t2:
#             if network.has_node(tt):
#                 nodes_to.append(str(tt))
#         print(i)
#         print(d1,t1,nodes_from)
#         print(d2,t2,nodes_to)


#         if len(nodes_from) == 0 or len(nodes_to) == 0:
#             d = 99999999999
#             S_AB_score.append(d)
#         else:
#             d = wrapper.calculate_separation_proximity(network, nodes_from, nodes_to, min_bin_size = 2)
#             S_AB_score.append(d)
#         print(d)

#     S_AB_small["S_AB"] = S_AB_score
#     S_AB_small.to_csv("D:/TASK2/Data/S_AB/S_AB_small/S_AB_small_" + str(k+7) + "_new.csv",index=False)

