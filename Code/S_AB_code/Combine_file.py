import pandas as pd

S_AB = pd.DataFrame()

for k in range(14):
    print("File"+str(k))
    S_AB_small = pd.read_csv("D:/TASK2/Data/S_AB/S_AB_small/S_AB_small_" + str(k) + "_new.csv")
    S_AB = S_AB.append(S_AB_small)
    print(S_AB)

S_AB = S_AB.reset_index(drop=True)

print(S_AB)


drop_list = []

for i in range(S_AB.shape[0]):
    print(i)
    if S_AB.iloc[i][3] == 99999999999:
        drop_list.append(i)
print(drop_list)

S_AB = S_AB.drop(drop_list)

    
S_AB.to_csv("D:/TASK2/Data/S_AB/S_AB.csv",index=False)