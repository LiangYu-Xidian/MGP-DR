import pandas as pd
ppi = pd.read_csv("PPI.csv")

self_self = []
drop_list = []

for i in range(ppi.shape[0]):
    if ppi.iloc[i][0] == ppi.iloc[i][1]:
        self_self.append(ppi.iloc[i][0])
        drop_list.append(i)

no_self_ppi = ppi.drop(drop_list)
p = list(set(no_self_ppi["Protein_A_Entrez_ID"].values.tolist() + no_self_ppi["Protein_B_Entrez_ID"].values.tolist()))

self_self = list(set(self_self))

print(self_self)
print(p)

sss = []
for ss in self_self:
    if ss not in p:
        sss.append(ss)
        print(ss)
print(sss)

