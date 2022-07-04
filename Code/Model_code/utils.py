
# coding=utf-8

from numpy.lib.function_base import delete
from pandas.core.frame import DataFrame
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
import openbabel as ob
import pandas as pd


def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile


def ifconvert(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smile))
    if mol is None:
        print(smile + ' is not valid ' )
        return False
    else:
        return True


def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=True):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))

    if mol is None:
        print(smiles + ' is not valid ' )
        return "none","none"

    else:

        if explicit_hydrogens:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)

        if canonical_atom_order:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        num_atoms = mol.GetNumAtoms()
        atoms_list = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atoms_list.append(atom.GetSymbol())

        adjoin_matrix = np.eye(num_atoms)
        # Add edges
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            adjoin_matrix[u,v] = 1.0
            adjoin_matrix[v,u] = 1.0
        return atoms_list,adjoin_matrix
    


def count_atom(chems):
    atoms_dict = {}
    all_atom = []
    for i,smiles in enumerate(chems['SMILES']):
        atoms,_ = smiles2adjoin(smiles)
        all_atom = all_atom + atoms
    for atom in all_atom:
        if atom in atoms_dict.keys():
            atoms_dict[atom]+=1
        else:
            atoms_dict[atom] = 1
    atoms_dict = sorted(atoms_dict.items(),key=lambda y:y[1],reverse=True)
    return atoms_dict




# ===========================================================================================
'''
test
'''
smile = "CCCC1=NC2=C(N1CC3=CC=C(C=C3)C4=CC=CC=C4C(=O)O)C=C(C=C2C)C5=NC6=CC=CC=C6N5C"
atom, adj = smiles2adjoin(smile)

print(atom)
print(len(atom))
print(adj) 
# print(ifconvert(smile))
print(smile)

#-----------------------------------------------------------

# drugs = pd.read_csv("D:/TASK2/Data/S_AB/S_AB.csv",error_bad_lines=False)
# delete_list = []

# for i in range(drugs.shape[0]):
#     if ifconvert(drugs.iloc[i][2]) is False:
#         delete_list.append(i)
             
# print(delete_list)
# filtered_drugs = drugs.drop(delete_list)
# print(filtered_drugs.shape)
# filtered_drugs.to_csv("D:/TASK2/Data/S_AB/S_AB_delete.csv",index=None)

#-----------------------------------------------------------

# drugs = pd.read_csv("D:/TASK2/Data/S_AB/S_AB.csv",error_bad_lines=False)

# d_list = []
# delete_list = []

# for i in range(drugs.shape[0]):
#     print(i)
#     d1 = drugs.iloc[i][2].split("?")[0]
#     d2 = drugs.iloc[i][2].split("?")[0]
#     if ifconvert(d1) is False or ifconvert(d2) is False:
#         delete_list.append(i)
             
# print(delete_list)
# filtered_drugs = drugs.drop(delete_list)
# print(filtered_drugs.shape)
# filtered_drugs.to_csv("D:/TASK2/Data/S_AB/S_AB_delete.csv",index=None)

# ===========================================================================================



# ===========================================================================================
'''
delete drug which can not be converted
'''
# drugs = pd.read_csv("D:/TASK2/Data/DDI/DeepDDI/DEEPDDI.csv")
# smile = pd.read_csv("D:/TASK2/Data/DDI/DeepDDI/smiles.csv")
# Smiles = []

# for i in range(drugs.shape[0]):
#     drug_2 = drugs.iloc[i,4].split("?")
#     if ifconvert(drug_2[0]) is False or ifconvert(drug_2[1]) is False:
#         print("///////////////////")
#         for j in range(smile.shape[0]):
#             if smile.iloc[j][0] == drugs.iloc[i][0]:
#                 d1 = smile.iloc[j][1]
#             if smile.iloc[j][0] == drugs.iloc[i][1]:
#                 d2 = smile.iloc[j][1]
#         s = d1+"?"+d2
#         Smiles.append(s)
#     else:
#         Smiles.append(drugs.iloc[i,4])

# drugs["Smiles"] = Smiles
# drugs.to_csv("D:/TASK2/Data/DDI/DeepDDI/DEEPDDIiiii.csv",index=None)

# ==========================================================================================



# ==========================================================================================
'''
count atom type of all drug
'''
# drugs = pd.read_csv("D:/TASK2/Data/Drugbank/SMILE/filtered_approved.csv")
# atoms = count_atom(drugs)
# print(atoms)
# print(type(atoms))
# DataFrame(atoms).to_csv("D:/TASK2/Data/Drugbank/SMILE/count_results2.csv")

# ==========================================================================================



# ==========================================================================================
'''
convert SMILEs to drug name
'''
# import requests

# smiles = 'CN(C)c1ccc2nc3ccc(N(C)C)cc3[s+]c2c1.[Cl-]'


# CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


# def smiles_to_iupac(smiles):
#     rep = "iupac_name"
#     url = CACTUS.format(smiles, rep)
#     response = requests.get(url)
#     response.raise_for_status()
#     return response.text


# print(smiles_to_iupac(smiles))
# print(" ")


# import pubchempy


# compounds = pubchempy.get_compounds(smiles, namespace='smiles')
# match = compounds[0]
# print(match.iupac_name)
# ==========================================================================================




