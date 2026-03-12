
import joblib
import os

# 数据集名称
dataset = "Davis"
data_path = f"./DataSets/{dataset}.txt"
# 预处理数据路径
protein_path = f"./DataSets/Preprocessed/{dataset}-protein.pkl"
ligand_path = f"./DataSets/Preprocessed/{dataset}-ligand-hi.pkl"

print(f"Checking coverage for {dataset}...")

# Load raw data
# 加载原始数据文件
with open(data_path, "r") as f:
    data_list = f.read().strip().split('\n')

# 提取原始数据中的SMILES和蛋白质序列
raw_smiles = set([item.split(' ')[-3] for item in data_list])
raw_proteins = set([item.split(' ')[-2] for item in data_list])

print(f"Raw data: {len(raw_smiles)} unique ligands, {len(raw_proteins)} unique proteins.")

# Load preprocessed data
# 检查蛋白质数据覆盖率
if os.path.exists(protein_path):
    protein_dict = joblib.load(protein_path)
    print(f"Loaded protein dict: {len(protein_dict)} entries.")
    # 计算缺失的蛋白质
    missing_proteins = raw_proteins - set(protein_dict.keys())
    print(f"Missing proteins: {len(missing_proteins)}")
else:
    print("Protein pkl not found.")
    missing_proteins = raw_proteins

# 检查配体数据覆盖率
if os.path.exists(ligand_path):
    ligand_dict = joblib.load(ligand_path)
    print(f"Loaded ligand dict: {len(ligand_dict)} entries.")
    # 计算缺失的配体
    missing_ligands = raw_smiles - set(ligand_dict.keys())
    print(f"Missing ligands: {len(missing_ligands)}")
else:
    print("Ligand pkl not found.")
    missing_ligands = raw_smiles
