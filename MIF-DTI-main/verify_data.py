
import joblib
import os

try:
    # 加载预处理后的蛋白质和配体数据
    p = joblib.load('./DataSets/Preprocessed/Davis-protein-new.pkl')
    l = joblib.load('./DataSets/Preprocessed/Davis-ligand-hi-new.pkl')
    print(f"Proteins: {len(p)}")
    print(f"Ligands: {len(l)}")
    
    # 预期的数据数量（针对Davis数据集）
    expected_proteins = 379
    expected_ligands = 68
    
    # 验证数据数量是否符合预期
    if len(p) >= expected_proteins and len(l) >= expected_ligands:
        print("Data verification passed!")
    else:
        print(f"Data incomplete. Expected {expected_proteins} proteins and {expected_ligands} ligands.")
except Exception as e:
    print(f"Error loading data: {e}")
