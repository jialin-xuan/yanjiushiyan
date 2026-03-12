from rdkit import Chem
from rdkit.Chem import BRICS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

def get_mol(smiles):
    """
    从SMILES字符串创建RDKit分子对象
    """
    return Chem.MolFromSmiles(smiles)

def get_smiles(mol):
    """
    将RDKit分子对象转换为SMILES字符串 (Kekule形式)
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def copy_atom(atom):
    """
    复制原子及其属性（形式电荷、原子映射编号）
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    """
    复制并编辑分子（创建可编辑的分子对象，复制原子和键）
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    """
    获取团（Clique）对应的分子片段
    """
    # get the fragment of clique
    Chem.Kekulize(mol)
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    Chem.SanitizeMol(mol)
    return new_mol

def sanitize(mol):
    """
    清洗分子（尝试转换SMILES再转回分子对象以确保有效性）
    """
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def motif_decomp(mol):
    """
    基于BRICS算法的分子Motif分解
    功能：将分子分解为多个子结构（Motifs/Cliques），并处理环结构。
    返回：
        cliques (list of list): 每个元素是一个包含原子索引的列表，代表一个Motif。
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 

    # 合并重叠的cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]


    # 处理环结构 (SSSR: Smallest Set of Smallest Rings)
    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1: 
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
            cliques[i]=[]
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]
    return cliques