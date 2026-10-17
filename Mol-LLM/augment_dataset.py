from datasets import load_from_disk
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import re
import random
from ogb.utils.features import (
    allowable_features,
    atom_to_feature_vector,
    bond_to_feature_vector,
)
import selfies as sf
from tqdm import tqdm
from collections import Counter
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
)

# [1, 42, 44, 46, 103, 125, 162, 166] -> Noting to convert
# [2, 49, 59, 65, 90, 91, 101, 113, 120, 121, 128, 129, 135, 137, 144, 165] -> Error occured when converting
# [116, 118, 147]
smartsPatts = {
    # 1:('?',0), # ISOTOPE
    # 2:('[#104,#105,#106,#107,#106,#109,#110,#111,#112]',0),  # atomic num >103 Not complete
    2: ("[#104]", 0),  # limit the above def'n since the RDKit only accepts up to #104
    3: ("[#32,#33,#34,#50,#51,#52,#82,#83,#84]", 0),  # Group IVa,Va,VIa Rows 4-6
    4: ("[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]", 0),  # actinide
    5: ("[Sc,Ti,Y,Zr,Hf]", 0),  # Group IIIB,IVB (Sc...)
    6: ("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]", 0),  # Lanthanide
    7: ("[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]", 0),  # Group VB,VIB,VIIB
    8: ("[!#6;!#1]1~*~*~*~1", 0),  # QAAA@1
    9: ("[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]", 0),  # Group VIII (Fe...)
    10: ("[Be,Mg,Ca,Sr,Ba,Ra]", 0),  # Group IIa (Alkaline earth)
    11: ("*1~*~*~*~1", 0),  # 4M Ring
    12: ("[Cu,Zn,Ag,Cd,Au,Hg]", 0),  # Group IB,IIB (Cu..)
    13: ("[#8]~[#7](~[#6])~[#6]", 0),  # ON(C)C
    14: ("[#16]-[#16]", 0),  # S-S
    15: ("[#8]~[#6](~[#8])~[#8]", 0),  # OC(O)O
    16: ("[!#6;!#1]1~*~*~1", 0),  # QAA@1
    17: ("[#6]#[#6]", 0),  # CTC
    18: ("[#5,#13,#31,#49,#81]", 0),  # Group IIIA (B...)
    19: ("*1~*~*~*~*~*~*~1", 0),  # 7M Ring
    20: ("[#14]", 0),  # Si
    21: ("[#6]=[#6](~[!#6;!#1])~[!#6;!#1]", 0),  # C=C(Q)Q
    22: ("*1~*~*~1", 0),  # 3M Ring
    23: ("[#7]~[#6](~[#8])~[#8]", 0),  # NC(O)O
    24: ("[#7]-[#8]", 0),  # N-O
    25: ("[#7]~[#6](~[#7])~[#7]", 0),  # NC(N)N
    26: ("[#6]=;@[#6](@*)@*", 0),  # C$=C($A)$A
    27: ("[I]", 0),  # I
    28: ("[!#6;!#1]~[CH2]~[!#6;!#1]", 0),  # QCH2Q
    29: ("[#15]", 0),  # P
    30: ("[#6]~[!#6;!#1](~[#6])(~[#6])~*", 0),  # CQ(C)(C)A
    31: ("[!#6;!#1]~[F,Cl,Br,I]", 0),  # QX
    32: ("[#6]~[#16]~[#7]", 0),  # CSN
    33: ("[#7]~[#16]", 0),  # NS
    34: ("[CH2]=*", 0),  # CH2=A
    35: ("[Li,Na,K,Rb,Cs,Fr]", 0),  # Group IA (Alkali Metal)
    36: ("[#16R]", 0),  # S Heterocycle
    37: ("[#7]~[#6](~[#8])~[#7]", 0),  # NC(O)N
    38: ("[#7]~[#6](~[#6])~[#7]", 0),  # NC(C)N
    39: ("[#8]~[#16](~[#8])~[#8]", 0),  # OS(O)O
    40: ("[#16]-[#8]", 0),  # S-O
    41: ("[#6]#[#7]", 0),  # CTN
    # 42:('F',0), # F
    43: ("[!#6;!#1;!H0]~*~[!#6;!#1;!H0]", 0),  # QHAQH
    # 44:('?',0), # OTHER
    45: ("[#6]=[#6]~[#7]", 0),  # C=CN
    # 46:('Br',0), # BR
    47: ("[#16]~*~[#7]", 0),  # SAN
    48: ("[#8]~[!#6;!#1](~[#8])(~[#8])", 0),  # OQ(O)O
    # 49:('[!+0]',0), # CHARGE
    50: ("[#6]=[#6](~[#6])~[#6]", 0),  # C=C(C)C
    51: ("[#6]~[#16]~[#8]", 0),  # CSO
    52: ("[#7]~[#7]", 0),  # NN
    53: ("[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]", 0),  # QHAAAQH
    54: ("[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]", 0),  # QHAAQH
    55: ("[#8]~[#16]~[#8]", 0),  # OSO
    56: ("[#8]~[#7](~[#8])~[#6]", 0),  # ON(O)C
    57: ("[#8R]", 0),  # O Heterocycle
    58: ("[!#6;!#1]~[#16]~[!#6;!#1]", 0),  # QSQ
    # 59:('[#16]!:*:*',0), # Snot%A%A
    60: ("[#16]=[#8]", 0),  # S=O
    61: ("*~[#16](~*)~*", 0),  # AS(A)A
    62: ("*@*!@*@*", 0),  # A$!A$A
    63: ("[#7]=[#8]", 0),  # N=O
    64: ("*@*!@[#16]", 0),  # A$A!S
    # 65:('c:n',0), # C%N
    66: ("[#6]~[#6](~[#6])(~[#6])~*", 0),  # CC(C)(C)A
    67: ("[!#6;!#1]~[#16]", 0),  # QS
    68: ("[!#6;!#1;!H0]~[!#6;!#1;!H0]", 0),  # QHQH (&...) SPEC Incomplete
    69: ("[!#6;!#1]~[!#6;!#1;!H0]", 0),  # QQH
    70: ("[!#6;!#1]~[#7]~[!#6;!#1]", 0),  # QNQ
    71: ("[#7]~[#8]", 0),  # NO
    72: ("[#8]~*~*~[#8]", 0),  # OAAO
    73: ("[#16]=*", 0),  # S=A
    74: ("[CH3]~*~[CH3]", 0),  # CH3ACH3
    75: ("*!@[#7]@*", 0),  # A!N$A
    76: ("[#6]=[#6](~*)~*", 0),  # C=C(A)A
    77: ("[#7]~*~[#7]", 0),  # NAN
    78: ("[#6]=[#7]", 0),  # C=N
    79: ("[#7]~*~*~[#7]", 0),  # NAAN
    80: ("[#7]~*~*~*~[#7]", 0),  # NAAAN
    81: ("[#16]~*(~*)~*", 0),  # SA(A)A
    82: ("*~[CH2]~[!#6;!#1;!H0]", 0),  # ACH2QH
    83: ("[!#6;!#1]1~*~*~*~*~1", 0),  # QAAAA@1
    84: ("[NH2]", 0),  # NH2
    85: ("[#6]~[#7](~[#6])~[#6]", 0),  # CN(C)C
    86: ("[C;H2,H3][!#6;!#1][C;H2,H3]", 0),  # CH2QCH2
    87: ("[F,Cl,Br,I]!@*@*", 0),  # X!A$A
    88: ("[#16]", 0),  # S
    89: ("[#8]~*~*~*~[#8]", 0),  # OAAAO
    # 90:('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]',0), # QHAACH2A
    # 91:('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]',0), # QHAAACH2A
    92: ("[#8]~[#6](~[#7])~[#6]", 0),  # OC(N)C
    93: ("[!#6;!#1]~[CH3]", 0),  # QCH3
    94: ("[!#6;!#1]~[#7]", 0),  # QN
    95: ("[#7]~*~*~[#8]", 0),  # NAAO
    96: ("*1~*~*~*~*~1", 0),  # 5 M ring
    97: ("[#7]~*~*~*~[#8]", 0),  # NAAAO
    98: ("[!#6;!#1]1~*~*~*~*~*~1", 0),  # QAAAAA@1
    99: ("[#6]=[#6]", 0),  # C=C
    100: ("*~[CH2]~[#7]", 0),  # ACH2N
    # 101:('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]',0), # 8M Ring or larger. This only handles up to ring sizes of 14
    102: ("[!#6;!#1]~[#8]", 0),  # QO
    # 103:('Cl',0), # CL
    104: ("[!#6;!#1;!H0]~*~[CH2]~*", 0),  # QHACH2A
    105: ("*@*(@*)@*", 0),  # A$A($A)$A
    106: ("[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]", 0),  # QA(Q)Q
    107: ("[F,Cl,Br,I]~*(~*)~*", 0),  # XA(A)A
    108: ("[CH3]~*~*~*~[CH2]~*", 0),  # CH3AAACH2A
    109: ("*~[CH2]~[#8]", 0),  # ACH2O
    110: ("[#7]~[#6]~[#8]", 0),  # NCO
    111: ("[#7]~*~[CH2]~*", 0),  # NACH2A
    112: ("*~*(~*)(~*)~*", 0),  # AA(A)(A)A
    # 113:('[#8]!:*:*',0), # Onot%A%A
    114: ("[CH3]~[CH2]~*", 0),  # CH3CH2A
    115: ("[CH3]~*~[CH2]~*", 0),  # CH3ACH2A
    # 116:('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]',0), # CH3AACH2A
    117: ("[#7]~*~[#8]", 0),  # NAO
    # 118:('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]',1), # ACH2CH2A > 1
    119: ("[#7]=*", 0),  # N=A
    # 120:('[!#6;R]',1), # Heterocyclic atom > 1 (&...) Spec Incomplete
    # 121:('[#7;R]',0), # N Heterocycle
    122: ("*~[#7](~*)~*", 0),  # AN(A)A
    123: ("[#8]~[#6]~[#8]", 0),  # OCO
    124: ("[!#6;!#1]~[!#6;!#1]", 0),  # QQ
    # 125:('?',0), # Aromatic Ring > 1
    126: ("*!@[#8]!@*", 0),  # A!O!A
    127: ("*@*!@[#8]", 1),  # A$A!O > 1 (&...) Spec Incomplete
    # 128:('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]',0), # ACH2AAACH2A
    # 129:('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]',0), # ACH2AACH2A
    130: ("[!#6;!#1]~[!#6;!#1]", 1),  # QQ > 1 (&...)  Spec Incomplete
    131: ("[!#6;!#1;!H0]", 1),  # QH > 1
    132: ("[#8]~*~[CH2]~*", 0),  # OACH2A
    133: ("*@*!@[#7]", 0),  # A$A!N
    134: ("[F,Cl,Br,I]", 0),  # X (HALOGEN)
    # 135:('[#7]!:*:*',0), # Nnot%A%A
    136: ("[#8]=*", 1),  # O=A>1
    # 137:('[!C;!c;R]',0), # Heterocycle
    138: ("[!#6;!#1]~[CH2]~*", 1),  # QCH2A>1 (&...) Spec Incomplete
    139: ("[O;!H0]", 0),  # OH
    140: ("[#8]", 3),  # O > 3 (&...) Spec Incomplete
    141: ("[CH3]", 2),  # CH3 > 2  (&...) Spec Incomplete
    142: ("[#7]", 1),  # N > 1
    143: ("*@*!@[#8]", 0),  # A$A!O
    # 144:('*!:*:*!:*',0), # Anot%A%Anot%A
    145: ("*1~*~*~*~*~*~1", 1),  # 6M ring > 1
    146: ("[#8]", 2),  # O > 2
    # 147:('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]',0), # ACH2CH2A
    148: ("*~[!#6;!#1](~*)~*", 0),  # AQ(A)A
    149: ("[C;H3,H4]", 1),  # CH3 > 1
    150: ("*!@*@*!@*", 0),  # A!A$A!A
    151: ("[#7;!H0]", 0),  # NH
    152: ("[#8]~[#6](~[#6])~[#6]", 0),  # OC(C)C
    153: ("[!#6;!#1]~[CH2]~*", 0),  # QCH2A
    154: ("[#6]=[#8]", 0),  # C=O
    155: ("*!@[CH2]!@*", 0),  # A!CH2!A
    156: ("[#7]~*(~*)~*", 0),  # NA(A)A
    157: ("[#6]-[#8]", 0),  # C-O
    158: ("[#6]-[#7]", 0),  # C-N
    159: ("[#8]", 1),  # O>1
    160: ("[C;H3,H4]", 0),  # CH3
    161: ("[#7]", 0),  # N
    # 162:('a',0), # Aromatic
    163: ("*1~*~*~*~*~*~1", 0),  # 6M Ring
    164: ("[#8]", 0),  # O
    # 165:('[R]',0), # Ring
    # 166:('?',0), # Fragments  FIX: this can't be done in SMARTS
}

# Define atom groups for specific exclusions and selections
atom_groups = {
    "!#6;!#1": [  # Exclude #6 (carbon) and #1 (hydrogen)
        "#2",
        "#3",
        "#4",
        "#5",
        "#7",
        "#8",
        "#10",
    ],
    "!#6;!#1;!H0": [  # Exclude #6 (carbon) and #1 (hydrogen)
        "#2",
        "#3",
        "#4",
        "#5",
        "#7",
        "#8",
        "#10",
    ],
    "*": ["#6"],
    "~": ["-"],
    "R": ["#6"],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        safe_index(
            allowable_features["possible_chirality_list"], str(atom.GetChiralTag())
        ),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def mol2graph(mol):
    """
    Converts mol to graph Data object.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        dict: Graph representation with edge_index, edge_feat, node_feat, and num_nodes.
    """
    # Atoms
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


def graph2mol(graph):
    """
    Converts a graph representation of a molecule to an RDKit molecule.

    Args:
        graph (dict): A graph representation with 'edge_index', 'edge_feat', 'node_feat', and 'num_nodes'.

    Returns:
        rdkit.Chem.Mol: The constructed RDKit molecule.
    """
    mol = Chem.RWMol()

    # Add atoms to the molecule
    for atom_features in graph["node_feat"]:
        atomic_number = int(
            atom_features[0]
        )  # Assume atomic number is the first feature
        mol.AddAtom(Chem.Atom(atomic_number))

    # Add bonds to the molecule
    edge_index = graph["edge_index"]
    edge_feat = graph["edge_feat"]

    # Iterate through half of the edges (since they are bidirectional in the graph representation)
    for i in range(edge_index.shape[1] // 2):
        src, tgt = edge_index[:, i]
        bond_type = Chem.rdchem.BondType.SINGLE  # Default bond type

        # Determine bond type from edge features
        if edge_feat[i][0] == 1:
            bond_type = Chem.rdchem.BondType.SINGLE
        elif edge_feat[i][1] == 1:
            bond_type = Chem.rdchem.BondType.DOUBLE
        elif edge_feat[i][2] == 1:
            bond_type = Chem.rdchem.BondType.TRIPLE
        elif edge_feat[i][0] == 3:
            bond_type = Chem.rdchem.BondType.AROMATIC

        # Add bond only if it doesn't already exist
        if not mol.GetBondBetweenAtoms(int(src), int(tgt)):
            mol.AddBond(int(src), int(tgt), bond_type)

    return mol


# Example visualization function
def mol_visualize(mol):
    """
    Visualizes a molecule using RDKit.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object to visualize.

    Returns:
        None: Displays the molecule visualization.
    """
    from rdkit.Chem import Draw

    img = Draw.MolToImage(mol)
    img.show()


def make_specific_smarts(smarts):
    """
    Handles SMARTS patterns with multiple atom choices and operators,
    including exclusions like [!#6;!#1].

    Args:
        smarts (str): SMARTS pattern to process.

    Returns:
        str: A new SMARTS pattern with random atom choices selected.
    """
    # Regular expression to find atom groups and operators
    pattern = r"(\[\$\(.*?\)\]|[~=#:@\-\+*]|1|\[[^]]+\])"
    parts = re.findall(
        pattern, smarts
    )  # Extract groups, operators, and nested patterns

    result = []
    for part in parts:
        if part.startswith("[$(") and part.endswith(")]"):  # Handle nested SMARTS
            # Split into individual options within the [$(...),$(...)] structure
            nested_options = part[2:-1].split(
                "),$("
            )  # Remove `[$(` and `)]`, then split
            selected_option = random.choice(
                nested_options
            )  # Always select the first option
            randomized_nested = make_specific_smarts(
                selected_option
            )  # Recursively process selected SMARTS
            result.append(randomized_nested)  # No need to wrap with [$()]
        elif part.startswith("[") and part.endswith("]"):  # Check for atom group
            atom_group = part.strip("[]")  # Remove brackets
            if atom_group in atom_groups:  # Handle special exclusions
                selected_atom = random.choice(atom_groups[atom_group])
                # if atom_group == "!#6;!#1;!H0":  # Append [#1] for this specific case
                #    selected_atom += "]-[#1"
                result.append(f"[{selected_atom}]")
            elif (
                ";" in atom_group and "," in atom_group
            ):  # Handle patterns like [C;H3,H4]
                # Split into atom base and subgroups
                base, subgroups = atom_group.split(";")
                subgroup_choices = subgroups.split(",")
                selected_subgroup = random.choice(
                    subgroup_choices
                )  # Randomly select a subgroup
                result.append(f"[{base}{selected_subgroup}]")
            elif "," in atom_group:  # Handle multiple atom choices
                atom_choices = atom_group.split(",")
                selected_atom = random.choice(atom_choices)
                result.append(f"[{selected_atom}]")
            else:
                # No special handling needed, return as is
                result.append(part)
        elif part == "*":  # Handle wildcard '*'
            selected_atom = random.choice(atom_groups["*"])
            result.append(f"[{selected_atom}]")
        elif part == "~" or part == "@":  # Handle bond operator '~'
            selected_operator = random.choice(atom_groups["~"])
            result.append(selected_operator)
        else:
            # Append operators directly
            result.append(part)

    return "".join(result)


def add_replacement_mol(editable_mol, replacement_mol, attach_idx):
    """
    Adds all atoms and bonds from the replacement molecule to the editable molecule,
    and connects them with at least one bond.

    Args:
        editable_mol (rdkit.Chem.RWMol): The base molecule to modify.
        replacement_mol (rdkit.Chem.RWMol): The molecule to add.

    Returns:
        None
    """
    # Map for tracking new atom indices
    atom_map = {}

    # Add all atoms from the replacement molecule
    for atom in replacement_mol.GetAtoms():
        new_idx = editable_mol.AddAtom(atom)
        atom_map[atom.GetIdx()] = new_idx

    # Add all bonds from the replacement molecule
    for bond in replacement_mol.GetBonds():
        start_idx = atom_map[bond.GetBeginAtomIdx()]
        end_idx = atom_map[bond.GetEndAtomIdx()]
        editable_mol.AddBond(start_idx, end_idx, bond.GetBondType())

    # Add a single bond between the selected atom and one atom in the replacement molecule
    editable_mol.AddBond(attach_idx, atom_map[0], Chem.BondType.SINGLE)

    # Update properties to avoid implicit valence issues
    editable_mol.UpdatePropertyCache(strict=False)
    return editable_mol


def extract_and_modify(selfies, replace_ratio=0.1, modify_num=0):
    # Convert SELFIES to SMILES
    smiles = sf.decoder(selfies)
    if not smiles:
        raise ValueError(f"Invalid SELFIES, SMILES provided: {selfies}, {smiles}")

    # Convert SMILES to RDKit Molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to convert smiles: {smiles} to a valid molecule.")

    # Generate MACCS Keys
    maccs_keys = MACCSkeys.GenMACCSKeys(mol)

    # Find activated MACCS keys and their corresponding substructures
    active_keys_and_substructures = []
    for key, (smarts, _) in smartsPatts.items():
        if maccs_keys.GetBit(key):  # Check if the MACCS key is active
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    substructures = [
                        Chem.MolFragmentToSmiles(mol, match) for match in matches
                    ]
                    atom_indices = [
                        [idx for idx in match] for match in matches
                    ]  # Original SMILES indices
                    active_keys_and_substructures.append(
                        {
                            "key": key,
                            "smarts": smarts,
                            "substructures": substructures,
                            "atom_indices": atom_indices,
                        }
                    )

    # Find unused SMARTS patterns
    unused_smarts = []
    for key, (smarts, _) in smartsPatts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and not mol.HasSubstructMatch(pattern):
            unused_smarts.append(smarts)

    # Determine the number of MACCS keys to modify
    n = len(active_keys_and_substructures)
    if modify_num == 0:
        num_to_modify = max(int(replace_ratio * n), 1)
    else:
        num_to_modify = modify_num
    if num_to_modify == 0 or not active_keys_and_substructures or not unused_smarts:
        return {
            "original_smiles": smiles,
            "num_of_key_substructures": n,
            "non_removed_count": None,
            "non_added_count": None,
            "substructures_removed": None,
            "substructures_added": None,
            "removed_graph": None,
            "modified_smiles": "CCC",
            "modified_graph": mol2graph(Chem.MolFromSmiles("CCC")),
        }

    # Select random MACCS key substructures to remove
    substructures_to_remove = random.sample(
        active_keys_and_substructures, num_to_modify
    )

    # Select random unused SMARTS patterns to add
    new_substructures = random.sample(unused_smarts, num_to_modify)

    # Create an editable molecule for modification
    editable_mol = Chem.RWMol(mol)

    # Remove selected substructures
    # atom_need_to_remove = []
    not_removed_count = 0
    for substructure in substructures_to_remove:
        pattern = Chem.MolFromSmarts(substructure["smarts"])
        matches = editable_mol.GetSubstructMatches(pattern)
        if matches:
            for _ in range(len(matches)):
                matches = editable_mol.GetSubstructMatches(pattern)
                if matches:
                    try:
                        match = matches[0]
                        res = Chem.RWMol(editable_mol)
                        res.BeginBatchEdit()
                        for aid in match:
                            res.RemoveAtom(aid)
                        res.CommitBatchEdit()
                        Chem.SanitizeMol(res)
                        editable_mol = res
                    except:
                        continue
        else:
            not_removed_count += 1

    editable_mol = Chem.RWMol(editable_mol)
    removed_mol = mol2graph(editable_mol)
    # ----------------- Debug Completed for upper part -----------------

    # (version 2)
    # Add new substructures and connect them
    non_added_count = 0
    for smarts in new_substructures:
        new_smarts = make_specific_smarts(smarts)
        replacement_mol = Chem.MolFromSmarts(new_smarts)
        if replacement_mol:
            replacement_mol = Chem.RWMol(replacement_mol)
            # Select a random atom from the existing molecule to attach to
            attach_idx = 0
            while attach_idx < editable_mol.GetNumAtoms() - 1:
                try:
                    copy_mol = editable_mol
                    # Add the first atom of the replacement molecule
                    # copy_mol = add_replacement_mol(copy_mol, replacement_mol, attach_idx)

                    new_atom_idx = copy_mol.AddAtom(replacement_mol.GetAtomWithIdx(0))
                    # Add a single bond between the selected atom and the new atom
                    copy_mol.AddBond(attach_idx, new_atom_idx, Chem.BondType.SINGLE)
                    # Update properties to avoid implicit valence issues
                    editable_mol.UpdatePropertyCache(strict=False)
                    editable_mol = copy_mol
                    break
                except:
                    attach_idx += 1
                    pass
            if attach_idx == editable_mol.GetNumAtoms() - 1:
                non_added_count += 1

    # Generate the modified graph
    modified_smiles = Chem.MolToSmiles(editable_mol)
    modified_graph = mol2graph(editable_mol)
    if modified_graph["num_nodes"] == 0:
        return {
            "original_smiles": smiles,
            "num_of_key_substructures": n,
            "non_removed_count": not_removed_count,
            "non_added_count": non_added_count,
            "substructures_removed": substructures_to_remove,
            "substructures_added": new_substructures,
            "removed_graph": removed_mol,
            "modified_smiles": "CCC",
            "modified_graph": mol2graph(Chem.MolFromSmiles("CCC")),
        }
    assert modified_graph["num_nodes"] != 0, "Modified graph has no nodes."
    # print("Graph", modified_graph)
    return {
        "original_smiles": smiles,
        "num_of_key_substructures": n,
        "non_removed_count": not_removed_count,
        "non_added_count": non_added_count,
        "substructures_removed": substructures_to_remove,
        "substructures_added": new_substructures,
        "removed_graph": removed_mol,
        "modified_smiles": modified_smiles,
        "modified_graph": modified_graph,
    }


def size_augmentation_single_mol(mol, min_r=0.3, max_r=0.9):
    num_atoms = mol.GetNumAtoms()
    min_atoms = max(1, int(num_atoms * min_r))
    max_atoms = min(int(num_atoms * max_r), num_atoms - 1)

    if min_atoms >= max_atoms:
        edit_mol = add_atoms_based_on_mol(mol, num_atoms_to_add=min_atoms)
    else:
        num_changing_atoms = np.random.randint(min_atoms, max_atoms)

        prob = np.random.rand()

        if prob > 0.5:
            edit_mol = add_atoms_based_on_mol(mol, num_atoms_to_add=num_changing_atoms)
        else:
            edit_mol = remove_atoms_based_on_mol(
                mol, num_atoms_to_remove=num_changing_atoms
            )

    assert edit_mol is not None

    return edit_mol


def remove_atoms_based_on_mol(mol, num_atoms_to_remove):
    assert (
        num_atoms_to_remove < mol.GetNumAtoms()
    ), "num_atoms_to_remove should be less than the number of atoms in the molecule."
    assert num_atoms_to_remove > 0, "num_atoms_to_remove should be positive."

    sanitized_mol = Chem.RWMol(mol)

    while num_atoms_to_remove > 0:
        potential_remove_indices = []
        for atom in sanitized_mol.GetAtoms():
            # Check if all neighbors are connected by single bonds
            # to guarantee that the new bond will be single, so that might be kekulizable...
            all_single_bonds = True
            for neighbor in atom.GetNeighbors():
                bond = sanitized_mol.GetBondBetweenAtoms(
                    atom.GetIdx(), neighbor.GetIdx()
                )
                if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                    all_single_bonds = False
                    break
            if all_single_bonds:
                potential_remove_indices.append(atom.GetIdx())

        if not potential_remove_indices:
            break

        np.random.shuffle(potential_remove_indices)

        change_made = False
        for atom_index in potential_remove_indices:
            try:
                rw_mol = Chem.RWMol(sanitized_mol)
                rw_mol.RemoveAtom(atom_index)
                Chem.SanitizeMol(rw_mol)
                sanitized_mol = rw_mol.GetMol()
                num_atoms_to_remove -= 1
                change_made = True
                break
            except:
                continue

        if not change_made:
            break

    out = {
        "mol": sanitized_mol,
        "original_num_atoms": mol.GetNumAtoms(),
        "augmeted_num_atoms": sanitized_mol.GetNumAtoms(),
        "num_not_fulfilled_changes": num_atoms_to_remove,
    }
    return out


def get_unique_atoms_and_counts(molecule):
    """
    Get a list of unique atoms and their counts in the given RDKit molecule object.

    Args:
    molecule (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
    dict: A dictionary with atom symbols as keys and their counts as values.
    """
    # Extract atom symbols from the molecule
    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]

    # Count occurrences of each atom symbol
    atom_counts = Counter(atom_symbols)

    return dict(atom_counts)


def sample_atom(atom_counts):
    """
    Sample an atom from the atom_counts with a probability proportional to its count.

    Args:
    atom_counts (dict): Dictionary of atom symbols and their counts.

    Returns:
    str: A sampled atom symbol based on its relative abundance.
    """
    # Extract atoms and their respective counts
    atoms = list(atom_counts.keys())
    counts = list(atom_counts.values())

    # Sample one atom based on the counts as weights
    sampled_atom = np.random.choice(atoms, p=np.array(counts) / np.sum(counts)).item()

    return sampled_atom


def add_atoms_based_on_mol(mol, num_atoms_to_add):
    assert num_atoms_to_add > 0, "num_atoms_to_add should be positive."

    sanitized_mol = Chem.RWMol(mol)
    atom_counts = get_unique_atoms_and_counts(sanitized_mol)

    while num_atoms_to_add > 0:
        potential_attaching_indices = []
        for atom in sanitized_mol.GetAtoms():
            all_single_bonds = all(
                [
                    bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                    for bond in atom.GetBonds()
                ]
            )

            if atom.GetImplicitValence() > 0 or not all_single_bonds:
                potential_attaching_indices.append(atom.GetIdx())

        if not potential_attaching_indices:
            break

        np.random.shuffle(potential_attaching_indices)

        change_made = False
        for atom_index in potential_attaching_indices:
            try:
                rw_mol = Chem.RWMol(sanitized_mol)
                new_atom = Chem.Atom(sample_atom(atom_counts))
                new_atom_idx = rw_mol.AddAtom(new_atom)
                attaching_atom = rw_mol.GetAtomWithIdx(atom_index)
                if (
                    attaching_atom.GetTotalValence()
                    <= attaching_atom.GetExplicitValence()
                ):
                    # remove one bond other than single bond
                    bonds = attaching_atom.GetBonds()
                    one_level_downgrade = {
                        Chem.rdchem.BondType.DOUBLE: Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.TRIPLE: Chem.rdchem.BondType.DOUBLE,
                    }
                    bonds = [
                        bond
                        for bond in bonds
                        if bond.GetBondType() in one_level_downgrade
                    ]
                    np.random.shuffle(bonds)
                    bond = bonds[0]
                    new_bond_type = one_level_downgrade[bond.GetBondType()]
                    rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    rw_mol.AddBond(
                        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_bond_type)
                    )

                rw_mol.AddBond(
                    atom_index, new_atom_idx, order=Chem.rdchem.BondType.SINGLE
                )

                Chem.SanitizeMol(rw_mol)
                sanitized_mol = rw_mol.GetMol()
                atom_counts[new_atom.GetSymbol()] += 1
                num_atoms_to_add -= 1
                change_made = True
                break
            except:
                continue

        if not change_made:
            break

    out = {
        "mol": sanitized_mol,
        "original_num_atoms": mol.GetNumAtoms(),
        "augmeted_num_atoms": sanitized_mol.GetNumAtoms(),
        "num_not_fulfilled_changes": num_atoms_to_add,
    }
    return out


def map_by_substructure_replacement(
    data_point, replace_ratio=0.1, num_rejected_graphs=5
):

    replace_ratio = min(replace_ratio, 1.0)
    task = data_point["task"]
    input_mol_string = data_point["input_mol_string"]

    selfies = (
        input_mol_string.replace("<SELFIES>", "")
        .replace("</SELFIES>", "")
        .replace(" ", "")
    )

    for i in range(num_rejected_graphs):
        if task in REGRESSION_BENCHMARKS:
            prob = np.random.rand()
            if prob < 0.5:
                rejected_graph = extract_and_modify(
                    selfies, replace_ratio=replace_ratio
                )["modified_graph"]
                additional_rejected_graph = extract_and_modify(
                    selfies, replace_ratio=replace_ratio
                )["modified_graph"]
            else:
                smiles = sf.decoder(selfies)
                mol = Chem.MolFromSmiles(smiles)
                rejected_mol = size_augmentation_single_mol(mol)
                rejected_graph = mol2graph(rejected_mol["mol"])
                additional_rejected_graph = mol2graph(rejected_mol["mol"])

        elif "|>>|" in selfies:
            pair_selfies = selfies.split("|>>|")
            rejected_graph = extract_and_modify(
                pair_selfies[0], replace_ratio=replace_ratio
            )["modified_graph"]
            additional_rejected_graph = extract_and_modify(
                pair_selfies[1], replace_ratio=replace_ratio
            )["modified_graph"]
        elif task in TEXT2MOL_BENCHMARKS + [
            "smol-name_conversion-i2s",
            "smol-name_conversion-i2f",
        ]:
            # dummy graph for text2mol tasks
            dummy_selfies = "[C][C][C]"
            rejected_graph = extract_and_modify(
                dummy_selfies, replace_ratio=replace_ratio
            )["modified_graph"]
            additional_rejected_graph = extract_and_modify(
                dummy_selfies, replace_ratio=replace_ratio
            )["modified_graph"]
        else:
            rejected_graph = extract_and_modify(selfies, replace_ratio=replace_ratio)[
                "modified_graph"
            ]
            additional_rejected_graph = extract_and_modify(
                selfies, replace_ratio=replace_ratio
            )["modified_graph"]

        data_point[f"{i}-th_rejected_x"] = rejected_graph["node_feat"]
        data_point[f"{i}-th_rejected_edge_index"] = rejected_graph["edge_index"]
        data_point[f"{i}-th_rejected_edge_attr"] = rejected_graph["edge_feat"]
        data_point[f"{i}-th_additional_rejected_x"] = additional_rejected_graph[
            "node_feat"
        ]
        data_point[f"{i}-th_additional_rejected_edge_index"] = (
            additional_rejected_graph["edge_index"]
        )
        data_point[f"{i}-th_additional_rejected_edge_attr"] = additional_rejected_graph[
            "edge_feat"
        ]

    return data_point


if __name__ == "__main__":
    from data.builders.hard_negative.__main__ import main
    main()
