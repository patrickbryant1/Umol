#Map ligand atoms

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import copy
import matplotlib.pyplot as plt
import pdb



def bonds_from_smiles(smiles_string):
    """Get all bonds from the smiles
    """
    m = Chem.MolFromSmiles(smiles_string)
    atom_encoding = {'C':0, 'N':1, 'O':2}
    bond_encoding = {'SINGLE':1, 'DOUBLE':2, 'AROMATIC':3}
    #Go through the smiles and assign the atom types and a bond matrix
    atom_types = []
    num_atoms = len(m.GetAtoms())
    bond_matrix = np.zeros((num_atoms, num_atoms))


    #Get the atom types
    for atom in m.GetAtoms():
        atom_types.append(atom_encoding[atom.GetSymbol()])
        #Get neighbours and assign bonds
        for nb in atom.GetNeighbors():
            for bond in nb.GetBonds():
                bond_type = bond_encoding[str(bond.GetBondType())]
                si = bond.GetBeginAtomIdx()
                ei = bond.GetEndAtomIdx()
                bond_matrix[si,ei] = bond_type
                bond_matrix[ei,si] = bond_type

    #Get a distance matrix
    AllChem.EmbedMolecule(m)
    D=AllChem.Get3DDistanceMatrix(m)
    #Get the bond positions
    has_bond = copy.deepcopy(bond_matrix)
    has_bond[has_bond>0]=1

    return np.array(atom_types), bond_matrix, has_bond, D*has_bond



#Get the atom types and bonds
atom_types, bond_types, bond_mask, bond_lengths = bonds_from_smiles('OCCc1ccn2cnccc12')

ligand_feats = {}
ligand_feats['atom_types'] = atom_types
ligand_feats['bond_types'] = bond_types
ligand_feats['bond_mask'] = atom_types
ligand_feats['bond_lengths'] = bond_lengths
#Write out features as a pickled dictionary.
features_output_path = os.path.join(outdir, 'ligand_structure_features.pkl')
with open(features_output_path, 'wb') as f:
    pickle.dump(structure_feats, f, protocol=4)
print('Saved features to',features_output_path)
pdb.set_trace()
