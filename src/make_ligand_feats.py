#Map ligand atoms for feature generation
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import copy
import pickle
import argparse
from ast import literal_eval
import os
import sys
import pdb

parser = argparse.ArgumentParser(description = """Builds the ligand input feats and ground truth structural features for the loss calculations.""")

parser.add_argument('--input_smiles', nargs=1, type= str, default=sys.stdin, help = 'Smiles string. Note that these should be canonical.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############

def bonds_from_smiles(smiles_string, atom_encoding):
    """Get all bonds from the smiles
    """
    m = Chem.MolFromSmiles(smiles_string)

    bond_encoding = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':4, 'IONIC':5}

    #Go through the smiles and assign the atom types and a bond matrix
    atoms = []
    atom_types = []
    num_atoms = len(m.GetAtoms())
    bond_matrix = np.zeros((num_atoms, num_atoms))


    #Get the atom types
    for atom in m.GetAtoms():
        atoms.append(atom.GetSymbol())
        atom_types.append(atom_encoding.get(atom.GetSymbol(),10))
        #Get neighbours and assign bonds
        for nb in atom.GetNeighbors():
            for bond in nb.GetBonds():
                bond_type = bond_encoding.get(str(bond.GetBondType()), 6)
                si = bond.GetBeginAtomIdx()
                ei = bond.GetEndAtomIdx()
                bond_matrix[si,ei] = bond_type
                bond_matrix[ei,si] = bond_type

    #Get a distance matrix
    #Add Hs
    m = Chem.AddHs(m)
    #Embed in 3D
    AllChem.EmbedMolecule(m, maxAttempts=500)
    #Remove Hs to fit other dimensions (will cause error if mismatch on mult with has_bond)
    m = Chem.RemoveHs(m)
    D=AllChem.Get3DDistanceMatrix(m)
    #Get the bond positions
    has_bond = copy.deepcopy(bond_matrix)
    has_bond[has_bond>0]=1

    return np.array(atom_types), np.array(atoms), bond_matrix, D*has_bond, has_bond


##################MAIN#######################

#Parse args
args = parser.parse_args()
#Data
input_smiles = args.input_smiles[0]
outdir = args.outdir[0]

#Atom encoding - no hydrogens
atom_encoding = {'B':0, 'C':1, 'F':2, 'I':3, 'N':4, 'O':5, 'P':6, 'S':7,'Br':8, 'Cl':9, #Individual encoding
                'As':10, 'Co':10, 'Fe':10, 'Mg':10, 'Pt':10, 'Rh':10, 'Ru':10, 'Se':10, 'Si':10, 'Te':10, 'V':10, 'Zn':10 #Joint (rare)
                 }

#Get the atom types and bonds
atom_types, atoms, bond_types, bond_lengths, bond_mask = bonds_from_smiles(input_smiles, atom_encoding)

ligand_inp_feats = {}
ligand_inp_feats['atoms'] = atoms
ligand_inp_feats['atom_types'] = atom_types
ligand_inp_feats['bond_types'] = bond_types
ligand_inp_feats['bond_lengths'] = bond_lengths
ligand_inp_feats['bond_mask'] = bond_mask
#Write out features as a pickled dictionary.

features_output_path = os.path.join(outdir, 'ligand_inp_features.pkl')
with open(features_output_path, 'wb') as f:
    pickle.dump(ligand_inp_feats, f, protocol=4)
print('Saved features to',features_output_path)
