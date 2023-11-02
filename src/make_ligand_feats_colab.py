#Map ligand atoms for feature generation
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import copy
import os
import sys
import pdb


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




