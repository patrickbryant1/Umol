import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict, Counter
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem import AllChem
from ast import literal_eval
import re
import pdb

parser = argparse.ArgumentParser(description = '''Parse pocket from pdb file.''')

parser.add_argument('--pdb_file', nargs=1, type= str, default=sys.stdin, help = 'Path to PDB file.')
parser.add_argument('--protein_chain', nargs=1, type= str, default=sys.stdin, help = 'Protein chain name in PDB file.')
parser.add_argument('--ligand_name', nargs=1, type= str, default=sys.stdin, help = 'Ligand name in PDB file.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

##############FUNCTIONS##############
def read_pdb(pdbname):
    """Read protein and ligand info from PDB
    """




    f=open(pdbname,'rt')

    if '.pdb' in pdbname:
        parser = PDBParser()
        struc = parser.get_structure('', f)
    else:
        parser = MMCIFParser()
        struc = parser.get_structure('',f)

    #Save protein info
    protein_coords = {}


    #Save other coords
    ligand_atoms = {}
    ligand_elements = {}
    ligand_coords = {}
    ligand_chains = {}

    for protein in struc:
        for chain in protein:
            #Save
            protein_coords[chain.id]=[]

            #Go through al residues
            for residue in chain:
                #Check if AA
                if is_aa(residue)==True:
                    res_name = residue.get_resname()
                    for atom in residue:
                        #Save
                        atom_name = atom.get_id()
                        if (res_name=='GLY' and atom_name=='CA') or atom_name=='CB':
                            protein_coords[chain.id].append(atom.get_coord())

                else:
                    res_id = residue.get_id()[0]
                    if '_' in res_id:
                        res_id = res_id.split('_')[1]

                    #Check if water
                    if res_id=='W':
                        continue

                    ligand_atoms[res_id+'_'+chain.id] = []
                    ligand_elements[res_id+'_'+chain.id] = []
                    ligand_coords[res_id+'_'+chain.id] = []
                    ligand_chains[res_id+'_'+chain.id] = chain.id

                    for atom in residue:
                        #Save
                        ligand_atoms[res_id+'_'+chain.id].append(atom.get_id())
                        ligand_elements[res_id+'_'+chain.id].append(atom.element)
                        ligand_coords[res_id+'_'+chain.id].append(atom.get_coord())


    return [protein_coords], [ligand_atoms, ligand_coords, ligand_chains, ligand_elements]



##################MAIN#######################

#Parse args
args = parser.parse_args()
#Data

pdb_file = args.pdb_file[0]
protein_chain = args.protein_chain[0]
ligand_name = args.ligand_name[0]
outdir = args.outdir[0]

protein_info, ligand_info = read_pdb(pdb_file)


protein_coords = np.array(protein_info[0][protein_chain])
ligand_coords = np.array(ligand_info[1][ligand_name+'_'+protein_chain])

#Calculate dists
cat_coords = np.concatenate([protein_coords, ligand_coords])
dmat = np.sqrt(1e-10+np.sum((cat_coords[:,None]-cat_coords[None,:])**2,axis=-1))
protein_len = len(protein_coords)
dmat_sel = dmat[:protein_len,protein_len:]
pocket_residues = np.unique(np.argwhere(dmat_sel<10)[:,0])
np.save(outdir+'pocket_indices.npy', pocket_residues)
