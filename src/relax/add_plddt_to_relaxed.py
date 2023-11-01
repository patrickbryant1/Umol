#Map ligand atoms for feature generation
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
from rdkit.Geometry import Point3D
import pandas as pd
import argparse

import os
import sys
import pdb

parser = argparse.ArgumentParser(description = """Add the plDDT scores to the relaxed complex.""")

parser.add_argument('--raw_complex', nargs=1, type= str, default=sys.stdin, help = 'Path to pdb file with predicted protein-ligand positions.')
parser.add_argument('--relaxed_complex', nargs=1, type= str, default=sys.stdin, help = 'Path to pdb file with relaxed protein-ligand positions.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############

def read_pdb(pdbname):
    '''Read PDB
    '''


    f=open(pdbname,'rt')
    parser = PDBParser()
    struc = parser.get_structure('', f)


    #Save
    model_coords = []
    model_chains = []
    model_atom_numbers = []
    model_3seq = []
    model_resnos = []
    model_atoms = []
    model_bfactors = []

    for model in struc:
        for chain in model:

            #Go through all residues
            for residue in chain:
                res_name = residue.get_resname()
                for atom in residue:
                    atom_id = atom.get_id()
                    if 'H' in atom_id:
                        continue
                    atm_name = atom.get_name()
                    #Save
                    model_coords.append(atom.get_coord())
                    model_chains.append(chain.id)
                    model_atom_numbers.append(atom.get_serial_number())
                    model_3seq.append(res_name)
                    model_resnos.append(residue.get_id()[1])
                    model_atoms.append(atom_id)
                    model_bfactors.append(atom.bfactor)


    return model_coords, model_chains, model_atom_numbers, model_3seq, model_resnos, model_atoms, model_bfactors


def format_line(atm_no, atm_name, res_name, chain, res_no, coord, occ, B , atm_id):
    '''Format the line into PDB
    '''

    #Get blanks
    atm_no = ' '*(5-len(atm_no))+atm_no
    atm_name = atm_name+' '*(4-len(atm_name))
    res_name = ' '*(3-len(res_name))+res_name
    res_no = ' '*(4-len(res_no))+res_no
    x,y,z = coord
    x,y,z = str(np.round(x,3)), str(np.round(y,3)), str(np.round(z,3))
    x =' '*(8-len(x))+x
    y =' '*(8-len(y))+y
    z =' '*(8-len(z))+z
    occ = ' '*(6-len(occ))+occ
    B = ' '*(6-len(B))+B

    if 'x' in atm_name:
        line = 'HETATM'+atm_no+'  '+atm_name+res_name+' '+chain+res_no+' '*4+x+y+z+occ+B+' '*11+atm_id+'  '
    else:
        line = 'ATOM  '+atm_no+'  '+atm_name+res_name+' '+chain+res_no+' '*4+x+y+z+occ+B+' '*11+atm_id+'  '
    return line

def write_pdb(coords, chains, atm_nos, seq, resnos, atoms, bfacs, outname):
    """Write PDB
    """

    #OWrite file
    with open(outname, 'w') as file:
        for i in range(len(coords)):
            if i>=len(bfacs):
                bfac = bfacs[-1]
            else:
                bfac = bfacs[i]
            file.write(format_line(str(atm_nos[i]), atoms[i], seq[i], chains[i], str(resnos[i]), coords[i],str(1),str(np.round(bfac*100)), atoms[i][0])+'\n')



##################MAIN#######################

#Parse args
args = parser.parse_args()

#Data
raw_coords, raw_chains, raw_atom_numbers, raw_3seq, raw_resnos, raw_atoms, raw_bfactors = read_pdb(args.raw_complex[0])
relaxed_coords, relaxed_chains, relaxed_atom_numbers,  relaxed_3seq, relaxed_resnos, relaxed_atoms, relaxed_bfactors = read_pdb(args.relaxed_complex[0])
outdir = args.outdir[0]

#Write PDB
id=args.raw_complex[0].split('/')[-1].split('_')[0]
outname=outdir+id+'_relaxed_plddt.pdb'
write_pdb(relaxed_coords, relaxed_chains, relaxed_atom_numbers, relaxed_3seq, relaxed_resnos, relaxed_atoms, raw_bfactors, outname)
