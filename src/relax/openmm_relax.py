import openmm as mm
import argparse
import sys
import numpy as np
import openmm.app as mm_app
import openmm.unit as mm_unit
import pdbfixer
import pdb
from sys import stdout
from openmm.app import PDBFile, Modeller
import mdtraj
from openmmforcefields.generators import SystemGenerator
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import UndefinedStereochemistryError, RadicalsNotSupportedError
from openmm import CustomExternalForce
import time

###########Functions##########
def fix_pdb(
        pdbname,
        outdir,
        file_name
        ):
    """Add hydrogens to the PDB file
    """
    fixer = pdbfixer.PDBFixer(pdbname)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    mm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{outdir}/{file_name}_hydrogen_added.pdb', 'w'))
    return fixer.topology, fixer.positions

def set_system(topology):
    """
    Set the system using the topology from the pdb file
    """
    #Put it in a force field to skip adding all particles manually
    forcefield = mm_app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(topology,
                                     removeCMMotion=False,
                                     nonbondedMethod=mm_app.NoCutoff,
                                     rigidWater=True #Use implicit solvent
                                     )
    return system

def minimize_energy(
        topology,
        system,
        positions,
        outdir,
        out_title
        ):
    '''Function that minimizes energy, given topology, OpenMM system, and positions '''
    #Use a Brownian Integrator
    integrator = mm.BrownianIntegrator(
        100 * mm.unit.kelvin,
        100. / mm.unit.picoseconds,
        2.0 * mm.unit.femtoseconds
    )
    simulation = mm.app.Simulation(topology, system, integrator)

    # Initialize the DCDReporter
    reportInterval = 100  # Adjust this value as needed
    reporter = mdtraj.reporters.DCDReporter('positions.dcd', reportInterval)

    # Add the reporter to the simulation
    simulation.reporters.append(reporter)

    simulation.context.setPositions(positions)

    simulation.minimizeEnergy(1, 1000)
    # Save positions
    minpositions = simulation.context.getState(getPositions=True).getPositions()
    mm_app.PDBFile.writeFile(topology, minpositions, open(outdir+f'{out_title}.pdb','w'))

    reporter.close()

    return topology, minpositions

def add_restraints(
        system,
        topology,
        positions,
        restraint_type
        ):
    '''Function to add restraints to specified group of atoms

    Code adapted from https://gist.github.com/peastman/ad8cda653242d731d75e18c836b2a3a5

    '''
    restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    system.addForce(restraint)
    restraint.addGlobalParameter('k', 100.0*mm_unit.kilojoules_per_mole/mm_unit.nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atom in topology.atoms():
        if restraint_type == 'protein':
            if 'x' not in atom.name:
                restraint.addParticle(atom.index, positions[atom.index])
        elif restraint_type == 'CA+ligand':
            if ('x' in atom.name) or (atom.name == "CA"):
                restraint.addParticle(atom.index, positions[atom.index])

    return system

if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser(description = '''Fold with l-BFGS in openmm using predicted constraints.''')

    parser.add_argument('--input_pdb', nargs=1, type= str, default=sys.stdin, help = 'Path to folded structure with protein.')
    parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir')
    parser.add_argument('--ligand_sdf', nargs=1, type= str, default=sys.stdin, help = 'Path to ligand structure saved as sdf file.')
    parser.add_argument('--file_name', nargs=1, type= str, default=sys.stdin, help = 'Name of protein to be saved.')
    parser.add_argument('--restraint_type', nargs=1, type= str, default=sys.stdin, help = 'Flag to restrain atoms')
    parser.add_argument('--relax_protein_first', action=argparse.BooleanOptionalAction, help = 'Flag to relax protein before ligand addition')

    #Parse args
    args = parser.parse_args()
    input_pdb = args.input_pdb[0]
    outdir = args.outdir[0]
    mol_in = args.ligand_sdf[0]
    file_name = args.file_name[0]
    relax_protein_first = args.relax_protein_first
    restraint_type = args.restraint_type[0]

    ## Read in ligand
    print('Reading ligand')
    try:
        ligand_mol = Molecule.from_file(mol_in)
    # Check for undefined stereochemistry, allow undefined stereochemistry to be loaded
    except UndefinedStereochemistryError:
        print('Undefined Stereochemistry Error found! Trying with undefined stereo flag True')
        ligand_mol = Molecule.from_file(mol_in, allow_undefined_stereo=True)
    # Check for radicals -- break out of script if radical is encountered
    except RadicalsNotSupportedError:
        print('OpenFF does not currently support radicals -- use unrelaxed structure')
        sys.exit()
    # Assigning partial charges first because the default method (am1bcc) does not work
    ligand_mol.assign_partial_charges(partial_charge_method='gasteiger')

    ## Read protein PDB and add hydrogens
    protein_topology, protein_positions = fix_pdb(input_pdb, outdir, file_name)
    print('Added all atoms...')

    # Minimize energy for the protein
    system = set_system(protein_topology)
    print('Creating system...')
    #Relax
    if relax_protein_first:
        print('Relaxing ONLY protein structure...')
        protein_topology, protein_positions = minimize_energy(
            protein_topology,
            system,
            protein_positions,
            outdir,
            f'{file_name}_relaxed_protein'
        )


    print('Preparing complex')
    ## Add protein first
    modeller = Modeller(protein_topology, protein_positions)
    print('System has %d atoms' % modeller.topology.getNumAtoms())

    ## Then add ligand
    print('Adding ligand...')
    lig_top = ligand_mol.to_topology()
    modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
    print('System has %d atoms' % modeller.topology.getNumAtoms())

    print('Preparing system')
    # Initialize a SystemGenerator using the GAFF for the ligand and implicit water.
    # forcefield_kwargs = {'constraints': mm_app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*mm_unit.amu }
    system_generator = SystemGenerator(
        forcefields=['amber14-all.xml', 'implicit/gbn2.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=[ligand_mol],
        # forcefield_kwargs=forcefield_kwargs
    )

    ## Create system
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    if restraint_type == 'protein':
        print('Adding restraints on entire protein')
    elif restraint_type == 'CA+ligand':
        print('Adding restraints on protein CAs and ligand atoms')

    system = add_restraints(system, modeller.topology, modeller.positions, restraint_type=restraint_type)

    ## Minimize energy
    minimize_energy(
        modeller.topology,
        system,
        modeller.positions,
        outdir,
        f'{file_name}_complex'
    )

    print(f'Time taken for calculation is {time.time()-start_time:.1f} seconds')
