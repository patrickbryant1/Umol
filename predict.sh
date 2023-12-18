#Predict the test case 7NB4
ID=7NB4
FASTA=./data/test_case/7NB4/7NB4.fasta
POCKET_INDICES=./data/test_case/7NB4/7NB4_pocket_indices.npy #Zero indexed numpy array of what residues are in the pocket (all CBs within 10Å from the ligand)
LIGAND_SMILES='CCc1sc2ncnc(N[C@H](Cc3ccccc3)C(=O)O)c2c1-c1cccc(Cl)c1C' #Make sure these are canonical as in RDKit
UNICLUST=./data/uniclust30_2018_08/uniclust30_2018_08
OUTDIR=./data/test_case/7NB4/

## Search Uniclust30 with HHblits to generate an MSA (a few minutes)
HHBLITS=./hh-suite/build/bin/hhblits
$HHBLITS -i $FASTA -d $UNICLUST -E 0.001 -all -oa3m $OUTDIR/$ID'.a3m'

wait
## Generate input feats (seconds)
python3 ./src/make_msa_seq_feats.py --input_fasta_path $FASTA \
--input_msas $OUTDIR/$ID'.a3m' \
--outdir $OUTDIR

python3 ./src/make_ligand_feats.py --input_smiles $LIGAND_SMILES \
--outdir $OUTDIR

wait
## Generate a pocket indices file from a list of what residues (zero indexed) are in the pocket (all CBs within 10Å from the ligand). (seconds)
python3 ./src/make_targetpost_npy.py --outfile $POCKET_INDICES --target_pos "50,51,53,54,55,56,57,58,59,60,61,62,64,65,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,103,104,124,127,128"

## Predict (a few minutes)
MSA_FEATS=$OUTDIR/msa_features.pkl
LIGAND_FEATS=$OUTDIR/ligand_inp_features.pkl
PARAMS=data/params/params40000.npy
NUM_RECYCLES=3

python3 ./src/predict.py --msa_features  $MSA_FEATS \
--ligand_features $LIGAND_FEATS \
--id $ID \
--ckpt_params $PARAMS \
--target_pos $POCKET_INDICES \
--num_recycles $NUM_RECYCLES \
--outdir $OUTDIR

wait
RAW_PDB=$OUTDIR/$ID'_pred_raw.pdb'
python3 ./src/relax/align_ligand_conformer.py --pred_pdb $RAW_PDB \
--ligand_smiles $LIGAND_SMILES --outdir $OUTDIR

grep ATOM $OUTDIR/$ID'_pred_raw.pdb' > $OUTDIR/$ID'_pred_protein.pdb'
echo "The unrelaxed predicted protein can be found at $OUTDIR/$ID'_pred_protein.pdb' and the ligand at $OUTDIR/$ID'_pred_ligand.sdf'"


## Relax the protein (a few minutes)
#This fixes clashes mainly in the protein, but also in the protein-ligand interface.

PRED_PROTEIN=$OUTDIR/$ID'_pred_protein.pdb'
PRED_LIGAND=$OUTDIR/$ID'_pred_ligand.sdf'
RESTRAINTS="CA+ligand" # or "protein"
python3 ./src/relax/openmm_relax.py --input_pdb $PRED_PROTEIN \
                        --ligand_sdf $PRED_LIGAND \
                        --file_name $ID \
                        --restraint_type $RESTRAINTS \
                        --outdir $OUTDIR

wait
#Write plDDT to Bfac column
RAW_COMPLEX=$OUTDIR/$ID'_pred_raw.pdb'
RELAXED_COMPLEX=$OUTDIR/$ID'_relaxed_complex.pdb'
python3 ./src/relax/add_plddt_to_relaxed.py  --raw_complex $RAW_COMPLEX \
--relaxed_complex $RELAXED_COMPLEX  \
--outdir $OUTDIR
echo "The final relaxed structure can be found at $OUTDIR/$ID'_relaxed_plddt.pdb'"
