# Umol
## Structure prediction of protein-ligand complexes from sequence information


Umol is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). \
The Umol parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).

# Colab
It is possible to run Umol directly in the web with Google Colab. \
Click this link and follow the instructions.


# Local installation
The entire installation takes <1 hour on a standard computer. \
The runtime will depend on the GPU you have available and the size of the protein-ligand complex you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.


## Install Python packages (several minutes)
* For the python environment, we recommend to install it with pip as described below. \
You can do this in your virtual environment of choice.

```
pip install -U jaxlib==0.3.24+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.3.24
pip install ml-collections==0.1.1
pip install dm-haiku==0.0.9
pip install pandas==1.3.5
pip install biopython==1.81
pip install chex==0.1.5
pip install dm-tree==0.1.8
pip install immutabledict==2.0.0
pip install numpy==1.21.6
pip install scipy==1.7.3
pip install tensorflow==2.11.0
pip install optax==0.1.4
pip install rdkit-pypi
```

## Get network parameters for Umol (a few minutes)

```
wget https://zenodo.org/records/10048543/files/params40000.npy
mkdir data/params
mv params40000.npy  data/params/
```


## Get Uniclust30 (a few minutes)

```
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
mkdir data/uniclust30
mv uniclust30_2018_08_hhsuite.tar.gz data/uniclust30
tar -zxvf data/uniclust30/uniclust30_2018_08_hhsuite.tar.gz
```

## Install HHblits (a few minutes)
```
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ..
```

# Run the test case
```
ID=7NB4
FASTA=./data/test_case/7NB4/7NB4.fasta
POCKET_INDICES=./data/test_case/7NB4/7NB4_pocket_indices.npy
LIGAND_SMILES='CCc1sc2ncnc(N[C@H](Cc3ccccc3)C(=O)O)c2c1-c1cccc(Cl)c1C'
UNICLUST=./data/uniclust30_2018_08/uniclust30_2018_08
OUTDIR=./data/test_case/7NB4/
```
## Search Uniclust30 with HHblits to generate an MSA (a few minutes)
```
HHBLITS=./hh-suite/build/bin/hhblits
$HHBLITS -i $FASTA -d $UNICLUST -E 0.001 -all -oa3m $OUTDIR/$ID'.a3m'
```

## Generate input feats (seconds)
```
python ./src/make_msa_seq_feats.py --input_fasta_path $FASTA \
--input_msas $OUTDIR/$ID'.a3m' \
--outdir $OUTDIR

python ./src/make_ligand_feats.py --input_smiles $LIGAND_SMILES \
--outdir $OUTDIR
```

## Predict (a few minutes)
```
MSA_FEATS=$OUTDIR/msa_features.pkl
LIGAND_FEATS=$OUTDIR/ligand_inp_features.pkl
PARAMS=data/params/params40000.npy
NUM_RECYCLES=3

python ./src/predict.py --msa_features  $MSA_FEATS \
--ligand_features $LIGAND_FEATS \
--id $ID \
--ckpt_params $PARAMS \
--target_pos $POCKET_INDICES \
--num_recycles $NUM_RECYCLES \
--outdir $OUTDIR
```

## Generate ligand conformer and relax (a few minutes).
```
RAW_PDB=$OUTDIR/$ID'_pred_raw.pdb'
RESTRAINTS="CA+ligand" # or "protein"

python ./src/relax/align_ligand_conformer.py --pred_pdb $RAW_PDB \
--ligand_smiles $LIG_SMILES --outdir $OUTDIR


python ./src/relax/openmm_relax.py --input_pdb $protein_pdb \
                        --ligand_sdf $ligand_sdf \
                        --file_name $ID \
                        --restraint_type $RESTRAINTS \
                        --outdir $OUTDIR

python ./src/relax/add_plddt_to_relaxed.py  --raw_complex $RAW_COMPLEX \
--relaxed_complex $RELAXED_COMPLEX  \
--outdir $OUTDIR

```
