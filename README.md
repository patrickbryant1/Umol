# Umol - **U**niversal **mol**ecular framework

## Structure prediction of protein-ligand complexes from sequence information
The protein is represented with a multiple sequence alignment and the ligand as a SMILES string, allowing for unconstrained flexibility in the protein-ligand interface. There are two versions of Umol: one that uses protein pocket information (recommended) and one that does not. Please see the runscript (predict.sh) for more information.

[Read the paper here](https://www.biorxiv.org/content/10.1101/2023.11.03.565471v1)

<img src="./Network.svg"/>

Umol is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). \
The Umol parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).


# Colab (run Umol in the browser)

[Colab Notebook](https://colab.research.google.com/github/patrickbryant1/Umol/blob/master/Umol.ipynb)

# Local installation
## (several minutes)
The entire installation takes <1 hour on a standard computer. \
We assume you have CUDA12. For CUDA11, you will have to change the installation of some packages. \
The runtime will depend on the GPU you have available and the size of the protein-ligand complex you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.

First install miniconda, see: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html or https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html

```
bash install_dependencies.sh
```


# Run the test case
## (a few minutes)
```
conda activate umol
bash predict.sh
```

## Extract target positions from a pdb file of your choice
```
PDB_FILE=./data/test_case/7NB4/7NB4.pdb1
PROTEIN_CHAIN='A'
LIGAND_NAME='U6Q'
OUTDIR=./data/test_case/7NB4/
python3 ./src/parse_pocket.py --pdb_file $PDB_FILE \
--protein_chain $PROTEIN_CHAIN \
--ligand_name $LIGAND_NAME \
--outdir $OUTDIR
```

# Citation
Structure prediction of protein-ligand complexes from sequence information with Umol
Patrick Bryant, Atharva Kelkar, Andrea Guljas, Cecilia Clementi, Frank Noé
bioRxiv 2023.11.03.565471; doi: https://doi.org/10.1101/2023.11.03.565471
