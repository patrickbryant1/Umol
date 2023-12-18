# Umol - **U**niversal **mol**ecular framework

## Structure prediction of protein-ligand complexes from sequence information
The protein is represented with a multiple sequence alignment and the ligand as a SMILES string, allowing for unconstrained flexibility in the protein-ligand interface. At a high accuracy threshold, unseen protein-ligand complexes can be predicted more accurately than for RoseTTAFold-AA, and at medium accuracy even classical docking methods that use known protein structures as input are surpassed.

[Read the paper here](https://www.biorxiv.org/content/10.1101/2023.11.03.565471v1)

<img src="./Network.svg"/>

Umol is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). \
The Umol parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).


# Colab (run Umol in the browser)

[Colab Notebook](https://colab.research.google.com/github/patrickbryant1/Umol/blob/master/Umol.ipynb)

# Local installation
The entire installation takes <1 hour on a standard computer. \
The runtime will depend on the GPU you have available and the size of the protein-ligand complex you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.


## Install packages and databases (several minutes)
To do this, first install miniconda, see: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

```
bash install_dependencies.sh
```


# Run the test case (a few minutes)
```
conda activate umol
bash predict.sh
```

# Citation
Structure prediction of protein-ligand complexes from sequence information with Umol
Patrick Bryant, Atharva Kelkar, Andrea Guljas, Cecilia Clementi, Frank Noé
bioRxiv 2023.11.03.565471; doi: https://doi.org/10.1101/2023.11.03.565471
