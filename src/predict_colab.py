import os
import pickle
import sys
import time
from typing import NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
#import optax
#Silence tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.config.set_visible_devices([], 'GPU')

import pandas as pd
import numpy as np
from collections import Counter
from scipy.special import softmax
import pdb

#net imports
from net.common import protein
from net.common import residue_constants
from net.model import config
from net.model import features
from net.model import modules


##############FUNCTIONS##############

##########INPUT DATA#########
def process_protein_features(raw_features, config, random_seed):
    """Processes features to prepare for feeding them into the model.

    Args:
    raw_features: The output of the data pipeline either as a dict of NumPy
      arrays or as a tf.train.Example.
    random_seed: The random seed to use when processing the features.

    Returns:
    A dict of NumPy feature arrays suitable for feeding into the model.
    """
    return features.np_example_to_features(np_example=raw_features,
                                            config=config,
                                            random_seed=random_seed)


def make_uniform(protein_feats, ligand_feats, crop_size):
    """Add all feats together - pad if needed
    """

    batch_ex = {}
    #Dimension independent feats (do not need to be made uniform)
    batch_ex['seq_length'] = np.array([protein_feats['seq_length']])
    batch_ex['is_distillation'] = np.array([protein_feats['is_distillation']])
    batch_ex['msa_row_mask'] = protein_feats['msa_row_mask']
    batch_ex['random_crop_to_size_seed'] = np.array([protein_feats['random_crop_to_size_seed']])

    ################Place holders################
    batch_ex['aatype'] = np.zeros(crop_size, dtype='int32')
    batch_ex['seq_mask'] = np.zeros(crop_size)
    batch_ex['msa_mask'] = np.zeros((128, crop_size))
    batch_ex['residx_atom14_to_atom37'] = np.zeros((crop_size, 14), dtype='int32')
    batch_ex['residx_atom37_to_atom14'] = np.zeros((crop_size, 37), dtype='int32')
    batch_ex['atom37_atom_exists'] = np.zeros((crop_size, 37))
    batch_ex['extra_msa'] = np.zeros((1024, crop_size), dtype='int32')
    batch_ex['extra_msa_mask'] = np.zeros((1024, crop_size))
    batch_ex['bert_mask'] = np.zeros((128, crop_size))
    batch_ex['true_msa'] = np.zeros((128, crop_size), dtype='int32')
    batch_ex['extra_has_deletion'] = np.zeros((1024, crop_size))
    batch_ex['extra_deletion_value'] = np.zeros((1024, crop_size))
    batch_ex['msa_feat'] = np.zeros((128, crop_size, 50))
    batch_ex['target_feat'] = np.zeros((crop_size, 22+11)) #22 AAs + 11 atoms
    batch_ex['atom14_atom_exists'] = np.zeros((crop_size, 14))

    batch_ex['residue_index'] = np.zeros(crop_size, dtype='int32')

    #Ligand feats: bond types (7), bond lengths (1), target positions (1)
    batch_ex['ligand_feats'] = np.zeros((crop_size, crop_size, 9))
    #Bond mask for ligand harmonic potentials
    batch_ex['ligand_bond_mask'] = np.zeros((crop_size, crop_size))
    #1D masks for lDDT
    batch_ex['ligand_mask'] = np.zeros(crop_size)
    batch_ex['protein_mask'] = np.zeros(crop_size)

    ################Assign feats (pad if needed)################
    protein_len, ligand_len = len(protein_feats['aatype']), len(ligand_feats['atom_types'])
    tot_len = protein_len+ligand_len
    batch_ex['protein_len'] = [protein_len]
    batch_ex['ligand_len'] = [ligand_len]
    #The aatype will contain the atom types as well.
    #The types are zero indexed (+21) but the target feat below are +1 since they are onehot encoded
    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'. Put 0 (GLY) for ligand atoms - will take care of lots of mapping inside the net
    batch_ex['aatype'][:protein_len] = protein_feats['aatype']
    batch_ex['seq_mask'][:tot_len] = 1
    batch_ex['msa_mask'][:,:protein_len] = protein_feats['msa_mask']
    batch_ex['residx_atom14_to_atom37'][:protein_len,:] = protein_feats['residx_atom14_to_atom37']
    batch_ex['residx_atom14_to_atom37'][protein_len:,1]=1 #Ligand mapping - 1to1
    batch_ex['residx_atom37_to_atom14'][:protein_len,:] = protein_feats['residx_atom37_to_atom14']
    batch_ex['residx_atom37_to_atom14'][protein_len:,1]=1
    batch_ex['atom37_atom_exists'][:protein_len,:] = protein_feats['atom37_atom_exists']
    batch_ex['atom37_atom_exists'][protein_len:tot_len,1] = 1
    batch_ex['residx_atom14_to_atom37'][protein_len:tot_len,1]=1 #Ligand mapping
    batch_ex['extra_msa'][:,:protein_len] = protein_feats['extra_msa']
    batch_ex['extra_msa_mask'][:,:protein_len] = protein_feats['extra_msa_mask']
    batch_ex['bert_mask'][:,:protein_len] = protein_feats['bert_mask']
    batch_ex['true_msa'][:,:protein_len] = protein_feats['true_msa']
    batch_ex['extra_has_deletion'][:,:protein_len] = protein_feats['extra_has_deletion']
    batch_ex['extra_deletion_value'][:,:protein_len] = protein_feats['extra_deletion_value']
    batch_ex['msa_feat'][:,:protein_len,:] = protein_feats['msa_feat']

    #Target feat - addition of 8 atom types (22 for AAs)
    batch_ex['target_feat'][:protein_len,:22] = protein_feats['target_feat']
    batch_ex['target_feat'][protein_len:tot_len,:] = np.eye(22+11)[ligand_feats['atom_types']+22]


    batch_ex['atom14_atom_exists'][:protein_len] = protein_feats['atom14_atom_exists']
    batch_ex['atom14_atom_exists'][protein_len:tot_len,1] = 1 #The CA index is 1 - the ligand atom pos

    #Cat and increase indices - will be an offset feature clipped at 32 (check the biggest ligands?)
    batch_ex['residue_index'] = np.array(range(tot_len), dtype=np.int32)
    batch_ex['residue_index'][protein_len:tot_len] += 200

    #Assign the ligand feats
    #Ligand feats: bond types (7), bond lengths (1), target positions (1)
    #Remember to update the losses if this changes (harmonic bond loss)
    batch_ex['ligand_feats'][protein_len:tot_len,protein_len:tot_len,:7] = np.eye(7)[np.array(ligand_feats['bond_types'],dtype=int)]
    batch_ex['ligand_feats'][protein_len:tot_len,protein_len:tot_len,7] = ligand_feats['bond_lengths']
    target_pos_sq = protein_feats['msa_feat'][0,:,-1][:,None]*protein_feats['msa_feat'][0,:,-1][None,:] #What positions in the protein to target for binding
    batch_ex['ligand_feats'][:protein_len,:protein_len,8] = target_pos_sq
    #Bond mask for ligand
    batch_ex['ligand_bond_mask'][protein_len:tot_len,protein_len:tot_len]=ligand_feats['bond_mask']
    #1D masks
    batch_ex['ligand_mask'][protein_len:tot_len]=1
    batch_ex['protein_mask'][:protein_len]=1

    return batch_ex



def load_input_feats(pdbid, msa_features, ligand_features, config, pocket_indices):
    """
    Load all input feats.
    """


    #Load raw protein features
    msa_feature_dict = np.load(msa_features, allow_pickle=True)

    #Process the features on CPU (sample MSA)
    protein_feats = process_protein_features(msa_feature_dict, config, np.random.choice(sys.maxsize))

    #Load the ligand features
    ligand_feats = np.load(ligand_features, allow_pickle=True)
    ligand_size = ligand_feats['atom_types'].shape[0]

    #Assign
    protein_pocket_indication = np.zeros(protein_feats['seq_length'])
    if len(pocket_indices)>0:
        protein_pocket_indication[pocket_indices]=1
    protein_pocket_indication = np.expand_dims(np.repeat(np.expand_dims(protein_pocket_indication,axis=0),128,axis=0),axis=2)
    #Add to MSA feats: 128xLx49 --> 128xLx50
    protein_feats['msa_feat'] = np.append(protein_feats['msa_feat'],protein_pocket_indication,axis=2)
    #Add in all the ligand feats
    protein_len, ligand_len = len(protein_feats['aatype']), len(ligand_feats['atom_types'])
    tot_len=protein_len+ligand_len
    #Add index
    batch_ex = make_uniform(protein_feats, ligand_feats, tot_len)

    return batch_ex, ligand_feats['atoms']



##########MODEL#########

def predict(config,
          msa_features,
          ligand_features,
          id,
          target_pos,
          ckpt_params=None,
          num_recycles=3,
          outdir=None):
    """Predict a structure
    """
    #Define the forward function
    def _forward_fn(batch):
        '''Define the forward function - has to be a function for JAX
        '''
        model = modules.Umol(config.model)

        return model(batch,
                    is_training=True,
                    compute_loss=False,
                    ensemble_representations=False,
                    return_representations=True)

    #The forward function is here transformed to apply and init functions which
    #can be called during training and initialisation (JAX needs functions)
    forward = hk.transform(_forward_fn)
    apply_fwd = forward.apply
    #Get a random key
    rng = jax.random.PRNGKey(42)


    #Load input feats
    batch, ligand_atoms = load_input_feats(id, msa_features, ligand_features, config, target_pos)
    for key in batch:
        try:
            batch[key] = np.reshape(batch[key], (1, *batch[key].shape))
        except:
            print(key)
    batch['num_iter_recycling'] = [num_recycles]

    ret = apply_fwd(ckpt_params, rng, batch)
    #Save structure
    save_feats = {'aatype':np.argmax(batch['target_feat'],axis=-1)-1,
                  'residue_index':batch['residue_index'],
                  'ligand_atoms':ligand_atoms}
    result = {'predicted_lddt':ret['predicted_lddt'],
            'structure_module':{'final_atom_positions':ret['structure_module']['final_atom_positions'],
            'final_atom_mask': ret['structure_module']['final_atom_mask']
            }}
    outname = outdir+id+'_pred_raw.pdb'
    save_structure(save_feats, result, outname)



def save_structure(save_feats, result, outname):
    """Save prediction

    save_feats = {'aatype':np.argmax(batch['target_feat'][0][0],axis=-1)-1,
                  'residue_index':batch['residue_index'][0][0]}
    result = {'predicted_lddt':aux['predicted_lddt'],
            'structure_module':{'final_atom_positions':aux['structure_module']['final_atom_positions'][0],
            'final_atom_mask': aux['structure_module']['final_atom_mask'][0]
            }}
    save_structure(save_feats, result, ts, outdir)


    """
    #Define the plDDT bins
    bin_width = 1.0 / 50
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    # Add the predicted LDDT in the b-factor column.
    plddt_per_pos = jnp.sum(jax.nn.softmax(result['predicted_lddt']['logits']) * bin_centers[None, :], axis=-1)
    plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=save_feats, result=result,  b_factors=plddt_b_factors)
    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)

    with open(outname, 'w') as f:
        f.write(unrelaxed_pdb)
