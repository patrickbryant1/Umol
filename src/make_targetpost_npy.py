'''
Usage: python make_targetpost_npy.py --outfile test.npy --target_pos "51,52,54,55,56,57,58,59,60,61,62,63,65,66,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,93,94,95,96,97,98,99"
'''
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = """Make target positions: from str to np""")
parser.add_argument('--target_pos', type= str, default='', help = 'A list of positions which are delimitered by comma. Ex: "51,52,54,55,56,57,58,59,60,61,62,63,65,66,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,93,94,95,96,97,98,99"')
parser.add_argument('--outfile', type= str, default='test.npy', help = 'save to .npy')
args = parser.parse_args()

target_position = np.array([int(x) for x in args.target_pos.split(',')])
np.save(args.outfile, target_position)
