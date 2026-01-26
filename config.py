import os

# Dataset root path
'''
    define your root directory
'''
ROOT = './dataset'

 

# Model settings
MEGAD_NAME = 'hf-hub:BVRA/MegaDescriptor-L-384'
EVA_NAME = 'EVA02-B-16'
EVA_WEIGHT_NAME = 'merged2b_s8b_b131k'
DEVICE = 'cuda'

# Threshold
THRESHOLD = 0.35