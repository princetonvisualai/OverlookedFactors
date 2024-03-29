This is the code to reproduce results from "Overlooked Factors in Concept-based Explanations: Dataset Choice, Concept Learnability, and Human Capability" (to appear in CVPR 2023); see https://arxiv.org/abs/2207.09615 for a preprint

To start, download the Broden dataset using scripts from https://github.com/CSAILVision/NetDissect-Lite and download the CUB dataset from http://www.vision.caltech.edu/datasets/cub_200_2011/

Use data_processing.py from https://github.com/yewsiang/ConceptBottleneck/tree/master/CUB to get train/val/test splits for the CUB dataset

Run data_setup.py and get_features.py to compute ADE20k and Pascal features

Run cub_features.py to get features for the CUB dataset

For results from Sec 4 of the paper, use notebook DatasetChoice.ipynb 

For results from Sec 5 of the paper, use notebook ConceptLearning.ipynb 

Code for Sec 6 of the paper is in humanstudy_code 
