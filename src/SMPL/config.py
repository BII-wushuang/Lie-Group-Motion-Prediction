import numpy as np

# Map joints Name to SMPL joints idx
JOINT_MAP = {
'MidHip': 0,
'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
'LShoulder': 16, 'LElbow': 18, 'LWrist': 20, 'LHand': 22, 
'RShoulder': 17, 'RElbow': 19, 'RWrist': 21, 'RHand': 23,
'spine1': 3, 'spine2': 6, 'spine3': 9,  'Neck': 12, 'Head': 15,
'LCollar':13, 'Rcollar' :14, 
'Nose':24, 'REye':26,  'LEye':26,  'REar':27,  'LEar':28, 
'LHeel': 31, 'RHeel': 34,
'OP RShoulder': 17, 'OP LShoulder': 16,
'OP RHip': 2, 'OP LHip': 1,
'OP Neck': 12,
}

full_smpl_idx = range(24)
key_smpl_idx = [0,  1, 4, 7,  2, 5, 8,  17, 19, 21,  16, 18, 20]

# H36m Joint Indices
h36m_JOINT_MAP = {
'MidHip': 0, 'spine2': 12, 'Neck': 13, 'Head': 14,
'LHip': 6, 'LKnee': 7, 'LAnkle': 8, 'LFoot': 9, 'LFootTIP': 10,
'RHip': 1, 'RKnee': 2, 'RAnkle': 3, 'RFoot': 4, 'RFootTIP': 5,
'LShoulder': 17, 'LElbow': 18, 'LWrist': 19, 'LHand': 22,
'RShoulder': 25, 'RElbow': 26, 'RWrist': 27, 'RHand': 30,
}
h36m_idx =     [0,12,13,15, 1,2,3,4,    6,7,8,9,    17,18,19,22,    25,26,27,30]
h36m_smpl_idx =[0,6,12,15,  2,5,8,11,   1,4,7,10,   16,18,20,22,    17,19,21,23]

# Dance Dataset Indices
WU_JOINT_MAP = {
'MidHip': 0, 'Chest': 1, 'Head':2,
'LHip': 16, 'LKnee': 17, 'LAnkle': 18, 'LFoot': 19, 'LFootTIP': 20,
'RHip': 7, 'RKnee': 8, 'RAnkle': 9, 'RFoot': 10, 'RFootTIP': 11,
'RShoulder': 3, 'RElbow': 4, 'RWrist': 5, 'RHand': 6,
'LShoulder': 12, 'LElbow': 13, 'LWrist': 14, 'LHand': 15,
}
wu_idx =     [0, 2,   16, 17, 18, 19,  7, 8, 9, 10,    3,  4, 5, 6,    12, 13, 14, 15]
wu_smpl_idx =[0, 12,  1, 4, 7, 10,     2, 5, 8, 11,   17, 19, 21, 23,  16, 18, 20, 22]

# NTU Dataset Indices
NTU_JOINT_MAP = {
'MidHip': 0, 'spine2': 1, 'Neck': 20, 'Head': 2,
'LHip': 16, 'LKnee': 17, 'LAnkle': 18, 'LFoot': 19,
'RHip': 12, 'RKnee': 13, 'RAnkle': 14, 'RFoot': 15,
'LShoulder': 8, 'LElbow': 9, 'LWrist': 10, 'LHand': 11,
'RShoulder': 4, 'RElbow': 5, 'RWrist': 6, 'RHand': 7,
}
ntu_idx = 		[0,1,20,2,		12,13,14,15,	16,17,18,19,	4,5,6,		8,9,10]
ntu_smpl_idx =	[0,6,12,15,  	2,5,8,11,   	1,4,7,10,   	16,18,20,	17,19,21]

# CMU Dataset Indices
CMU_JOINT_MAP = {
'MidHip': 2, 
'LHip': 12, 'LKnee': 13, 'LAnkle': 14,
'RHip': 6, 'RKnee': 7, 'RAnkle': 8,
'RShoulder': 3, 'RElbow': 4, 'RWrist': 5,
'LShoulder': 9, 'LElbow': 10, 'LWrist': 11,
}
cmu_idx = [2, 3, 4, 5, 9, 10, 11, 6, 7, 8, 12, 13, 14] 
cmu_smpl_idx =[0, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]


SMPL_MODEL_DIR = "./SMPLmodel/models/"
GMM_MODEL_DIR = "./SMPLmodel/models/"
SMPL_MEAN_FILE = "./SMPLmodel/neutral_smpl_mean_params.h5"