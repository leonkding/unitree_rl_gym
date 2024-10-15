import numpy as np
import torch
import joblib

def load_target_jt(device, file, offset):

    with open(file, 'rb') as fo:
        data = joblib.load(fo)
   
    size = []
    l_max = 0
    num_seq = 0
    for k,v in data.items():
        num_seq += 1
        l = v['h1_pose'].shape[0]
        pos_dim = v['h1_pose'].shape[-1]
        size.append(l)
        if l > l_max:
            l_max = l
    target_jt = torch.zeros((num_seq, l_max, pos_dim)).to(device)
    
    i = 0
    for k,v in data.items():

        l = v['h1_pose'].shape[0]
        target_jt[i,:l] = torch.tensor(v['h1_pose']).to(device).float()
        i += 1

    #one_target_jt = torch.from_numpy(one_target_jt).to(device)
    #target_jt = one_target_jt.unsqueeze(0)
    #target_jt += offset

    size = torch.tensor(size).to(device)
    return target_jt, size


# def load_target_jt(device, file, offset):

#     with open(file, 'rb') as fo:
#         data = joblib.load(fo)
   
#     size = []
#     l_max = 0
#     num_seq = 0
#     for k,v in data.items():
#         num_seq += 1
#         l = v['pose_h1'].shape[0]
#         pos_dim = v['pose_h1'].shape[-1]
#         size.append(l)
#         if l > l_max:
#             l_max = l
#     target_jt = torch.zeros((num_seq, l_max, pos_dim)).to(device)
    
#     i = 0
#     for k,v in data.items():
#         l = v['pose_h1'].shape[0]
#         target_jt[i,:l] = torch.tensor(v['pose_h1']).to(device).float()
#         i += 1

# #     #one_target_jt = torch.from_numpy(one_target_jt).to(device)
# #     #target_jt = one_target_jt.unsqueeze(0)
# #     #target_jt += offset

#     size = torch.tensor(size).to(device)

#     return target_jt[2:3], size[2:3]

# def load_target_jt(device, file, offset):
#     one_target_jt = np.load(f"{file}").astype(np.float32)
#     one_target_jt = torch.from_numpy(one_target_jt).to(device)
#     target_jt = one_target_jt.unsqueeze(0)
#     target_jt += offset

#     size = torch.tensor([one_target_jt.shape[0]]).to(device)
# #     #print(target_jt.shape)
#     return target_jt, size