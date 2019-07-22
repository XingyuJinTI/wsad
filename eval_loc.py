import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)
parser.add_argument('-model', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



import numpy as np

from wsgn import WSGN, WSGN_2fc

from charades_i3d_rgb_data_for_eval import Charades as Dataset


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))
            
def map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """ 
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1)==0
    fix[empty, :] = np.NINF
    return map(fix, gt_array)

def run(max_steps=64e3, mode='rgb', root='i3d_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir='', model=''):
    # setup dataset
    #test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

#    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
#    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = Dataset(split, 'testing', root, mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)    

#    dataloaders = {'train': dataloader, 'val': val_dataloader}
#    datasets = {'train': dataset, 'val': val_dataset}
    print("dataset Done")
    
    # setup the model
    if model=='wsgn':
        wsgn = WSGN(num_classes=157, mode=mode)
    if model=='WSGN_2fc':
        wsgn = WSGN_2fc(num_classes=157, mode=mode)
    wsgn.load_state_dict(torch.load(load_model))
    wsgn.cuda()
    wsgn = nn.DataParallel(wsgn)
    print("WSGN model set up.")

    outputs = []
    gts = []
    ids = []
    wsgn.train(False)  # Set model to evaluate mode
                
    # Iterate over data.
    for data in val_dataloader:
        # get the inputs
        inputs, labels, name = data
        t = inputs.shape[2]

            # (C x T x 1)

        # wrap them in Variable
        inputs = Variable(inputs.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)
        cls_score, loc_score = wsgn(inputs)
        score = cls_score*loc_score
        #score = F.upsample(score, t*8, mode='linear')

        # store predictions
        for i in range(25):
            
            outputs.append(score[:,:,i*t//25].squeeze(0).data.cpu().numpy())
            gts.append(labels[:,:,i].squeeze(0).data.cpu().numpy())
            ids.append(name[0]+' '+str(i+1))


    #mAP, _, ap = map.map(np.vstack(outputs), np.vstack(gts))
    mAP, _, ap = charades_map(np.vstack(outputs), np.vstack(gts))
    print(ap)
    print(' * mAP {:.9f}'.format(mAP))
    submission_file(
        ids, outputs, args.save_dir)
    return mAP



if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir, model=args.model)
