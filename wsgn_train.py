import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-model', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


from wsgn import WSGN, WSGN_2fc, WSGN_sigmoid

#from charades_dataset import Charades as Dataset
from charades_i3d_rgb_data import Charades as Dataset

def run(init_lr=1e-3, max_steps=10e3, mode='rgb', root='i3d_rgb_charades', train_split='charades/charades.json', batch_size=128, save_model='models/baseline/',model='wsgn'):
    # setup dataset
    dataset = Dataset(train_split, 'training', root, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    print("Dataset loaded.")

    # setup the model
    if model=='wsgn':
        wsgn = WSGN(num_classes=157, mode=mode)
        print("WSGN model set up.")
    if model=='WSGN_2fc':
        wsgn = WSGN_2fc(num_classes=157, mode=mode)
    if model=='WSGN_sigmoid':
        wsgn = WSGN_sigmoid(num_classes=157, mode=mode)
    #wsgn.load_state_dict(torch.load('models/wsgn/010000.pt'))
    wsgn.cuda()
    wsgn = nn.DataParallel(wsgn)
    

    lr = init_lr
    # optimizer = optim.SGD(wsgn.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,wsgn.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [])


    print("Start training")
    num_steps_per_update = 128 / batch_size # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)
        print('lr',lr)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                wsgn.train(True)
            else:
                wsgn.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                cls_score, loc_score = wsgn(inputs)
                
                score = cls_score*loc_score

                # compute classification loss (with softmax over classes then average along time B x C x T)
                cls_loss = F.binary_cross_entropy(score.mean(2), labels) * 157

                loss = cls_loss/num_steps_per_update
                tot_loss += loss.data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} Cls Loss: {:.8f}'.format(phase, tot_loss/10))
                        # save model
                        torch.save(wsgn.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = 0.
            if phase == 'val':
                print('{} Cls Loss: {:.8f}'.format(phase, (tot_loss*num_steps_per_update)/num_iter))
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model, model=args.model)
