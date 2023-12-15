import torch
import time
import copy

import os, sys
directory = os.path.abspath('')
sys.path.append(directory) # setting path can also append directory.parent

import datasets
import utils
import metrics

#run this code on GPU if you can (much faster), otherwise run on CPU
if torch.cuda.is_available():
    dev = "cuda"
    device = torch.device(dev)
else:
    dev = "cpu"
    device = torch.device(dev)

#Sets up the loss function that you'll use. This is a very basic L1 loss function, but this is definitely something
#you'll want to explore to see what works best
class NetLoss(torch.nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        self.reduction= None
    def forward(self,pred, target, reduction = 0, loss_options=['l2','NRMSE',0,1]):
        #the pred and target_img should already be thresholded prior to calling NetLoss
        target_img = target[:,0,:,:]; 
        seg = target[:,3,:,:]; 
        not_seg = torch.sub(1,seg)
        
        
        # define target map here to look at the loss. 
        # We can try weighting the entire volume, cartilage region, combination of both?
        
        #modify loss here
        #normalized loss functions
        #l1_loss     = torch.mean(torch.abs(targ_map - pred_map)) #l1 norm, entire volume
        #l2_loss     = torch.mean(torch.square(targ_map - pred_map)) # l2 norm, entire volume
        
        #segmentation loss for NRMSE metric
        if loss_options[1]=='NRMSE':
            abs_diff_map = torch.abs(target_img - pred)
            l1_not_seg_loss = torch.div(torch.sum(abs_diff_map.mul(not_seg)),torch.sum(not_seg)) # l2 norm only in cartilage segmentation
            l2_not_seg_loss = torch.div(torch.sum(torch.square(abs_diff_map).mul(not_seg)),torch.sum(not_seg)) # l2 norm only in cartilage segmentation
            seg_num_pixels = torch.sum(seg)
            if seg_num_pixels > 0:
                l1_seg_loss = torch.div(torch.sum(abs_diff_map.mul(seg)),seg_num_pixels) #l1 norm only in cartilage segmentation
                l2_seg_loss = torch.div(torch.sum(torch.square(abs_diff_map).mul(seg)),seg_num_pixels) # l2 norm only in cartilage segmentation
            else:
                l1_seg_loss = 0
                l2_seg_loss = 0
        else:
            print('error, selection loss metric')
            
        w_seg = loss_options[2]
        w_not_seg = loss_options[3]
        if loss_options[0]=='l1':
            loss = l1_seg_loss*w_seg+l1_not_seg_loss*w_not_seg
        elif loss_options[0]=='l2':
            loss = l2_seg_loss*w_seg+l2_not_seg_loss*w_not_seg
        
        return loss
    

"""
This is the engine of all your code -- it does your training!

Inputs:
-Dataloaders: a dictionary with 2 entries, 'train' and 'val', where both entries are the corresponding pytorch
dataloader classes
-Model: the model you'll be training (i.e. UNet or whatever else you might want)
-Optimizer: the method you'll be using to update weights in the model as you train (i.e. Adam, stochastic gradient
descent, batch gradient descent, etc.)
-Scheduler: the method you'll be using to adjust your learning rate as you go through your training
-Log_save_file: the path to the text file to which you'll save a log of this training (keeps track of your losses
and whatever other metrics you want to track, at each epoch)
-Max_map: upper bound T1rho value in the maps at which you cap your predicted and ground truth maps
-num_epochs: number of iterations of the training data your model will see
"""
def train_model(dataloaders, model, criterion, optimizer, scheduler, log_save_file, 
                loss_options, scaleMetric_options, scheduler_options, num_epochs=25, return_stats=False):
    f = open(log_save_file,"w+")
    
    since = time.time()
    best_model_wts   = copy.deepcopy(model.state_dict())
    best_loss        = 1e10
    epoch_losses     = {'train': [], 'val': []}
    epoch_ssims      = {'train': [], 'val': []}
    epoch_nrmses     = {'train': [], 'val': []}
    epoch_nrmses_seg = {'train': [], 'val': []}
    epoch_corrs_seg   = {'train': [], 'val': []}
        
    #Iterate through all your data for the specified number of epochs
    for epoch in range(num_epochs):
        if epoch > 0:
            f.write('----------------------------------------------------- \n')
            print('--' * 30)
        f.write('Epoch {}/{} \n'.format(epoch+1, num_epochs))
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        epoch_start = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss     = 0.0
            running_ssim     = 0.0
            running_nrmse    = 0.0
            running_nrmse_seg= 0.0
            running_corr_seg = 0.0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)#gets model outputs given the inputs
                    
                    for entry in range(inputs.size(0)):
                        # Scale images
                        pred = outputs[entry,:,:,:].unsqueeze(0)
                        targ = labels[entry,0,:,:].unsqueeze(0).unsqueeze(0)
                        seg  = labels[entry,1,:,:].unsqueeze(0).unsqueeze(0)
                        
                        if scaleMetric_options[0]=='clip':
                            si_range = scaleMetric_options[1:3]
                            pred = utils.clip_volume(pred,si_range)
                            targ = utils.clip_volume(targ,si_range) 
                        seg[seg > 1] = 1
                        seg[seg < 1] = 0

                        outputs[entry,:,:,:] = pred
                        labels[entry,0,:,:] = targ
                        labels[entry,1,:,:] = seg
                        
                    loss    = criterion(outputs, labels, loss_options)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()#Calculate all the gradients at the different weights
                        optimizer.step()#Update the weights

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                #calculate the ssim, nrmse for each entry and add
                for entry in range(inputs.size(0)):
                    # Scale images
                    pred = outputs[entry,:,:,:].unsqueeze(0)
                    targ = labels[entry,0,:,:].unsqueeze(0).unsqueeze(0)
                    seg  = labels[entry,1,:,:].unsqueeze(0).unsqueeze(0)

                    if scaleMetric_options[0]=='clip':
                        si_range = scaleMetric_options[1:3]
                        pred = utils.clip_volume(pred,si_range)
                        targ = utils.clip_volume(targ,si_range) 
                    seg[seg > 1] = 1
                    seg[seg < 1] = 0
                    
                    # calculate metrics
                    running_ssim += metrics.ssim(pred,targ).cpu().detach().numpy()
                    running_nrmse += metrics.calculate_NMSE(pred.cpu().detach().numpy(),targ.cpu().detach().numpy())
                    running_nrmse_seg += metrics.calculate_NMSE(torch.mul(pred,seg).cpu().detach().numpy(),torch.mul(targ,seg).cpu().detach().numpy())
                    running_corr_seg += metrics.calculate_corr(pred.cpu().detach().numpy(),targ.cpu().detach().numpy(),seg.cpu().detach().numpy())
                
                
            # Update the learning rate based on the percent change in loss
            if phase == 'train':
                
                if scheduler_options[0]=='percentLoss':
                    percent_change_thresh=scheduler_options[2]
                    percent_decrease = scheduler_options[3]
                    
                    #if the percent change in loss is less than 10%, reduce the learning rate by 10%
                    epoch_loss = running_loss / len(dataloaders[phase])
                    lr = scheduler.get_last_lr()[0]
                    print('lr: '+str(lr))
                    if epoch > 0:
                        percent_change_loss = (previous_epoch_loss-epoch_loss)/epoch_loss*100 
                        print('% change loss: '+'{:.2f}'.format(percent_change_loss))
                        if percent_change_loss <= percent_change_thresh:
                            lr = lr*percent_decrease
                        optimizer = torch.optim.Adam(generator_model.parameters(), lr=lr, weight_decay = 0)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_options[1], gamma=0.1)
                    previous_epoch_loss = copy.deepcopy(epoch_loss)
                else:
                    scheduler.step()
            
            #store the losses, ssim, and nrmse of the given epoch
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_losses[phase].append(epoch_loss)
            epoch_ssim = running_ssim / len(dataloaders[phase])
            epoch_ssims[phase].append(epoch_ssim)
            epoch_nrmse= running_nrmse/ len(dataloaders[phase])
            epoch_nrmses[phase].append(epoch_nrmse)
            epoch_nrmse_seg = running_nrmse_seg/ len(dataloaders[phase])
            epoch_nrmses_seg[phase].append(epoch_nrmse_seg)
            epoch_corr_seg = running_corr_seg / len(dataloaders[phase])
            epoch_corrs_seg[phase].append(epoch_corr_seg)
            
            #write a summary of this epoch to the training log file
            f.write('{} Loss: {:.4f}, SSIM: {:.4f}, NRMSE: {:.4f}, NRMSE_seg: {:.4f}, corr_seg: {:.4f} \n'.format(phase, epoch_loss, epoch_ssim, epoch_nrmse, epoch_nrmse_seg, epoch_corr_seg))
            print('{} loss: {:.4f}, ssim: {:.4f}, nrmse: {:.4f}, nrmse_seg: {:.4f}, corr_seg: {:.4f}'.format(phase, epoch_loss, epoch_ssim, epoch_nrmse, epoch_nrmse_seg, epoch_corr_seg))
                
            # If the loss obtained during this epoch is less than the previous best loss, update the best model weights
            if phase == 'val' and epoch_loss < best_loss:
                f.write('')
                print('Updating best model')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('Epoch time: {:.0f}m {:.0f}s  \n'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since
    f.write('Training complete in {:.0f}m {:.0f}s  \n'.format(time_elapsed // 60, time_elapsed % 60))
    f.write('Best loss: {:4f}  \n'.format(best_loss))
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (6400 % 3600) % 60))
    print('Best loss: {:4f}'.format(best_loss))

    f.close()
    # load best model weights so that the optimal model can be returned
    model.load_state_dict(best_model_wts)

    if return_stats:
        return model, epoch_losses
    else:
        return model