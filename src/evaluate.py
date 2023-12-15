import pandas as pd
import numpy as np
from UNet import UNet
from torch.utils.data import Dataset, DataLoader

import os, sys
directory = directory = os.path.abspath('')
sys.path.append(directory) # setting path can also append directory.parent

import datasets
import utils
import metrics
import torch


#run this code on GPU if you can (much faster), otherwise run on CPU
if torch.cuda.is_available():
    dev = "cuda"
    device = torch.device(dev)
else:
    dev = "cpu"
    device = torch.device(dev)
    
"""
This function returns the predictions (outputs) and corresponding ground truth (labels) for all entries in a
dataloader, given a trained model

In addition, this version, unlike the one in the generator_only_train, will return the nrmse across all items in the
given dataloader
"""
def infer_preds(dataloader, model):
    count = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            if count == 0:
                images = data.to(device)
                outputs = model(images).cpu().detach()
            else:
                images_ = data.to(device)
                outputs_= model(images_).cpu().detach()
                
                images = torch.cat((images,images_),dim = 0)
                outputs = torch.cat((outputs,outputs_),dim = 0) 
            count += 1
    
    return outputs, images

def test_preds(dataloader, model):
    count = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            if count == 0:
                images, labels = data
                images = images.to(device)
                labels = labels
                outputs = model(images).cpu().detach()
            else:
                images_, labels_ = data
                images_ = images_.to(device)
                labels_ = labels_
                outputs_= model(images_).cpu().detach()
                
                images = torch.cat((images,images_),dim = 0)
                labels  = torch.cat((labels,labels_),dim = 0)
                outputs = torch.cat((outputs,outputs_),dim = 0) 
            count += 1
    
    return outputs, labels, images

def test_metrics(outputs, labels, scaleMetrics_options, inputs, df_patientInfo=pd.DataFrame(data=None)):
    count = 0
    
    nmse_list = []
    nmse_seg_list = []
    mean_pred_list = []
    mean_targ_list = []
    median_list = []
    corr_seg_list = []
    corr_seg_t1rt2_list = []
    psnr_list = []
    ssim_list = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    if df_patientInfo.empty:
        metric_iters = outputs.size(0)
    else:
        metric_iters = len(df_patientInfo)
        
    for ii in range(metric_iters):
        if df_patientInfo.empty:
            slices = np.copy(ii)
            
            pred  = outputs[slices,:,:,:].unsqueeze(0).cpu().detach().numpy()
            targ  = labels[slices,0,:,:].unsqueeze(0).unsqueeze(0).cpu().detach().numpy()
            seg   = labels[slices,1,:,:].unsqueeze(0).unsqueeze(0).cpu().detach().numpy()
            input_img = inputs[slices,:,:,:].unsqueeze(0).cpu().detach().numpy()
        else:
            slices = df_patientInfo['testSetSlice_inds'][ii]
            
            pred  = outputs[slices,:,:,:].squeeze(1).unsqueeze(0).cpu().detach().numpy()
            targ  = labels[slices,0,:,:].unsqueeze(0).cpu().detach().numpy()
            seg   = labels[slices,1,:,:].unsqueeze(0).cpu().detach().numpy()
            input_img = inputs[slices,:,:,:].squeeze(1).unsqueeze(0).cpu().detach().numpy()
        
        # Scale images
        if scaleMetrics_options[0]=='clip':
            si_range=scaleMetrics_options[1:3]
            pred = utils.clip_volume(pred,si_range)
            targ = utils.clip_volume(targ,si_range) 
            input_img = utils.clip_volume(input_img,si_range) 
        seg[seg > 1] = 1
        seg[seg < 1] = 0

        #NMSE
        nmse = metrics.calculate_NMSE(pred,targ)
        nmse_seg = metrics.calculate_NMSE(pred*seg,targ*seg)

        nmse_list.append(round(nmse,3))
        nmse_seg_list.append(round(nmse_seg,3))

        #MEAN, STD, MEDIAN
        mean_pred, stdev_pred, median_pred = metrics.calculate_stats(pred,seg)
        mean_targ, stdev_targ, median_targ = metrics.calculate_stats(targ,seg)

        mean_pred_list.append([round(mean_pred,2),round(stdev_pred,2)])
        mean_targ_list.append([round(mean_targ,2),round(stdev_targ,2)])
        median_list.append([round(median_pred,2),round(median_targ,2)])

        #CORR
        corrNum = metrics.calculate_corr(pred,targ,seg)
        corr_seg_list.append(round(corrNum,4))

        corrTrue = metrics.calculate_corr(targ,input_img,seg)
        corr_seg_t1rt2_list.append(round(corrTrue,4))
        
        #PEAK SNR
        psnrNum = metrics.calculate_peak_snr(pred,targ,seg)
        psnr_list.append(round(psnrNum,3))

        #SSIM
        ssimNum = metrics.ssim(torch.tensor(pred), torch.tensor(targ.squeeze(0)), window_size = 11, size_average = True)
        ssim_list.append(ssimNum.numpy().round(4))

    df_metrics = []

    col_names = ['nmse','nmse_seg','mean_pred','mean_targ','median_pred_targ','corr_seg_preds','corr_seg_t1rt2','psnr','ssim']
    df_metrics = pd.DataFrame(data=None,columns=col_names)

    df_metrics['nmse'] = nmse_list
    df_metrics['nmse_seg']  = nmse_seg_list
    df_metrics['mean_pred'] = mean_pred_list
    df_metrics['mean_targ'] = mean_targ_list
    df_metrics['median_pred_targ'] = median_list
    df_metrics['corr_seg_preds'] = corr_seg_list
    df_metrics['corr_seg_t1rt2'] = corr_seg_t1rt2_list
    df_metrics['psnr'] = psnr_list
    df_metrics['ssim'] = ssim_list
    
    return df_metrics


def get_scan_info(test_df):
    df_dataset = test_df.reset_index(drop=True)
    df_dataset.rename(columns={'session number':'scan_idx'},inplace=True) #scan_idx = idx out of total number of visits
    df_dataset.rename(columns={'idx':'splitDfSlice_inds'},inplace=True) #splitDF_Idx = index in the split train/val/test file
    df_dataset['testSetSlice_inds'] = [i for i in range(len(df_dataset))] #subsetScan_idx = idx out of total number of visits in test

    # Identify the number of scans and associated indices for each patient
    #patient_num_scans = df_dataset.groupby('patient')['idx'].size()
    #patient_scan_indicies = df_dataset.groupby('patient')['idx'].apply(list)

    df_set_index = df_dataset.groupby(['scan_idx'])['testSetSlice_inds'].apply(list)
    patient_session = df_dataset.groupby(['scan_idx'])['splitDfSlice_inds'].apply(list)
    patient_list = df_dataset.groupby('scan_idx')['patient'].apply(list)
    for ii in patient_list.index:
        patient_list[ii] = list(set(patient_list[ii]))
    total_scans = df_dataset.groupby('scan_idx')['patient'].size()
    research_study = df_dataset.groupby('scan_idx')['research study'].apply(list)
    for ii in research_study.index:
        research_study[ii] = list(set(research_study[ii]))[0]
    corr_t1r_t2_true = df_dataset.groupby('scan_idx')['corr t2 t1r seg 150 thresh'].apply(list)
    for ii in corr_t1r_t2_true.index:
        corr_t1r_t2_true[ii] = list(set(corr_t1r_t2_true[ii]))[0]


    #patient
    frames                 = [         patient_list, total_scans,   patient_session,    df_set_index,    research_study,corr_t1r_t2_true]
    df_patientInfo = pd.concat(frames,axis=1).reset_index()
    df_patientInfo.columns = ['scan_idx','patient','total_slices','splitDfSlice_inds','testSetSlice_inds','study','t1r_t2_vol_true_corr']
    if 0:
        print('num test slices = '+ str(len(test_df)))
        print('Are all the test slices accounted for in the dataframe testSetSlice_inds? ')
        print(len(test_df) == (df_patientInfo.loc[:,'testSetSlice_inds'].str.len().sum()))

    if 1:
        df_patientInfo.loc[:,'study'] = df_patientInfo.loc[:,'study'].replace({'vanitha2': 'vanitha'}, regex=True)
        df_patientInfo.loc[:,'study'] = df_patientInfo.loc[:,'study'].replace({'AF_HSS': 'AF_UCSF'}, regex=True)
        df_patientInfo.loc[:,'study'] = df_patientInfo.loc[:,'study'].replace({'AF_MAYO': 'AF_UCSF'}, regex=True)
        print('num test scan visits (some patients have multiple visits) = ' + str(len(df_patientInfo)))

    return df_patientInfo

"""
Given the csv of all trained models and a row index, this function will get the predictions for it and return the
train losses, validation losses, and nrmse
"""
def get_model_performance(row_idx,df_modelInfo):
    log_file   = df_modelInfo['Log_Path'].iloc[row_idx]
    
    #Get your training and validation losses from the training log file
    f         = open(log_file,'r')
    log_text  = f.read()
    f.close()

    # read loss and similairty metrics from the text file
    train_loss= np.array([float(i.split(',')[0]) for i in log_text.split('train Loss: ')[1:]])
    val_loss  = np.array([float(i.split(',')[0]) for i in log_text.split('val Loss: ')[1:]])
    best_loss  = np.array([float(i.split(',')[0]) for i in log_text.split('Best loss: ')[1:]])
    best_loss  = np.array([float(i.split(',')[0]) for i in log_text.split('Best loss: ')[1:]])
    best_loss  = np.array([float(i.split(',')[0]) for i in log_text.split('Best loss: ')[1:]])
    
    best_loss_num = round(best_loss[0],4)
    best_metrics = [(i.split('  \n')[0]) for i in log_text.split('val Loss: '+f'{best_loss_num:.4f}'+',')][1]
    best_ssim = [float(i.split(',')[0]) for i in best_metrics.split('SSIM: ')[1:]][0]
    best_nrmse = [float(i.split(',')[0]) for i in best_metrics.split('NRMSE: ')[1:]][0]
    best_nrmse_seg = [float(i.split(',')[0]) for i in best_metrics.split('NRMSE_seg: ')[1:]][0]

    return train_loss,val_loss,best_loss,best_ssim,best_nrmse,best_nrmse_seg


def get_model_preds(row_idx,df_modelInfo,subset='test'):
    #Get the path to your model and log file of interest
    model_path = df_modelInfo['Model_Path'].iloc[row_idx]
    log_file   = df_modelInfo['Log_Path'].iloc[row_idx]
    split_path = df_modelInfo['Split_Path'][row_idx]
    
    num_epochs          =df_modelInfo['epochs'][row_idx]
    lr                  =df_modelInfo['lr'][row_idx]
    scheduler_options   =list(eval(df_modelInfo['scheduler:type,step,%lossChange,%lrDecrease'][row_idx].replace(" ","")[1:-1]))
    loss_options        =list(eval(df_modelInfo['loss:type,metric,wSeg,wNotSeg'][row_idx].replace(" ","")[1:-1]))
    scaleInput_options  =list(eval(df_modelInfo['scaleInputs:method,min,max'][row_idx].replace(" ","")[1:-1]))
    scaleMetrics_options=list(eval(df_modelInfo['scaleMetrics:method,min,max'][row_idx].replace(" ","")[1:-1]))
    do_augmentation     =df_modelInfo['augment'][row_idx]

    #Load splits
    label_df = pd.read_csv(split_path)
    
    # determine whether to augment data
    #if do_augmentation == True:
    #    transform_type = transform
    #else:
    #    transform_type = None
    
    #Load CNN of choice and assign to correct device
    generator_model = UNet(init_features = 64, in_channels = 1, out_channels = 1)
    
    if dev == "cuda":
        generator_model.to(device)
        generator_model = torch.load(model_path) # get weights
    else:
        generator_model = torch.load(model_path,map_location='cpu') # get weights
    
    # load the training, validation, and test sets using the dataloader
    #trainset = datasets.T1RT2Data(labels_df=label_df, set_name='train', transforms=transform_type, scale_options=scaleInput_options)
    if subset == 'val':
        testset   = datasets.T1RT2Data(labels_df=label_df, set_name='val', scale_options=scaleInput_options)
    elif subset == 'test':
        testset  = datasets.T1RT2Data(labels_df=label_df, set_name='test', scale_options=scaleInput_options)
    elif subset == 'new_distribution':
        testset  = datasets.T1RT2Data(labels_df=label_df, set_name='new_distribution', scale_options=scaleInput_options)
    elif subset == 'bilateral_knee':
        testset  = datasets.T1RT2Data(labels_df=label_df, set_name='bilateral_knee', scale_options=scaleInput_options)

    val_batch_size = 16
    num_workers    = 1
    #trainset_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=num_workers,
    #                             drop_last=True)
    #valset_loader   = DataLoader(valset, batch_size=val_batch_size,shuffle=False,num_workers=num_workers,
    #                             drop_last=False)
    testset_loader  = DataLoader(testset, batch_size=val_batch_size,shuffle=False,num_workers=num_workers,
                                 drop_last=False)
    
    #Get your predictions (from the test dataset if that's what you're looking for)
    preds, labels, images = test_preds(testset_loader, generator_model)

    return preds,labels,images,testset

def get_model_stats(preds,labels,images,scaleMetrics_options,df_patientInfo):
    
    df_metrics_slice = test_metrics(preds,labels,scaleMetrics_options,images)
    df_metrics_vol = test_metrics(preds,labels,scaleMetrics_options,images,df_patientInfo)
    
    return df_metrics_slice, df_metrics_vol

def get_model_infers(label_df,model_path,scaleInput_options):
    #Get the path to your model and log file of interest
    
    #Load CNN of choice and assign to correct device
    generator_model = UNet(init_features = 64, in_channels = 1, out_channels = 1)
    
    if dev == "cuda":
        generator_model.to(device)
        generator_model = torch.load(model_path) # get weights
    else:
        generator_model = torch.load(model_path,map_location='cpu') # get weights
    
    # load the training, validation, and test sets using the dataloader
    inferset   = datasets.T2Data(labels_df=label_df, scale_options=scaleInput_options)

    inferset_loader  = DataLoader(inferset, batch_size=16,shuffle=False,num_workers=1,
                                 drop_last=False)
    
    #Get your predictions (from the test dataset if that's what you're looking for)
    preds, images = infer_preds(inferset_loader, generator_model)

    return preds,images,inferset