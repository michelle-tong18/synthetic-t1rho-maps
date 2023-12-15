import numpy as np
from scipy.io import loadmat
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

import os, sys
directory = os.path.abspath('')
sys.path.append(directory) # setting path can also append directory.parent
import utils

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


#This is the Pytorch Dataset class, which we manipulate to create a class that extracts from our split the relevant
#data, and returns the appropriate T2/T1rho maps as images and labels. The class needs an __init__ function that 
#initializes the class, a __getitem__ function that returns an item at a given index, and a __len__ function that 
#returns the length of the given dataset
class T1RT2Data(Dataset):
    def __init__(self, labels_df, set_name, transforms = None, scale_options=[3,0,150]):
        
        #Extract from the input df the subset of it that corresponds to train, val or test
        self.set_num = -1
        if set_name == 'train':
            self.labels_df = labels_df.loc[labels_df.set == 0, :]
            self.set_num = 0
        elif set_name == 'val':
            self.labels_df = labels_df.loc[labels_df.set == 1, :]
            self.set_num = 1
        elif set_name == 'test':
            self.labels_df = labels_df.loc[labels_df.set == 2, :]
            self.set_num = 2
        elif set_name == 'new_distribution':
            self.labels_df = labels_df.loc[labels_df.set == -1, :]
            self.set_num = -1
        elif set_name == 'bilateral_knee':
            self.labels_df = labels_df.loc[labels_df.set == -3, :]
            self.set_num = -2
        else:
            print("Wrong set name was given")

        self.t2_path  = list(self.labels_df['t2'])
        self.t1r_path = list(self.labels_df['t1r'])
        self.seg_path = list(self.labels_df['cartilage mask'])
        self.e1_path = list(self.labels_df['e1'])
        self.len      = len(self.t2_path)
        self.slice    = list(self.labels_df['slice number'])
        self.transforms = transforms
        self.scale_options = scale_options
        
        
    def __getitem__(self, index, vol_flag=False):
        'Generates one sample of data'
        #Get paths to t1rho, t2 maps and load the maps
        full_t2_path = self.t2_path[index].replace('knee_mri6','knee_mri9')
        full_t1r_path= self.t1r_path[index].replace('knee_mri6','knee_mri9')
        full_seg_path= self.seg_path[index]
        slice_num = self.slice[index]
        
        t2_map = utils.open_int2(full_t2_path).astype(np.float64)
        t1r_map= utils.open_int2(full_t1r_path).astype(np.float64)

        seg_extension = os.path.splitext(full_seg_path)[1]
        if (seg_extension == '.mat'):
            try:
                seg = loadmat(f'{full_seg_path}')['mask']
                seg = np.sum(seg,axis=3)
            except:
                seg = loadmat(f'{full_seg_path}')['seg']
                if seg.shape[2] > t1r_map.shape[2]:
                    num_diff_slices = seg.shape[2]-t1r_map.shape[2]
                    seg = seg[:,:,int(np.ceil(num_diff_slices/2)):int((t1r_map.shape[2]+np.ceil(num_diff_slices/2)))]
                    seg = np.flip(seg,axis = 2) 
        elif (seg_extension == '.hdf5'):
            f = h5py.File(full_seg_path, 'r')
            seg = f['prediction'][()].astype(np.float64)
            seg = seg.transpose((1,2,0))
        elif (seg_extension == '.int2'):
            seg = utils.open_int2_v2(full_seg_path)
            
        #For a 2D network, you need your dataset class to return data in 3 dimensions: channel, dimX, dimY.
        #For non-imaging applications, images are often RGB, so a 224x224 image, for example, will be 3x224x224.
        #Because we are working with medical images, we need to add in this extra dimension
        
        #For now, just setting up a 2D network for demonstration; pulling the central slice out
        if vol_flag == False:
            t2_map = np.expand_dims(t2_map[:,:,slice_num],axis = 0)
            t1r_map = np.expand_dims(t1r_map[:,:,slice_num],axis = 0)
            seg = np.expand_dims(seg[:,:,slice_num],axis = 0)
        
        #Always scale image inputs to your network; makes training more consistent, robust, and adjusts better
        #for eventual real-world data!
        
        if self.scale_options[0] == 'percentile':
            #normalize by the middle 95th percentile signal intensity
            perctl_low=self.scale_options[1]
            perctl_high=self.scale_options[2]
            t2_map_scaled  = (t2_map - np.percentile(t2_map,perctl_low))/(np.percentile(t2_map,perctl_high)-np.percentile(t2_map,perctl_low))
            t1r_map_scaled = (t1r_map - np.percentile(t1r_map,perctl_low))/(np.percentile(t1r_map,perctl_high)-np.percentile(t1r_map,perctl_low))
            low_thresh = np.percentile(t2_map,perctl_low)
            high_thresh = np.percentile(t2_map,perctl_high)
            
        elif self.scale_options[0] == 'clipAtPercentile':
            perctl_low=self.scale_options[1]
            perctl_high=self.scale_options[2]
            low_thresh = np.percentile(t2_map,perctl_low)
            high_thresh = np.percentile(t2_map,perctl_high)
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            t1r_map_scaled = utils.clip_volume(t1r_map,[low_thresh,high_thresh])
            
            #t2_map_scaled[t2_map_scaled < np.percentile(t2_map,2.5)] = 0
            #t1r_map_scaled[t1r_map < np.percentile(t2_map,2.5)] = 0
            
        elif self.scale_options[0] == 'clip':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            t1r_map_scaled = utils.clip_volume(t1r_map,[low_thresh,high_thresh])
        
        elif self.scale_options[0] == 'clipAndScale':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            t1r_map_scaled = utils.clip_volume(t1r_map,[low_thresh,high_thresh])
            t2_map_scaled  = t2_map_scaled/high_thresh
            t1r_map_scaled = t1r_map_scaled/high_thresh
            
        elif self.scale_options[0] == 'clip_MaxAjusted':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            adjust_val_t2 = self.scale_options[3]
            adjust_val_t1r = self.scale_options[4]
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            t1r_map_scaled = utils.clip_volume(t1r_map,[low_thresh,high_thresh])
            t2_map_scaled[t2_map_scaled==high_thresh]  = adjust_val_t2
            t1r_map_scaled[t1r_map_scaled==high_thresh] = adjust_val_t1r
        else:
            t1r_map_scaled = np.copy(t1r_map)
            t2_map_scaled = np.copy(t2_map)
            
        #Do specified transformation, if parameter is set. In order to use tensorvision transforms, the nxn grayscale
        #image is converted to a RBG 3xnxn image where the intensity values are repeated in the first dimension. This
        #np array is converted to a Torch.tensor for the transform and converted back to np.array after the transform.
        if self.transforms:
            t2_map_img = np.repeat(t2_map[np.newaxis, 0, :, :], 3, axis=0)
            t1r_map_img = np.repeat(t1r_map[np.newaxis, 0, :, :], 3, axis=0)
            seg_img = np.repeat(seg[np.newaxis, 0, :, :], 3, axis=0)

            transform_input = torch.from_numpy(np.stack([t2_map_img,t1r_map_img,seg_img],axis=0))
            transform_output = self.transforms(transform_input).numpy()
            
            t2_map = transform_output[0,:,:,:]
            t1r_map = transform_output[1,:,:,:]
            seg = transform_output[2,:,:,:]
        
        #Since pytorch is only able to return 2 outputs in a cetain format, combine t1r map and seg in separate 
        #channels to return in the label
        #the network will try to predict the first input in the label [0,:,:]
        label = np.concatenate((t1r_map_scaled,seg),axis = 0)
        #Convert your numpy arrays into pytorch tensors and return
        image = torch.from_numpy(t2_map_scaled)
        label = torch.from_numpy(label)
        return image.type(torch.FloatTensor), label.type(torch.FloatTensor)
    
    def __len__(self):
        return self.len 
    
    def __getEcho__(self, index):
        full_e1_path = self.e1_path[index].replace('knee_mri6','knee_mri9')
        slice_num = self.slice[index]
        e1_img = open_int2(full_e1_path).astype(np.float64)
        e1_img = e1_img[:,:,slice_num]
        
        return e1_img
    
    def get_imgs_no_scaling(self,index):
        full_t2_path = self.t2_path[index].replace('knee_mri6','knee_mri9')
        full_t1r_path= self.t1r_path[index].replace('knee_mri6','knee_mri9')
        
        t2_map = utils.open_int2(full_t2_path).astype(np.float64)
        t1r_map= utils.open_int2(full_t1r_path).astype(np.float64)
        return t1r_map,t2_map
    
    def get_scaled_region(self,index):
        'Generates one sample of data'
        #Get paths to t1rho, t2 maps and load the maps
        full_t2_path = self.t2_path[index]
        full_t1r_path= self.t1r_path[index]
        full_seg_path= self.seg_path[index]
        slice_num = self.slice[index]
        
        t2_map = open_int2(full_t2_path).astype(np.float64)
        t1r_map= open_int2(full_t1r_path).astype(np.float64)
        
        seg_extension = os.path.splitext(full_seg_path)[1]
        if (seg_extension == '.mat'):
            try:
                seg = loadmat(f'{full_seg_path}')['mask']
                seg = np.sum(seg,axis=3)
            except:
                seg = loadmat(f'{full_seg_path}')['seg']
                if seg.shape[2] > t1r_map.shape[2]:
                    num_diff_slices = seg.shape[2]-t1r_map.shape[2]
                    seg = seg[:,:,math.ceil(num_diff_slices/2):(t1r_map.shape[2]+math.ceil(num_diff_slices/2))]
                    seg = np.flip(seg,axis = 2) 
        elif (seg_extension == '.hdf5'):
            f = h5py.File(full_seg_path, 'r')
            seg = f['prediction'][()].astype(np.float64)
            seg = seg.transpose((1,2,0))
        
        #For now, just setting up a 2D network for demonstration; pulling the central slice out
        t2_map = np.expand_dims(t2_map[:,:,slice_num],axis = 0)
        t1r_map = np.expand_dims(t1r_map[:,:,slice_num],axis = 0)
        seg = np.expand_dims(seg[:,:,slice_num],axis = 0)
        
        if self.scale_options[0] == 'percentile':
            #normalize by the middle 95th percentile signal intensity
            perctl_low=self.scale_options[1]
            perctl_high=self.scale_options[2]
            low_thresh = np.percentile(t2_map,perctl_low)
            high_thresh = np.percentile(t2_map,perctl_high)
            
        elif self.scale_options[0] == 'clipAtPercentile':
            perctl_low=self.scale_options[1]
            perctl_high=self.scale_options[2]
            low_thresh = np.percentile(t2_map,perctl_low)
            high_thresh = np.percentile(t2_map,perctl_high)
            
        elif self.scale_options[0] == 'clip':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
        
        elif self.scale_options[0] == 'clipAndScale':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
        
        elif self.scale_options[0] == 'clip_MaxAjusted':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            
        """testing: visualize where clipping is occuring"""
        t2_map_scaled_region = np.zeros(t2_map.shape)
        t2_map_scaled_region[seg==1] = -0.5
        t2_map_scaled_region[t2_map>high_thresh] = 1
        t2_map_scaled_region[t2_map<low_thresh] = -1
        
        t1r_map_scaled_region  = np.zeros(t1r_map.shape)
        t1r_map_scaled_region[seg==1] = -0.5
        t1r_map_scaled_region[t1r_map>high_thresh] = 1
        t1r_map_scaled_region[t1r_map<low_thresh] = -1
        
        return t2_map_scaled_region,t1r_map_scaled_region

#This is the Pytorch Dataset class, which we manipulate to create a class that extracts from our split the relevant
#data, and returns the appropriate T2/T1rho maps as images and labels. The class needs an __init__ function that 
#initializes the class, a __getitem__ function that returns an item at a given index, and a __len__ function that 
#returns the length of the given dataset
class T2Data(Dataset):
    def __init__(self, labels_df, transforms = None, scale_options=[3,0,150]):
        
        #Extract from the input df the subset of it that corresponds to train, val or test
        self.labels_df = labels_df

        self.t2_path  = list(self.labels_df['t2'])
        self.len      = len(self.t2_path)
        self.slice    = list(self.labels_df['slice number'])
        self.transforms = transforms
        self.scale_options = scale_options
        
        
    def __getitem__(self, index, vol_flag=False):
        'Generates one sample of data'
        #Get paths to t1rho, t2 maps and load the maps
        full_t2_path = self.t2_path[index].replace('knee_mri6','knee_mri9')
        slice_num = self.slice[index]
        
        t2_map = utils.open_int2(full_t2_path).astype(np.float64)
            
        #For a 2D network, you need your dataset class to return data in 3 dimensions: channel, dimX, dimY.
        #For non-imaging applications, images are often RGB, so a 224x224 image, for example, will be 3x224x224.
        #Because we are working with medical images, we need to add in this extra dimension
        
        #For now, just setting up a 2D network for demonstration; pulling the central slice out
        if vol_flag == False:
            t2_map = np.expand_dims(t2_map[:,:,slice_num],axis = 0)
        
        #Always scale image inputs to your network; makes training more consistent, robust, and adjusts better
        #for eventual real-world data!
        
        if self.scale_options[0] == 'percentile':
            #normalize by the middle 95th percentile signal intensity
            perctl_low=self.scale_options[1]
            perctl_high=self.scale_options[2]
            t2_map_scaled  = (t2_map - np.percentile(t2_map,perctl_low))/(np.percentile(t2_map,perctl_high)-np.percentile(t2_map,perctl_low))
            low_thresh = np.percentile(t2_map,perctl_low)
            high_thresh = np.percentile(t2_map,perctl_high)
            
        elif self.scale_options[0] == 'clipAtPercentile':
            perctl_low=self.scale_options[1]
            perctl_high=self.scale_options[2]
            low_thresh = np.percentile(t2_map,perctl_low)
            high_thresh = np.percentile(t2_map,perctl_high)
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            
            
        elif self.scale_options[0] == 'clip':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            
        
        elif self.scale_options[0] == 'clipAndScale':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            t2_map_scaled  = t2_map_scaled/high_thresh
            
        elif self.scale_options[0] == 'clip_MaxAjusted':
            low_thresh = self.scale_options[1]
            high_thresh = self.scale_options[2]
            adjust_val_t2 = self.scale_options[3]
            t2_map_scaled = utils.clip_volume(t2_map,[low_thresh,high_thresh])
            t2_map_scaled[t2_map_scaled==high_thresh]  = adjust_val_t2
        else:
            t2_map_scaled = np.copy(t2_map)
            
        #Do specified transformation, if parameter is set. In order to use tensorvision transforms, the nxn grayscale
        #image is converted to a RBG 3xnxn image where the intensity values are repeated in the first dimension. This
        #np array is converted to a Torch.tensor for the transform and converted back to np.array after the transform.
        if self.transforms:
            t2_map_img = np.repeat(t2_map[np.newaxis, 0, :, :], 3, axis=0)

            transform_input = torch.from_numpy(np.stack([t2_map_img],axis=0))
            transform_output = self.transforms(transform_input).numpy()
            
            t2_map = transform_output[0,:,:,:]
        
        #Since pytorch is only able to return 2 outputs in a cetain format, combine t1r map and seg in separate 
        #channels to return in the label
        #the network will try to predict the first input in the label [0,:,:]
        image = torch.from_numpy(t2_map_scaled)
        return image.type(torch.FloatTensor)
    
    def __len__(self):
        return self.len 
    