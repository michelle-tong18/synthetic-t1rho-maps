import matplotlib.pyplot as plt
import numpy as np

import os, sys
directory = os.path.abspath('')
sys.path.append(directory) # setting path can also append directory.parent

import datasets

#Plot images in dataloader
def plot_dataloader(dataset,ind,max_map):
    
    image,label = dataset.__getitem__(ind)
    
    fig  = plt.figure(figsize = (12,5))
    plt.rcParams.update({'font.size': 12})
    plt.subplot(121)
    im1 = plt.imshow(label[2,:,:].numpy(),cmap = 'jet')
    #Recall that we scaled the T2 maps such that the middle 95% of pixel values lie between 0 and 1. So whereas for the
    #T1rho maps, our values of interest may be from 0 to 100, for the scaled version, they'll be from 0 to something 
    #between 0 and 1. I sort of arbitrarily set this to 0.15, but you can play around with this--there's likely a more
    #consistent and better way to display this data
    plt.clim([0,max_map])
    plt.axis('off')
    plt.title('T2 Map')

    plt.subplot(122)
    im2 = plt.imshow(label[1,:,:].numpy(),cmap = 'jet')
    plt.clim([0,max_map])
    plt.axis('off')
    plt.title('T1rho Map')

    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.128, 0.01, 0.75])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('T2 relaxation time (ms)',rotation = 270,labelpad = 20)

    fig  = plt.figure(figsize = (12,5))
    plt.rcParams.update({'font.size': 12})
    plt.subplot(121)
    im1 = plt.imshow(image[0,:,:].numpy(),cmap = 'jet')
    #Region where we clipped the images
    plt.clim([0,1])
    plt.axis('off')
    plt.title('T2 Map Clipped')

    plt.subplot(122)
    im2 = plt.imshow(label[0,:,:].numpy(),cmap = 'jet')
    plt.clim([0,1])
    plt.axis('off')
    plt.title('T1rho Map Clipped')

    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.128, 0.01, 0.75])
    cbar = fig.colorbar(im2, cax=cbar_ax)

    


#Plots the training and validation loss curves for the training of the given model
def plot_losses(train_loss_list,val_loss_list,run_list=[]):
    train_color_list = ['blue','red','green','tab:orange','cyan','magenta','yellow']
    val_color_list = ['blue','red','green','tab:orange','cyan','magenta','yellow']
    if len(run_list)==0:
        run_list=np.arange(1,len(train_loss_list)+1)
          
    fig = plt.figure(figsize = (10,5))
    for ii in range(len(train_loss_list)):
        train_loss = train_loss_list[ii]
        val_loss = val_loss_list[ii]
        epochs_list = np.arange(1,len(train_loss)+1)
        
        plt.plot(epochs_list,train_loss,'--',label = 'train '+str(run_list[ii]),color=train_color_list[ii])
        plt.plot(epochs_list,val_loss,':', label = 'val '+str(run_list[ii]),color=val_color_list[ii])
    
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model loss curves')
    
    return fig


#Returns a matplotlib figure with predicted and ground truth maps for a list of indices inds
def plot_maps(preds,labels,dataset,inds,maxDisp):
    #note dataset can only be the val or train set where the data is not shuffled
    print(inds)
    """Plot images and ground truth"""
    for ind in inds:
        start = time.time()
        image,label = dataset.__getitem__(ind)
        echo_1 = dataset.__getEcho__(ind)
        #image_vol,label_vol = dataset.__getitem__(ind,vol_flag=True)
        end   = time.time()
        print('Image loading time: '+str(np.round(end-start,3))+' seconds')
        
        fig  = plt.figure(figsize = (16,10))
        plt.rcParams.update({'font.size': 12})
        
        T2_map = image[0,:,:].numpy()
        T1r_map = labels[ind,0,:,:].cpu().detach().numpy()
        T1r_map_pred = preds[ind,0,:,:].cpu().detach().numpy()
        seg_ = labels[ind,3,:,:].cpu().detach().numpy().astype(np.float32)
        seg_[seg_ == 0] = np.nan
        e1_img = echo_1[:,:]
        
        T2_map = clip_volume(T2_map,[0,maxDisp])
        T1r_map = clip_volume(T1r_map,[0,maxDisp])
        T1r_map_pred = clip_volume(T1r_map_pred,[0,maxDisp])
        
        #T2_map_vol = image_vol[0,:,:,:].cpu().detach().numpy()
        #T1r_map_vol = label_vol[0,:,:,:].cpu().detach().numpy()
        #T1r_map_pred = 
        corr_pred = calculate_corr(T1r_map_pred,T1r_map,seg_)
        corr_t2 = calculate_corr(T2_map,T1r_map,seg_)
        nrmse_= calculate_NMSE(T1r_map_pred,T1r_map)
        
        plt.subplot(2,3,1)
        im1 = plt.imshow(T2_map,cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('Input $T_{2}$ Map\ntest set idx '+str(ind))
        
        plt.subplot(2,3,2)
        im2 = plt.imshow(T1r_map,cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('Ground Truth $T_{1}rho$ Map\ntest set idx '+str(ind))
        #plt.text(130,275,,horizontalalignment='center', verticalalignment='center')
        
        plt.subplot(2,3,3)
        im3 = plt.imshow(T1r_map_pred,cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('Predicted $T_{1}rho$ Map\ntest set idx '+str(ind))
        #plt.text(130,275,,horizontalalignment='center', verticalalignment='center')
        
        # Cartilage maps overlayed on echo 1
        plt.subplot(2,3,4)
        im4 = plt.imshow(e1_img,cmap = 'gray')
        #plt.clim([-0.5,0.5])
        im4 = plt.imshow(np.multiply(T2_map,seg_),cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('full slice nMSE = '+str(np.round(nrmse_,3)))
        
        plt.subplot(2,3,5)
        im5 = plt.imshow(e1_img,cmap = 'gray')
        #plt.clim([-0.5,0.5])
        im5 = plt.imshow(np.multiply(T1r_map,seg_),cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('seg corr($T_{2}$,$T_{1}rho$)= '+str(np.round(corr_pred,3)))

        plt.subplot(2,3,6)
        im6 = plt.imshow(e1_img,cmap = 'gray')
        #plt.clim([-1,1])
        im6 = plt.imshow(np.multiply(T1r_map_pred,seg_),cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('seg corr (Pred,Truth)= '+str(np.round(corr_t2,3)))
        
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.01, 0.76])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('$T_{2}$ & $T_{1}rho$ relaxation time (ms)',rotation = 270,labelpad = 20)
    
    return fig


#Returns a matplotlib figure with predicted and ground truth maps for a list of indices inds
def plot_compare_maps(inds, preds_plot_list,label_plot, maxDisp):

    n_cases = len(preds_plot_list)
    for ind in inds:
        fig  = plt.figure(figsize = (12,5*n_cases))
        plt.rcParams.update({'font.size': 12})
        count = 0
        for ii in range(n_cases):
            preds_plot = preds_plot_list[ii]
            if ii==0:
                preds_plot1 = preds_plot[ind,0,:,:].cpu().detach().numpy().copy()
            
            plt.subplot(n_cases,3,count+1)
            im1 = plt.imshow(label_plot[ind,0,:,:].cpu().detach().numpy(),cmap = 'jet')
            plt.clim([0,maxDisp])
            plt.axis('off')
            plt.title('Ground Truth T1rho Map\n test set idx '+str(ind))
            

            plt.subplot(n_cases,3,count+2)
            im2 = plt.imshow(preds_plot[ind,0,:,:].cpu().detach().numpy(),cmap = 'jet')
            plt.clim([0,maxDisp])
            plt.axis('off')
            plt.title('Predicted T1rho Map: '+str(ii)+'\n test set idx '+str(ind))
        
            plt.subplot(n_cases,3,count+3)
            img = np.abs(preds_plot[ind,0,:,:].cpu().detach().numpy()-preds_plot1)
            im3 = plt.imshow(img,cmap = 'jet')
            plt.axis('off')
            plt.title('Diff T1rho Map abs(#'+str(ii)+'-#0)\n test set idx'+str(ind))
            count += 3
    
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, 0.13, 0.01, 0.75])
        cbar = fig.colorbar(im3, cax=cbar_ax)
        cbar.set_label('T1rho relaxation time abs diff (ms)',rotation = 270,labelpad = 20)

    return fig


#Returns a matplotlib figure with predicted and ground truth maps for a list of indices inds
def plot_compare_maps_horiz(inds, preds_plot_list,label_plot,dataset,maxDisp,run_num_list=None):
    if len(run_num_list)==0:
        run_num_list=list(range(len(preds_plot_list)))
    
    n_cases = len(preds_plot_list)+1
    for ind in inds:
        #load echo1 and seg
        start = time.time()
        image,label = dataset.__getitem__(ind)
        echo_1 = dataset.__getEcho__(ind)
        end   = time.time()
        print('Image loading time: '+str(np.round(end-start,3))+' seconds')
        
        T2_map = image[0,:,:].numpy()
        seg_ = labels[ind,3,:,:].cpu().detach().numpy().astype(np.float32)
        seg_[seg_ == 0] = np.nan
        e1_img = echo_1[:,:]
        
        
        fig  = plt.figure(figsize = (8*n_cases,26))
        plt.rcParams.update({'font.size': 12})
        count = 1
        
        plt.subplot(4,n_cases,count)
        im1 = plt.imshow(label_plot[ind,0,:,:].cpu().detach().numpy(),cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        plt.title('Ground Truth T1rho Map\ntest set idx '+str(ind),fontsize=20)
        
        plt.subplot(4,n_cases,count+n_cases)
        im3 = plt.imshow(e1_img,cmap = 'gray')
        im3 = plt.imshow(np.multiply(label_plot[ind,0,:,:].cpu().detach().numpy(),seg_),cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        corr_t2 = calculate_corr(T2_map,label_plot[ind,0,:,:],seg_)
        plt.title('seg corr (T2 Truth,T1r Truth)= '+str(np.round(corr_t2,3)),fontsize=20)
        
        plt.subplot(4,n_cases,count+2*n_cases)
        im6 = plt.imshow(e1_img,cmap = 'gray')
        im6 = plt.imshow(np.multiply(T2_map,seg_),cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        corr_t2 = calculate_corr(T2_map,label_plot[ind,0,:,:],seg_)
        plt.title('Ground Truth T2 Map\ntest set idx '+str(ind),fontsize=20)
        
        plt.subplot(4,n_cases,count+3*n_cases)
        im7 = plt.imshow(e1_img,cmap = 'gray')
        im7 = plt.imshow(T2_map,cmap = 'jet')
        plt.clim([0,maxDisp])
        plt.axis('off')
        corr_t2 = calculate_corr(T2_map,label_plot[ind,0,:,:],seg_)
        plt.title('Ground Truth T2 Map\ntest set idx '+str(ind),fontsize=20)
        
        ref_plot1 = label_plot[ind,0,:,:].cpu().detach().numpy().copy()
        for ii in range(len(preds_plot_list)):
            preds_plot = preds_plot_list[ii]
            count += 1
            if ii==0:
                ref_plot2 = preds_plot[ind,0,:,:].cpu().detach().numpy().copy()
            
            plt.subplot(4,n_cases,count)
            im2 = plt.imshow(preds_plot[ind,0,:,:].cpu().detach().numpy(),cmap = 'jet')
            plt.clim([0,maxDisp])
            plt.axis('off')
            plt.title('Predicted T1rho Map: '+str(run_num_list[ii])+'\n test set idx '+str(ind),fontsize=18)
            
            plt.subplot(4,n_cases,count+n_cases)
            im3 = plt.imshow(e1_img,cmap = 'gray')
            im3 = plt.imshow(np.multiply(preds_plot[ind,0,:,:].cpu().detach().numpy(),seg_),cmap = 'jet')
            plt.clim([0,maxDisp])
            plt.axis('off')
            corr_pred = calculate_corr(preds_plot[ind,0,:,:],label_plot[ind,0,:,:],seg_)
            plt.title('seg corr (Pred,Truth)= '+str(np.round(corr_pred,3)),fontsize=20)
            

            plt.subplot(4,n_cases,(count+n_cases*2))
            img = np.abs(preds_plot[ind,0,:,:].cpu().detach().numpy()-ref_plot1)
            img = np.multiply(img,seg_)
            im4 = plt.imshow(img,cmap = 'jet')
            plt.clim([0,12])
            plt.axis('off')
            plt.title('Diff T1rho Map abs(#'+str(run_num_list[ii])+'-truth)\n test set idx'+str(ind),fontsize=18)
            
            plt.subplot(4,n_cases,(count+n_cases*3))
            img = np.abs(preds_plot[ind,0,:,:].cpu().detach().numpy()-ref_plot2)
            img = np.multiply(img,seg_)
            im5 = plt.imshow(img,cmap = 'jet')
            plt.clim([0,12])
            plt.axis('off')
            plt.title('Diff T1rho Map abs(#'+str(run_num_list[ii])+'-#0)\n test set idx'+str(ind),fontsize=18)
               
        fig.subplots_adjust(right=0.83)
        """ 
        #Two axes
        cbar_ax = fig.add_axes([0.85, 0.53, 0.01, 0.35])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('T1rho relaxation time (ms)',rotation = 270,labelpad = 20)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.01, 0.35])
        cbar = fig.colorbar(im3, cax=cbar_ax)
        cbar.set_label('Abs diff in relaxation time (ms)',rotation = 270,labelpad = 20)
        """
        #Three axes
        cbar_ax = fig.add_axes([0.85, 0.52, 0.01, 0.37])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('T1rho relaxation time (ms)',rotation = 270,labelpad = 20,fontsize=18)
        cbar_ax = fig.add_axes([0.85, 0.32, 0.01, 0.18])
        cbar = fig.colorbar(im4, cax=cbar_ax)
        cbar.set_label('Abs diff in ground truth-pred (ms)',rotation = 270,labelpad = 20,fontsize=18)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.01, 0.18])
        cbar = fig.colorbar(im5, cax=cbar_ax)
        cbar.set_label('Abs diff in pred1-pred (ms)',rotation = 270,labelpad = 20,fontsize=18)
        
    return fig

def blandAltman_plot(pred_metric_list,targ_metric_list,study_list):
    """
    This function takes predicted and target data to generate N bland altman plots. 
    Points in the plot can be color coded as specified by the study_list variable.
    
    Input:
    1. pred_metric_list = (N,) list where each entry is a np.array of predicted data for subplot n.
    2. targ_metric_list = (N,) list where each entry is a np.array of target data for subplot n.
    3. study_list       = (N,) where each entry is a list of strings to indicate the plot point color.
    Note, each np.array may be a different size to handle uneven number of points in each group. For 
    example pred_metric_list[0] = [50x1], targ_metric_list[0] = [50x1], & study_list[0] = [50x1]  
    while pred_metric_list[1] = [48x1], targ_metric_list[1] = [48x1], & study_list[1] = [48x1]  
    
    TODO Modifications:
    1. Modify the colors variable to a dictionary with key-value pairs linking the strings from the
       study_list entries to python colors.
    2. Update the titles variable to a dictionary with key-value pairs linking the strings from the
       study_list entries to titles for the subplot. The first datapoint will specify the title.
    3. (Optional) the fg_color and txt_color variable can be updated to generate plots for white or 
       black backgrounds
    
    Output:
    - BA plot with N subplots, one for each np array of data in the list. The subplot will be laid 
      out in a [2 x ceil(N/2)] grid.
    
    """
    colors = {'P50_ACL':'dodgerblue', 'AF_UCSF':'m', 'AF_MAYO':'m', 'AF_HSS':'m',
          'vanitha':'limegreen','vanitha2':'limegreen','KICK':'orange'}
    titles = {'P50_ACL':'UCSF', 'AF_UCSF':'Multi-Center', 
              'AF_MAYO':'Mayo Research', 'AF_HSS':'HSS Research',
          'vanitha':'Clinical','vanitha2':'Clinical','KICK':'Bilateral Research'}
    
    pred_all = np.concatenate(pred_metric_list,axis=0)
    targ_all = np.concatenate(targ_metric_list,axis=0)
    x_lims = [np.min(np.mean([pred_all,targ_all],axis=0))-2,
             np.max(np.mean([pred_all,targ_all],axis=0))+2]
    y_lim = np.max(np.abs(pred_all-targ_all))
    y_lims = [(y_lim+2)*-1,y_lim+2]
    
    fg_color = 'white'
    txt_color = 'black'
    ft_sz = 16
    ft_sz_mean = 14
    #fig  = plt.figure(figsize = (8,5))
    fig = plt.figure(figsize=(18,10*np.ceil(len(pred_metric_list)/3)))
    fig = plt.figure(figsize=(20,5*np.ceil(len(pred_metric_list)/3)))
    
    ax = plt.gca()
    fig.set_facecolor(fg_color)
    count = 1
    for ii in range(len(pred_metric_list)):
        pred_metric = pred_metric_list[ii]
        targ_metric = targ_metric_list[ii]
        study_metric = study_list[ii]
        
        # Bland Altman plot
        mean = np.mean([pred_metric,targ_metric], axis=0)
        diff = np.array(pred_metric) - np.array(targ_metric)
        md   = np.mean(diff)                   # Mean of the difference
        sd   = np.std(diff, axis=0)            # Standard deviation of the difference
        
        
        plt.subplot(np.ceil(len(pred_metric_list)/2),3,count)
        plt.scatter(mean, diff,c=study_metric.map(colors),s=19)
        plt.axhline(md,           color='gray', linestyle='-')
        plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
        #plt.title('Bland-Altman Plot for $T_{1}rho$ in cartilage')
        
        plt.xlim(x_lims)
        plt.ylim(y_lims)

        xlabel_pos = plt.xticks()[0][-2]
        plt.title(study_metric.map(titles).tolist()[0]+' Study',color=txt_color,fontsize=18)
        if ii==2:
            plt.title('All Studies',color=txt_color,fontsize=18)
        if ii==0:
            plt.title('Clinical Data',color=txt_color,fontsize=18)
        if ii==0:
            plt.xlabel('Mean $T_{1}rho$ [ms]',color=txt_color, fontsize=ft_sz)
        if ii==0:
            plt.ylabel('Predicted - Target $T_{1}rho$ [ms]',color=txt_color, fontsize=ft_sz)

        # set tick and ticklabel color
        plt.tick_params(color=txt_color, labelcolor=txt_color)
        plt.xticks(fontsize=ft_sz-2)
        plt.yticks(fontsize=ft_sz-2)
        
        plt.text(0, 0, 'mean diff:'+"\n"+"{:.2f}".format(md), fontsize=ft_sz_mean,
            horizontalalignment='right',
            verticalalignment='center',
            position=(xlabel_pos,md))
        plt.text(0.97, 0.94, '+SD*1.96:'+"\n"+"{:.2f}".format(md + 1.96*sd), fontsize=ft_sz_mean,
            horizontalalignment='right',
            verticalalignment='center',
            position=(xlabel_pos,(md + 1.96*sd)))
        plt.text(0.97, 0.373, '-SD*1.96:'+"\n"+"{:.2f}".format(md - 1.96*sd), fontsize=ft_sz_mean,
            horizontalalignment='right',
            verticalalignment='center',
            position=(xlabel_pos,(md - 1.96*sd)))

        count+=1
        
    return plt
        
def corr_plot(pred_metric_list,targ_metric_list,study_list):
    """
    This function takes predicted and target data to generate N correlation plots. 
    Points in the plot can be color coded as specified by the study_list variable.
    
    Input:
    1. pred_metric_list = (N,) list where each entry is a np.array of predicted data for subplot n.
    2. targ_metric_list = (N,) list where each entry is a np.array of target data for subplot n.
    3. study_list       = (N,) where each entry is a list of strings to indicate the plot point color.
    Note, each np.array may be a different size to handle uneven number of points in each group. For 
    example pred_metric_list[0] = [50x1], targ_metric_list[0] = [50x1], & study_list[0] = [50x1]  
    while pred_metric_list[1] = [48x1], targ_metric_list[1] = [48x1], & study_list[1] = [48x1]  
    
    TODO Modifications:
    1. Modify the colors variable to a dictionary with key-value pairs linking the strings from the
       study_list entries to python colors.
    2. Update the titles variable to a dictionary with key-value pairs linking the strings from the
       study_list entries to titles for the subplot. The first datapoint will specify the title.
    3. (Optional) the fg_color and txt_color variable can be updated to generate plots for white or 
       black backgrounds
    
    Output:
    - Corr plot with N subplots, one for each np array of data in the list. The subplot will be laid 
      out in a [2 x ceil(N/2)] grid.
    
    """
    
    colors = {'P50_ACL':'dodgerblue', 'AF_UCSF':'m', 'AF_MAYO':'m', 'AF_HSS':'m',
          'vanitha':'limegreen','vanitha2':'limegreen'}
    titles = {'P50_ACL':'UCSF', 'AF_UCSF':'Multi-Center', 
              'AF_MAYO':'Mayo Research', 'AF_HSS':'HSS Research',
          'vanitha':'Clinical','vanitha2':'Clinical'}
    
    pred_all = np.concatenate(pred_metric_list,axis=0)
    targ_all = np.concatenate(targ_metric_list,axis=0)
    
    xy_lims = [np.floor(np.min([pred_all,targ_all])),np.ceil(np.max([pred_all,targ_all]))]
    
    fg_color = 'white'
    txt_color = 'black'
    ft_sz = 18
    fig, ax = plt.subplots(figsize=(14,6*np.ceil(len(pred_metric_list)/2)))
    ax = plt.gca()
    fig.set_facecolor(fg_color)
    count = 1
    for ii in range(len(pred_metric_list)):
        pred_metric = pred_metric_list[ii]
        targ_metric = targ_metric_list[ii]
        study_metric = study_list[ii]
        
        # Correlation plot
        plt.subplot(2,np.ceil(len(pred_metric_list)/2),count)
        scatter = plt.scatter(targ_metric,pred_metric,c=study_metric.map(colors),s=19)
        xy_line = np.linspace(xy_lims[0],xy_lims[1],10)
        plt.plot(xy_line,xy_line,'--',color=(0.5, 0.5, 0.5),label='x=y')
        plt.xlim(xy_lims)
        plt.ylim(xy_lims)
        # set tick and ticklabel color
        plt.tick_params(color=txt_color, labelcolor=txt_color)
        plt.xticks(fontsize=ft_sz-2)
        plt.yticks(fontsize=ft_sz-2)
        
        plt.legend()
        plt.title(study_metric.map(titles).tolist()[0]+' Study',color=txt_color, fontsize=ft_sz)
        plt.xlabel('Mean Ground Truth $T_{1}rho$ [ms]',color=txt_color, fontsize=ft_sz)
        plt.ylabel('Mean Predicted $T_{1}rho$ [ms]',color=txt_color, fontsize=ft_sz)
        count+=1
        
    return plt

#Returns a matplotlib figure with predicted and ground truth maps for a list of indices inds
def plot_corr_slice(preds,labels,dataset,inds,maxDisp,color):
    
    txt_color = 'black'
    #note dataset can only be the val or train set where the data is not shuffled
    print(inds)
    """Plot images and ground truth"""
    for ind in inds:
        start = time.time()
        image,label = dataset.__getitem__(ind)
        echo_1 = dataset.__getEcho__(ind)
        #image_vol,label_vol = dataset.__getitem__(ind,vol_flag=True)
        end   = time.time()
        print('Image loading time: '+str(np.round(end-start,3))+' seconds')
        
        fig  = plt.figure(figsize = (12,10))
        plt.rcParams.update({'font.size': 12})
        
        T2_map = image[0,:,:].numpy()
        T1r_map = labels[ind,0,:,:].cpu().detach().numpy()
        T1r_map_pred = preds[ind,0,:,:].cpu().detach().numpy()
        seg_ = labels[ind,3,:,:].cpu().detach().numpy().astype(np.float32)
        seg_[seg_ == 0] = np.nan
        e1_img = echo_1[:,:]
        
        T2_map = clip_volume(T2_map,[0,maxDisp])
        T1r_map = clip_volume(T1r_map,[0,maxDisp])
        T1r_map_pred = clip_volume(T1r_map_pred,[0,maxDisp])
        
        corr_pred = calculate_corr(T1r_map_pred,T1r_map,seg_)
        corr_t2 = calculate_corr(T2_map,T1r_map,seg_)
        
        T2_map_voxels = T2_map[seg_==1]
        T1r_map_voxels = T1r_map[seg_==1]
        T1r_map_pred_voxels = T1r_map_pred[seg_==1]     

        xy_lims = [0,105]
        plt.subplot(1,2,1, aspect='equal')
        scatter = plt.scatter(T1r_map_voxels,T2_map_voxels,c=color,s=12)
        xy_line = np.linspace(xy_lims[0],xy_lims[1],10)
        plt.plot(xy_line,xy_line,'--',color=(0.5, 0.5, 0.5),label='x=y')
        plt.xlim(xy_lims)
        plt.ylim(xy_lims)
        plt.tick_params(color=txt_color, labelcolor=txt_color)
        plt.legend()
        plt.title('Target vs Input Intensities in Cartilage',color=txt_color)
        plt.xlabel('Ground Truth $T_{1}rho$ [ms]',color=txt_color)
        plt.ylabel('Ground Truth $T_{2}$ [ms]',color=txt_color)      
        
        plt.subplot(1,2,2, aspect='equal')
        scatter = plt.scatter(T1r_map_voxels,T1r_map_pred_voxels,c=color,s=12)
        xy_line = np.linspace(xy_lims[0],xy_lims[1],10)
        plt.plot(xy_line,xy_line,'--',color=(0.5, 0.5, 0.5),label='x=y')
        plt.xlim(xy_lims)
        plt.ylim(xy_lims)
        plt.tick_params(color=txt_color, labelcolor=txt_color)
        plt.legend()
        plt.title('Predicted vs Target Intensities in Cartilage',color=txt_color)
        plt.xlabel('Ground Truth $T_{1}rho$ [ms]',color=txt_color)
        plt.ylabel('Predicted $T_{1}rho$ [ms]',color=txt_color)
        
    return plt

