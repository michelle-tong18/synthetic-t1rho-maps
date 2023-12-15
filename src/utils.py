import numpy as np

#Dataloader that opens up the int2 files and returns them with dimensions (256 x 256 x number of slices)
def open_int2(path_,dim_x=256, dim_y=256):
    img_raw = np.fromfile(path_, dtype='>i2')
    _img = img_raw.reshape((dim_x,dim_y,-1),order='F')
    _img = np.rot90(_img, axes=(1,0))
    _img = np.flip(_img, axis=1)
    return _img

#Dataloader that opens up the int2 files and returns them with dimensions (256 x 256 x number of slices)
#For KICK cartilage segmentations
def open_int2_v2(path_,dim_x=256, dim_y=256):
    img_raw = np.fromfile(path_, dtype='>i2')
    _img = img_raw.reshape((dim_x,dim_y,-1),order='F')
    _img = np.rot90(_img, axes=(1,0))
    _img = np.flip(_img, axis=1)
    _img = np.flip(_img, axis=2)
    return _img

def clip_volume(imgs,si_range):
    min_SI = si_range[0]
    max_SI = si_range[1]
    
    if isinstance(imgs,np.ndarray):
        imgs_scaled = np.ones(imgs.shape)*imgs
    else:
        imgs_scaled = imgs
    imgs_scaled[imgs_scaled>max_SI] = max_SI
    imgs_scaled[imgs_scaled<min_SI] = min_SI
    return imgs_scaled