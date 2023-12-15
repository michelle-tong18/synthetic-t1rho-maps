**Synthesizing T1rho From T2 Maps for Knee MRI**

*Michelle Tong (michelle.tong2@ucsf.edu)*

*2022*

Conda Environment: 
- environment_droplet.yml
- requirements.txt 

Training
1. Run *split_images.ipynb* or create your own splitting code. Split data based on so that each patient is exclusively in one split and so that metadata is similar across splits. Create a dataframe with a row for each 2D slice that goes into the model. Save splits in a new .csv. 

    Input: Create a csv file with paths to the dataset and the following columns and entires
    - cartilage mask slices: [min slice with a segmentation, max slice with a segmentation], [int, int]
    - (optional) any exclusion criteria: 1 to exclude
    - patient: unique identified for patient, string
    - t2: path to 3D T2 map file, string
    - t1r: path to 3D T1rho map file, string
    - cartilage mask: path to 3D cartilage mask file, string
    - e1: path to 3D echo 1 image file, string
    
    *For demographics*
    - research study: specifiy data cohort, string
    - age: float
    - weight: float
    - sex: F or M, string

    Output: Csv with the same information and additional columns:
    - set: 0 for train, 1 for val, 2 for test
    - slice number: 2D slice to extract from the 3D volume

2. Run *training.ipynb*. Train model and hyperparameter tune. 

Evaluation
1. Run *eval.ipynb*. Run inference on the test set and evaluate performance. Requires T2 map, T1rho map, cartilage segmentation, and echo 1 image.

Inference
1. Run *infer.ipynb*. Run inference with no evaluation. Only requires T2 maps.

