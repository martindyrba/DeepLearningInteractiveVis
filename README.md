# Deep Learning Interactive Visualization

This folder contains all code to learn a deep learning model to detect Alzheimer's disease and visualize contributing brain regions with high relevance.
The model structure has higher stability for whole-brain data as the first model which used only on hippocampal coronal slices as input (reduced field-of-view).  
Publishing of the results is submitted (Dec. 2020).


### CNN model training and performance evaluation

Order of script execution:
1. [CreateResiduals-ADNI2-StoreModels.ipynb](CreateResiduals-ADNI2-StoreModels.ipynb) and other scripts for the validation samples [CreateResiduals-ADNI3.ipynb](CreateResiduals-ADNI3.ipynb) (execution time: each 15-30 minutes)
2. [DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_dr0.1.ipynb](DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_dr0.1.ipynb) for model training based on twenty-fold cross-validation to evaluate general model accuracy (execution time: 2-10 hrs with CUDA-GPU?)
3. [CalcAccuracyPerGroup_ADNI2.ipynb](CalcAccuracyPerGroup_ADNI2.ipynb) and [CalcAccuracyPerGroup_ADNI2_Amy.ipynb](CalcAccuracyPerGroup_ADNI2_Amy.ipynb) to calculate the accuracy/AUC per comparison MCI vs. CN and AD vs. CN
4. [DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationADNI3_dr0.1.ipynb](DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationADNI3_dr0.1.ipynb), [DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationAIBL_dr0.1.ipynb](DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationAIBL_dr0.1.ipynb) for calculating model performance on the validation datasets (execution time: each 0.5-1 hr with CUDA-GPU?)
5. [extract_hippocampal_activation_newmodel.ipynb](extract_hippocampal_activation_newmodel.ipynb) to extract the hippocampal activity for all CV models (execution time: 10 minutes with CUDA-GPU)
6. [extract_relevance_maps_as_nifti.ipynb](extract_relevance_maps_as_nifti.ipynb) to extract the relevance map overlays as nifti file for all participants/scans for a single model (execution time: 30 minutes with CUDA-GPU)


### Running the interactive visualization

The interactive bokeh application [InteractiveVis](InteractiveVis) can be run for inspecting the created relevance maps overlaid on the original input images.

To run it, there are two options.

1. Point the anaconda prompt to the DeepLearningInteractiveVis main directory and then run bokeh using:
```
bokeh serve InteractiveVis --show
```

2. Alternatively download the docker container from https://nextcloud.dzne.de/index.php/s/MPWSDzKgfJWCH5W  
Then use the scripts ```install_docker_image_intvis.sh```, ```run_docker_intvis.sh```, and ```stop_docker_intvis.sh``` to install, run, and stop the Bokeh app, respectively.

After running Bokeh, the app will be available from your web browser: [http://localhost:5006/InteractiveVis](http://localhost:5006/InteractiveVis)


![Screenshot of the InteractiveVis app](Screenshot_InteractiveVis.png)*Screenshot of the InteractiveVis app*


***


### Requirements and installation:

To be able to run the visualization software, you need Python 2 or 3, specifically <3.8, in order to install tensorflow==1.15  
Note: on some systems it is recommended to install some dependencies using the default package manager instead of pip. e.g.
`sudo apt-get install python-numpy python-scipy python-tk`
or
`sudo yum install scipy numpy tkinter`

Run pip/pip3 to install the dependencies:
`pip install -r requirements.txt`  
Note: create a new Python environment first to avoid messing up your installed Python modules/versions when you have other coding projects.


### License:

Copyright (c) 2020 Martin Dyrba martin.dyrba@dzne.de, German Center for Neurodegenerative Diseases (DZNE), Rostock, Germany

This project and included source code is published under the MIT license. See [LICENSE](LICENSE) for details.
