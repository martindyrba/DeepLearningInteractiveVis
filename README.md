# Deep Learning Interactive Visualization

This project contains all code to learn a deep learning model to detect Alzheimer's disease and visualize contributing brain regions with high relevance.
The model structure has higher stability for whole-brain data as the first model which used only on hippocampal coronal slices as input (reduced field-of-view).  
Publishing of the results is submitted (Dec. 2020). You can find the preprint on [arXiv:2012.10294](https://arxiv.org/abs/2012.10294)

![Screenshot of the InteractiveVis app](Screenshot_InteractiveVis.png)*Screenshot of the InteractiveVis app*


***



### Running the interactive visualization

The interactive bokeh application [InteractiveVis](InteractiveVis) can be run for inspecting the created relevance maps overlaid on the original input images.

To run it, there are three options.

1. We set up a public web service to quickly try it out: [https://explaination.net/InteractiveVis](https://explaination.net/InteractiveVis)

2. Alternatively download the docker container from DockerHub: ```sudo docker pull martindyrba/interactivevis```
Then use the scripts ```sudo ./run_docker_intvis.sh``` and ```sudo ./stop_docker_intvis.sh``` to run or stop the Bokeh app. (You find both files above in this repository.)  
After starting the docker container, the app will be available from your web browser: [http://localhost:5006/InteractiveVis](http://localhost:5006/InteractiveVis)

3. Download this Git repository. Install the required Python modules (see below). Then point the anaconda prompt to the DeepLearningInteractiveVis main directory and run the Bokeh app using:
```
bokeh serve InteractiveVis --show
```



### Requirements and installation:

To be able to run the interactive visualization, you need Python 2 or 3 (specifically Python <3.8, in order to install tensorflow==1.15)  
Note: on some systems it is recommended to install some dependencies using the default package manager instead of pip. e.g.
`sudo apt-get install python-numpy python-scipy python-tk`
or
`sudo yum install scipy numpy tkinter`

Also, it is recommended to first create a new Python environment (using [virtualenv/venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)) to avoid messing up your local Python modules/versions when you have other coding projects or a system shared by various users.

Run pip/pip3 to install the dependencies:
`pip install -r requirements.txt`  
Then you can start the Bokeh application as indicated above.



***



### CNN model training and performance evaluation

The code for training the CNN models and evaluation is provided in this repository.  
The order of script execution was as follows:

1. [CreateResiduals-ADNI2-StoreModels.ipynb](CreateResiduals-ADNI2-StoreModels.ipynb) and other scripts for the validation samples [CreateResiduals-ADNI3.ipynb](CreateResiduals-ADNI3.ipynb) (execution time: each 15-30 minutes)
2. [DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_dr0.1.ipynb](DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_dr0.1.ipynb) for model training based on twenty-fold cross-validation to evaluate general model accuracy (execution time: 2-10 hrs with CUDA-GPU?)
3. [CalcAccuracyPerGroup_ADNI2.ipynb](CalcAccuracyPerGroup_ADNI2.ipynb) and [CalcAccuracyPerGroup_ADNI2_Amy.ipynb](CalcAccuracyPerGroup_ADNI2_Amy.ipynb) to calculate the accuracy/AUC per comparison MCI vs. CN and AD vs. CN
4. [DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationADNI3_dr0.1.ipynb](DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationADNI3_dr0.1.ipynb), [DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationAIBL_dr0.1.ipynb](DeepLearning3DxVal_wb_win_mwp1_MNI_newStructure_validationAIBL_dr0.1.ipynb) for calculating model performance on the validation datasets (execution time: each 0.5-1 hr with CUDA-GPU?)
5. [extract_hippocampal_activation_newmodel.ipynb](extract_hippocampal_activation_newmodel.ipynb) to extract the hippocampal activity for all CV models (execution time: 10 minutes with CUDA-GPU)
6. [extract_relevance_maps_as_nifti.ipynb](extract_relevance_maps_as_nifti.ipynb) to extract the relevance map overlays as nifti file for all participants/scans for a single model (execution time: 30 minutes with CUDA-GPU)
7. [CreateDemoDataset.ipynb](CreateDemoDataset.ipynb) to create the example files being used by the InteractiveVis demo. It contains a sample of 15 people per diagnostic group, representatively selected from the ADNI-2 phase based on the criteria: amyloid status (positive for Alzheimer's dementia and amnestic mild cogntive impairment, negative for controls), MRI field strength of 3 Tesla, RID greater than 4000, and age of 65 or older. 



***



### InteractiveVis architecture overview

*InteractiveVis UML class diagram (v4)*

![InteractiveVis class diagram (v4)](InteractiveVis_class_diagram_v4.svg)

*Select subject UML sequence diagram (v3)*

![Select subject sequence diagram (v3)](select_subject_sequence_diagram_v3.svg)



***



### License:

Copyright (c) 2020 Martin Dyrba martin.dyrba@dzne.de, German Center for Neurodegenerative Diseases (DZNE), Rostock, Germany

This project and included source code is published under the MIT license. See [LICENSE](LICENSE) for details.
