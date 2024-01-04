
#!/usr/bin/env python
# coding: utf-8

import glob
import os
import json

debug = False
selected_neuron = 1  # switch to neuron 0 if required
adaptive_relevance_scaling = False # fixed relevance score adjustment to range: abs(relevance)~=[0,1] if set to false, otherwise scale depending on min/max
scale_factor = 2.5  # scale factor for brain views
disable_gpu_for_tensorflow = True  # running only on CPU will usually speed up processing, as model/data conversion to GPU is omitted

background_images_path = 'orig_images_demo.hdf5'  # original images
residuals_path = 'residuals_demo.hdf5'  # stored residuals, replace by original images if desired
linear_model_path = 'linear_models_ADNI2.hdf5' # stored linear model coefficients -> set to None to skip residualization

# define list of available models and the model selected by default
stored_models = sorted(
        glob.glob(os.path.join('model_checkpoints','*model_wb_whole_ds.hdf5'))) + sorted(
        glob.glob(os.path.join('model_checkpoints','*.best.hdf5'))) # get list of available models
selected_model = os.path.join('model_checkpoints','resmodel_wb_whole_ds.hdf5') # model file name to load by default
do_model_prefetch = False # load ALL models already at application startup --> this will take LONG and is only advised for long-running web/app servers
flip_left_right_in_frontal_plot = False

# define path/name to excel file and the sheet name containing covariates
covariates_excel_file = 'results/hippocampus_volume_relevance_ADNI2.xlsx'
covariates_excel_sheet = 'ADNI2_LRP_CMP'

# load language files in a dictionary
translations = {}
lang_module = 'InteractiveVis/lang'
for lang_file in os.listdir(lang_module):
    language = os.path.splitext(lang_file)[0]  # Get the filename without the extension
    with open(os.path.join(lang_module, lang_file), encoding='utf-8') as file:
        translations[language] = json.load(file)

# Scan for nifti file names
#data_files = sorted(glob.glob('mwp1_MNI_demo/*/*.nii.gz'))
data_files = ['mwp1_MNI_demo/AD/mwp1AD_4001_037_MR_MT1__GradWarp__N3m_Br_20120412210525847_S100198_I297310.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4192_006_MR_MT1__N3m_Br_20111008152714764_S123807_I260245.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4209_116_MR_MT1__GradWarp__N3m_Br_20110906122954211_S120580_I254813.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4211_137_MR_MT1__GradWarp__N3m_Br_20110928093359933_S122836_I258692.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4215_098_MR_MT1__GradWarp__N3m_Br_20110928090225943_S121941_I258658.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4252_019_MR_MT1__N3m_Br_20120412203204509_S123970_I297290.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4258_137_MR_MT1__GradWarp__N3m_Br_20111015081754925_S125049_I261075.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4282_094_MR_MT1__GradWarp__N3m_Br_20111030173944058_S125980_I263736.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4307_029_MR_MT1__GradWarp__N3m_Br_20111121114941112_S130156_I267783.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4373_003_MR_MT1__GradWarp__N3m_Br_20120113181946124_S134985_I278123.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4477_019_MR_MT1__N3m_Br_20120130095540673_S138533_I281432.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4494_126_MR_MT1__GradWarp__N3m_Br_20120203153249265_S138968_I282672.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4500_127_MR_MT1__GradWarp__N3m_Br_20120216103509168_S140201_I285127.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4501_023_MR_MT1__GradWarp__N3m_Br_20120327104944026_S144533_I293685.nii.gz',
                'mwp1_MNI_demo/AD/mwp1AD_4526_123_MR_MT1__GradWarp__N3m_Br_20120216102639746_S140267_I285119.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4018_098_MR_MT1__GradWarp__N3m_Br_20110427153029046_S105025_I229328.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4020_023_MR_MT1__GradWarp__N3m_Br_20110505163410626_S107436_I233460.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4021_031_MR_MT1__N3m_Br_20110504105111444_S105268_I232862.nii.gz', 
                'mwp1_MNI_demo/CN/mwp1HC_4032_031_MR_MT1__N3m_Br_20110518141022326_S108840_I235655.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4037_041_MR_MT1__GradWarp__N3m_Br_20110602153315246_S109770_I238644.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4041_041_MR_MT1__GradWarp__N3m_Br_20110602154114697_S109867_I238651.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4043_116_MR_MT1__GradWarp__N3m_Br_20110524151658413_S109124_I236974.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4060_041_MR_MT1__GradWarp__N3m_Br_20110623104345546_S110364_I241342.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4066_941_MR_MT1__GradWarp__N3m_Br_20110623112737722_S112064_I241396.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4075_011_MR_MT1__GradWarp__N3m_Br_20110623111851601_S111256_I241381.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4076_099_MR_MT1__GradWarp__N3m_Br_20120405191022964_S111440_I296345.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4177_033_MR_MT1__GradWarp__N3m_Br_20110906122456523_S120558_I254809.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4200_041_MR_MT1__GradWarp__N3m_Br_20111008152511012_S123030_I260243.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4218_031_MR_MT1__N3m_Br_20110928090432395_S122024_I258660.nii.gz',
                'mwp1_MNI_demo/CN/mwp1HC_4222_011_MR_MT1__GradWarp__N3m_Br_20110928092632691_S122659_I258684.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4015_037_MR_MT1__GradWarp__N3m_Br_20110421142443441_S103155_I228531.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4030_037_MR_MT1__GradWarp__N3m_Br_20110511153647859_S108046_I234654.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4034_023_MR_MT1__GradWarp__N3m_Br_20110623111414208_S110986_I241370.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4035_023_MR_MT1__GradWarp__N3m_Br_20110804075157696_S116628_I248665.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4053_141_MR_MT1__GradWarp__N3m_Br_20110602153641337_S109691_I238647.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4057_072_MR_MT1__GradWarp__N3m_Br_20110623111632388_S110940_I241377.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4058_014_MR_MT1__GradWarp__N3m_Br_20110602154806705_S110000_I238666.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4197_127_MR_MT1__GradWarp__N3m_Br_20110906120411801_S119904_I254772.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4240_127_MR_MT1__GradWarp__N3m_Br_20111008150013756_S123236_I260225.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4243_023_MR_MT1__GradWarp__N3m_Br_20111121111732075_S128504_I267751.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4263_014_MR_MT1__GradWarp__N3m_Br_20111015080041021_S124549_I261065.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4287_129_MR_MT1__GradWarp__N3m_Br_20111030172838553_S125945_I263726.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4294_130_MR_MT1__N3m_Br_20111206103318860_S130334_I270027.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4303_137_MR_MT1__N3m_Br_20111108160515659_S127891_I265269.nii.gz',
                'mwp1_MNI_demo/LMCI/mwp1MCI_4346_006_MR_MT1__N3m_Br_20111121112405023_S128998_I267757.nii.gz']
