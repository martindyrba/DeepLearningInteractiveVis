#!/usr/bin/env python
# coding: utf-8

debug = False
selected_neuron = 1  # switch to neuron 0 if required
scale_factor = 2.5  # scale factor for brain views
disable_gpu_for_tensorflow = True

background_images_path = 'orig_images_demo.hdf5'  # original images
residuals_path = 'residuals_demo.hdf5'  # stored residuals, replace by original images if desired
linear_model_path = 'linearmodels.hdf5' # stored linear model coefficients -> set to None to skip residualization

# define list of available models and the model selected by default
import glob
import os
model_folder = 'newmodel' # folder containing our trained models
stored_models = sorted(glob.glob(os.path.join(model_folder, 'newmodel_wb_cv16.hdf5'))) #sorted(glob.glob(os.path.join(model_folder, 'newmodel_wb_cv[1-9].hdf5'))) + sorted(glob.glob(os.path.join('newmodel', 'newmodel_wb_cv[1-2][0-9].hdf5'))) # get list of available models, global sorting not possible because of different number of digits
selected_model = os.path.join(model_folder, 'newmodel_wb_cv16.hdf5') # model file name to load (default cv8 like before)
do_model_prefetch = True # load ALL models already at application startup --> this will take LONG and is only advised for long-running web/app servers
flip_left_right_in_frontal_plot = False

# define path/name to pickle file containing covariates
covariates_file = 'covariates.pkl'
