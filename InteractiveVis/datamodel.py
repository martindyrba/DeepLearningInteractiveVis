#!/usr/bin/env python
# coding: utf-8
import base64
import glob
import gzip
import h5py
import logging
import os
import re
from io import BytesIO
import sys

import innvestigate
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from keras.models import load_model
# from sklearn.model_selection import train_test_split
from sklearn import linear_model
from keras.utils import to_categorical

import multiprocessing
import concurrent.futures # this module was introduced in Python 3.2; use the futures backport module for Python 2.7

from config import debug, selected_neuron, adaptive_relevance_scaling, disable_gpu_for_tensorflow, stored_models, selected_model, \
    background_images_path, residuals_path, covariates_excel_file, covariates_excel_sheet, \
    do_model_prefetch, linear_model_path, data_files

if debug:
    print("stored_models = ")
    print(stored_models)
    print("selected_model = ")
    print(selected_model)

# Import data from Excel sheet
df = pd.read_excel(covariates_excel_file, sheet_name=covariates_excel_sheet, engine='openpyxl')

sid = df['subject_ID']
grp = df['Group at scan date (1=CN, 2=EMCI, 3=LMCI, 4=AD, 5=SMC)']
age = df['Age at scan']
sex = df['Sex (1=female)']
tiv = df['TIV_CAT12']
field = df['MRI_Field_Strength']
grpbin = (grp > 1)  # 1=CN, ...

numfiles = len(data_files)
print('Found ', str(numfiles), ' nifti files')

# Match covariate information
cov_idx = [-1] * numfiles  # list; array: np.full((numfiles, 1), -1, dtype=int)
print('Matching covariates for loaded files ...')
for i, id in enumerate(sid):
    p = [j for j, x in enumerate(data_files) if
         re.search('_%04d_' % id, x)]  # translate ID numbers to four-digit numbers, get both index and filename
    if len(p) == 0:
        if debug: print('Did not find %04d' % id)  # did not find Excel sheet subject ID in loaded file selection
    else:
        if debug: print('Found %04d in %s: %s' % (id, p[0], data_files[p[0]]))
        cov_idx[p[0]] = i  # store Excel index i for data file index p[0]
print('Checking for scans not found in Excel sheet: ', sum(x < 0 for x in cov_idx))

labels = pd.DataFrame({'Group': grpbin}).iloc[cov_idx, :]
grps = pd.DataFrame({'Group': grp, 'RID': sid}).iloc[cov_idx, :]

# Load prepared data from disk:
hf = h5py.File(residuals_path, 'r')
hf.keys  # read keys
model_input_images = np.array(hf.get('images'))  # note: was of data frame type before
hf.close()
if debug:
    print("model_input_images.shape=")
    print(model_input_images.shape)

# specify version of tensorflow in Google Colab
# %tensorflow_version 1.x
logging.getLogger('tensorflow').disabled = True  # disable tensorflow deprecation warnings
if debug:
    print("Tensorflow version:")
    print(tf.__version__)

if disable_gpu_for_tensorflow:
    if debug: print("Disabling GPU computation for Tensorflow...")
    os.environ[
        "CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU computation for tensorflow (https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu)

# Split data into training/validation and holdout test data
labels = to_categorical(np.asarray(labels))
# circumvent duplicate data:
idx = np.asarray(range(numfiles))
test_idX = idx  # select all
testgrps = grps
print(testgrps)  # prints diagnosis and RID
print('Distribution of diagnoses in data: [1=CN, 3=LMCI, 4=AD]')
print(testgrps.Group.value_counts())

test_images = model_input_images[test_idX, :]
del model_input_images

testgrps["Group"] = testgrps["Group"].map({1: "CN", 3: "MCI", 4: "AD"})
Option_grps = np.array(testgrps)
Option_grps = Option_grps.astype('str')
Opt_grp = []

for i in range(len(Option_grps)):
    Opt_grp.append(' - ID '.join(Option_grps[i]))

# https://stackoverflow.com/questions/48279640/sort-a-python-list-while-maintaining-its-elements-indices
Opt_grp = [(x, i) for (i, x) in enumerate(Opt_grp)]
Opt_grp = sorted(Opt_grp)


def unzip(ls):
    if isinstance(ls, list):
        if not ls:
            return [], []
        else:
            Opt_grp, ys = zip(*ls)

        return list(Opt_grp), list(ys)
    else:
        raise TypeError


sorted_xs, index_lst = unzip(Opt_grp)

# Load original images (background) from disk
hf = h5py.File(background_images_path, 'r')
hf.keys # read keys
images_bg = np.array(hf.get('images'))
hf.close()
testdat_bg = images_bg[test_idX, :]
del images_bg

if debug:
    print('testdat_bg.shape=')
    print(testdat_bg.shape)

# see https://github.com/albermax/innvestigate/blob/master/examples/notebooks/imagenet_compare_methods.ipynb for a list of alternative methods
methods = [  # tuple with method,     params,                  label
    # ("deconvnet",            {},                      "Deconvnet"),
    # ("guided_backprop",      {},                      "Guided Backprop"),
    # ("deep_taylor.bounded",  {"low": -1, "high": 1},  "DeepTaylor"),
    # ("input_t_gradient",     {},                      "Input * Gradient"),
    # ("lrp.z",                {},                      "LRP-Z"),
    # ("lrp.epsilon",          {"epsilon": 1},          "LRP-epsilon"),
    # ("lrp.alpha_1_beta_0", {"neuron_selection_mode": "index"}, "LRP-alpha1beta0"),
    ("lrp.sequential_preset_a", {"neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0"), # LRP CMP rule taken from https://github.com/berleon/when-explanations-lie/blob/master/when_explanations_lie.py
]

model_cache = dict()


def load_model_from_disk_into_cache(model_path):
    global model_cache
    print("Loading " + model_path + " from disk...")
    model_cache[model_path] = dict()
    model_cache[model_path]["mymodel"] = load_model(model_path)
    model_cache[model_path]["mymodel"].layers[-1].activation = tf.keras.activations.linear
    model_cache[model_path]["mymodel"].save('tmp_wo_softmax.hdf5')
    model_wo_softmax = load_model('tmp_wo_softmax.hdf5')
    os.remove('tmp_wo_softmax.hdf5')
    print("model_wo_softmax loaded.")
    print('Creating analyzer...')
    # create analyzer -> only one selected here!
    for method in methods:
        model_cache[model_path]["analyzer"] = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
    print('Analyzer created.')
    return model_cache[model_path]["mymodel"], model_cache[model_path]["analyzer"]


# Preload CNN models from disk:
if do_model_prefetch:
    print("Preloading all " + str(len(stored_models)) + " models. This may take a while...")
    for model_path in stored_models:
        load_model_from_disk_into_cache(model_path)

# load atlas nifti data:
img = nib.load('aal/aal.nii.gz')
img_drawn = nib.load('aal/canny_regions_by_border.nii.gz')
aal_drawn = img_drawn.get_fdata()

x_range_from = 10
x_range_to = 110  # sagittal
y_range_from = 13
y_range_to = 133  # coronal
z_range_from = 5
z_range_to = 105  # axial
aal = img.get_fdata()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
aal = np.transpose(aal, (2, 0, 1))  # reorder dimensions to match coronal view z*x*y in MRIcron etc.
aal = np.flip(aal, (1, 2))  # flip coronal and sagittal dimension
# aal is now orientated like this: [axial,sagittal,coronal]


if debug:
    print("aal.shape = ")
    print(aal.shape)

# load region names into array:
# region name with id 'i' is stored at index 'i'
aal_region_names = np.genfromtxt('aal/aal.csv', delimiter=',', usecols=(2), dtype=str, skip_header=1)


def get_region_id(axi, sag, cor):
    """

    :param int axi: the axial coordinate
    :param int sag: the sagittal coordinate
    :param int cor: the coronal coordinate
    :return: The region ID, given the coordinates above.
    :rtype: int
    """
    return aal[axi, sag, cor]


def get_region_name(axi, sag, cor):
    """

    :param axi:
    :param sag:
    :param cor:
    :return: The region name corresponding to the region ID in the CSV.
    :rtype: str
    """
    return aal_region_names[int(aal[axi, sag, cor])]


def scale_relevance_map(relevance_map, clipping_threshold):
    """
    Clips the relevance map to given threshold and adjusts it to range -1...1 float.

    :param numpy.ndarray relevance_map:
    :param int clipping_threshold: max value to be plotted, larger values will be set to this value
    :return: The relevance map, clipped to given threshold and adjusted to range -1...1 float.
    :rtype: numpy.ndarray
    """
    if debug: print("Called scale_relevance_map()")
    r_map = np.copy(relevance_map)  # leave original object unmodified.
    # perform intensity normalization
    if adaptive_relevance_scaling:
        scale = np.quantile(np.absolute(r_map), 0.9999)
    else:
        scale = 1/500 # multiply by 500
    if scale != 0:  # fallback if quantile returns zero: directly use abs max instead
        r_map = (r_map / scale)  # rescale range
    # corresponding to vmax in plt.imshow; vmin=-vmax used here
    # value derived empirically here from the histogram of relevance maps
    r_map[r_map > clipping_threshold] = clipping_threshold  # clipping of positive values
    r_map[r_map < -clipping_threshold] = -clipping_threshold  # clipping of negative values
    r_map = r_map / clipping_threshold  # final range: -1 to 1 float
    return r_map


def do_prepare(covs, lmcoeffs, data, first_slice, last_slice):
	"""
	Actually performs the processing of uploaded user content.
	Will be executed in parallel using multiprocessing to speed up calculations.
	
	:param numpy.ndarray covs: the covariates array
	:param numpy.ndarray lmcoeffs: the coefficients of the linear models (one vector for each voxel)
	:param numpy.ndarray data: the input image to prepare
	:param int first_slice: the first slice to processing
	:param int last_slice: the last slice to process (exclusive)
	:return: an array containing the residuals of the selected slices
	:rtype: numpy.ndarray 
	"""
	lm = linear_model.LinearRegression()
	out_size = list(data.shape)
	out_size[2] = last_slice-first_slice
	out = np.zeros(tuple(out_size), dtype=np.float32)
	#print(out.shape)
	if debug: print('Processing depth slices ', str(first_slice), ' to ', str(last_slice-1), ' of ', str(data.shape[2]))
	for k in range(first_slice, last_slice):
		for j in range(data.shape[1]):
			for i in range(data.shape[0]):
				if any(lmcoeffs[k, j, i, :] != 0): # skip empty voxels/space
					# load fitted linear model coefficients
					lm.coef_ = lmcoeffs[k, j, i, :4]
					lm.intercept_ = lmcoeffs[k, j, i, 4]
					pred = lm.predict(covs)  # calculate prediction for all subjects
					out[i, j, k-first_slice, 0] = data[i, j, k, 0] - pred  # % subtract effect of covariates from original values (=calculate residuals)
	return out



class Model:

    def set_model(self, new_model_name):
        """
        Callback for a new model being selected by path/file name.

        :param str new_model_name: the path of the new model file to be selected.
        :return: None
        """
        global model_cache
        if debug: print("Called set_model().")

        self.selected_model = new_model_name
        try:
            self.mymodel = model_cache[self.selected_model]["mymodel"]
            self.analyzer = model_cache[self.selected_model]["analyzer"]
            if debug: print("Model loaded from cache.")
        except KeyError:
            (self.mymodel, self.analyzer) = load_model_from_disk_into_cache(self.selected_model)
            if debug: print("Model loaded from disk.")

    def set_subject(self, subj_id):
        """
        Callback for a new subject being selected by id.

        :param int subj_id: the subject to select.
        :return: returns values by modifying instance variables: self.subj_bg, self.subj_img, self.pred, self.relevance_map
        """
        if debug: print("Called set_subject().")

        # global subj_bg, subj_img, pred, relevance_map # define global variables to store subject data
        self.set_subj_img(test_images[subj_id])
        self.set_subj_bg(testdat_bg[subj_id, :, :, :, 0])

    def set_subj_img(self, img):
        """
        Sets the model input image and creates relevance map and prediction.
        
        :param numpy.ndarray img: the prepared model input image
        :return: None
        """
        if debug: print("Called set_subj_img().")
        self.subj_img = img
        self.subj_img = np.reshape(self.subj_img, (
            1,) + self.subj_img.shape)  # add first subj index again to mimic original array structure

        # evaluate/predict diag for selected subject
        self.pred = (self.mymodel.predict(self.subj_img)[0, 1] * 100)  # scale probability score to percent
        # derive relevance map from CNN model
        self.relevance_map = self.analyzer.analyze(self.subj_img, neuron_selection=selected_neuron)
        self.relevance_map = np.reshape(self.relevance_map, self.subj_img.shape[1:4])  # drop first index again
        self.relevance_map = scipy.ndimage.filters.gaussian_filter(self.relevance_map, sigma=0.8)  # smooth activity image
        self.relevance_map = scale_relevance_map(self.relevance_map, 1)

    def set_subj_bg(self, bg):
        """
        Sets the visualization background image.
        
        :param numpy.ndarray bg: the background image
        :return: None
        """
        if debug: print("Called set_subj_bg().")
        self.subj_bg = bg

    def load_nifti(self, base64str, is_zipped):
        """
        Load base64 encoded string read from uploaded file upload into array.
        Also crops and flips the resulting image.

        :param str base64str: base64 encoded string (i.e. the uploaded file)
        :param boolean is_zipped: if the file is zipped (i.e. if the uploaded file name ends with '.gz')
        :return: the nifti image as numpy array:
            <li> 1. dimension: image row
            <li> 2. dimension: image column
            <li> 3. dimension: image depth
            <li> 4. dimension: color channels (should be 1, because of monochrome gray scan)
        :rtype: numpy.ndarray
        """
        # x: sag
        # y: cor
        # z: axi
        x_range_from = 10
        x_range_to = 110
        y_range_from = 13
        y_range_to = 133
        z_range_from = 5
        z_range_to = 105

        if debug: print('Called load_nifti().')

        img_arr = np.zeros(
            (z_range_to - z_range_from, x_range_to - x_range_from, y_range_to - y_range_from, 1),
            dtype=np.float32)  # z×x×y×1; avoid 64bit types
        
        base64_bytes = base64str.encode("ascii")
        raw_bytes = base64.b64decode(base64_bytes)
        if is_zipped:
            if debug: print("Unzipping...")
            nifti_bytes = gzip.decompress(raw_bytes)
        else:
            if debug: print("Not unzipping...")
            nifti_bytes = raw_bytes
        inputstream = BytesIO(nifti_bytes)
        file_holder = nib.FileHolder(fileobj=inputstream)
        
        img = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
        if img.shape != (121,145,121):
            raise ValueError("Invalid image shape for uploaded image! Got: " + str(img.shape) + ", expected: (121,145,121)")
        
        if debug:
            print("original img.shape:")
            print(img.shape)
        img = img.get_fdata()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
        inputstream.close()
        img = np.transpose(img, (2, 0, 1))  # reorder dimensions to match coronal view z*x*y in MRIcron etc.
        img = np.flip(img)  # flip all positions
        if debug:
            print("cropped img.shape: ")
            print(img.shape)
        img_arr[:, :, :, 0] = np.nan_to_num(img)  # orientation: axial,sagittal,coronal,colorchannel

        print("Successfully loaded uploaded nifti.")
        self.uploaded_bg_img = img_arr
        return img_arr


    def reset_prepared_data(self, img_arr):
        """
        Resets the "prepared" data (residuals) to a zero-filled array.
        This method is only being used for visualization purpose in order to already show an empty overlay while the data is still being processed/prepared.
        :param img_arr: numpy array to prepare
        :return: a copy of the img_array filled with zeros
        :rtype: numpy.ndarray
        """
        if debug: print('Called reset_prepared_data().')
        res = np.zeros(img_arr.shape, img_arr.dtype)
        self.uploaded_residual = res
        return res


    def set_covariates(self, age=73, sex=0.5, tiv=1400, field=2.86):
        """
        Sets the covariates information to be displayed in the visualization.
        Default covariate values are average values obtained from the ADNI2 sample.

        :param img_arr: numpy array to prepare
        :param int age: age of subject in years
        :param float sex: 1 = female, 0 = male
        :param float tiv: head volume in cm³ = ml
        :param field: the MRI field strength
        """
        self.entered_covariates_df = pd.DataFrame({'Age': [age], 'Sex': [sex], 'TIV': [tiv], 'FieldStrength': [field]})
        print(self.entered_covariates_df)


    def prepare_data(self, img_arr):
        """
        Performs linear regression-based covariates cleaning of given scan.
        May take about 1 min, but typically should be much faster on multi-core CPU systems.

        Default covariate values are average values obtained from the ADNI2 sample.

        :param img_arr: numpy array to prepare
        :return: the prepared array
        :rtype: numpy.ndarray
        """
        if debug: print('Called prepare_data().')
        # uncomment to directly use the provided data as input:
        # res = np.copy(img_arr)
        # return res
        
        print(self.entered_covariates_df) # assume covariates were already set using set_covariates() before
        covariates = self.entered_covariates_df.to_numpy(dtype=np.float32)  # convert data frame to nparray with 32bit types

        if linear_model_path is None: # shortcut to avoid processing if no model is given
            print("Skipping processing, returning original data.")
        else:
            if self.lmarray is None:
                # load coefficients for linear models from hdf5
                hf = h5py.File(linear_model_path, 'r')
                hf.keys  # read keys
                self.lmarray = np.array(hf.get('linearmodels'), dtype=np.float32)  # stores 4 coefficients + 1 intercept per voxel
                hf.close()

            # covCN = covariates[labels['Group'] == 0] # only controls as reference group to estimate effect of covariates
            # print("Controls covariates data frame size : ", covCN.shape)
            
            # TODO: move executor initialization and usage outside of the model class
            if self.executor is None:
                if sys.platform == 'win32': 
                    # disable multicore processing in Windows as the ProcessPoolExecutor does not seem to work properly
                    # (trows ModuleNotFoundError: No module named 'datamodel')
                    self.executor = concurrent.futures.ThreadPoolExecutor(1) # using single-threaded ThreadPoolExecutor isntead
                    self.num_threads = 1
                    self.step_size = img_arr.shape[2]
                else: # Linux, MacOS
                    self.executor = concurrent.futures.ProcessPoolExecutor(self.num_threads) # change to ThreadPoolExecutor for better debugging
                    self.step_size = np.ceil(img_arr.shape[2] / self.num_threads).astype(int)
            
            print("Submitting n=", str(self.num_threads), " parallel workers for processing")
            futures = []
            for i in range(self.num_threads):
                args = (covariates,
                        self.lmarray,
                        img_arr,
                        i*self.step_size, 
                        min((i+1)*self.step_size,img_arr.shape[2]))
                futures.append(self.executor.submit(do_prepare, *args))
            concurrent.futures.wait(futures)
            if debug:
                print(futures)
            results = []
            for f in futures:
                results.append(f.result())
            res = np.concatenate(results, axis=2)
            print("processing successful.")
        
        self.uploaded_residual = res
        return res


    def __init__(self):
        if (debug): print("Initializing new datamodel object...")

        # Instance attributes (values need to be set using set_model(...) and set_subject(...))
        self.analyzer = None # analyzer object for the currently loaded model
        self.relevance_map = None # filtered relevance map for the current input image
        self.selected_model = None # filename of selected model
        self.mymodel = None # currently selected/active model
        self.subj_img = None # input image for the CNN model
        self.subj_bg = None # background image for plotting
        self.pred = None # prediction for current image
        self.uploaded_bg_img = None # stores images uploaded by user
        self.uploaded_residual = None # stores residuals for user upload
        self.entered_covariates_df = None # DataFrame of entered covariates
        self.lmarray = None # linear model coefficients used for covariate cleaning of user upload
        self.executor = None # multithreading executor for parallel processing (used for processing)
        self.num_threads = multiprocessing.cpu_count()
        self.step_size = None # array index step size for multithreading

        # load selected model data from cache or disk:
        self.set_model(selected_model)

        # Call once to initialize first image and variables
        self.set_subject(index_lst[0])  # invoke with first subject


    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(False)
