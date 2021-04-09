#!/usr/bin/env python
# coding: utf-8
import base64
import glob
import gzip
import pickle
import h5py
import logging
import os
import re
from io import BytesIO

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

from config import debug, selected_neuron, disable_gpu_for_tensorflow, stored_models, selected_model, \
    background_images_path, residuals_path, covariates_file, do_model_prefetch

if debug:
    print("stored_models = ")
    print(stored_models)
    print("selected_model = ")
    print(selected_model)

# Import covariate data from file
df = pd.read_pickle(covariates_file)

sid = df['RID']
grp = df['Group at scan date (1=CN, 2=EMCI, 3=LMCI, 4=AD, 5=SMC)']
age = df['Age at scan']
sex = df['Sex (1=female)']
tiv = df['TIV']
field = df['MRI_Field_Strength']
grpbin = (grp > 1)  # 1=CN, ...


# Load matched covariate information
with open('matched_cov_idx.pkl', 'rb') as cov_idx_file:
    # read the data as binary data stream
    cov_idx = pickle.load(cov_idx_file)

labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
grps = pd.DataFrame({'Group':grp, 'RID':sid}).iloc[cov_idx, :]

# Load residualized data from disk:
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
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto(
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
# device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# set_session(session)

if disable_gpu_for_tensorflow:
    if debug: print("Disabling GPU computation for Tensorflow...")
    os.environ[
        "CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU computation for tensorflow (https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu)

# Split data into training/validation and holdout test data
labels = to_categorical(np.asarray(labels))
# circumvent duplicate data:
idx = np.asarray(range(len(cov_idx)))
# train_idX,test_idX,train_Y,test_Y = train_test_split(idx, labels, test_size=0.1, stratify = labels, random_state=1)
test_idX = idx  # select all
testgrps = grps  # .iloc[test_idX, :]
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
    #            ("deconvnet",            {},                      "Deconvnet"),
    #            ("guided_backprop",      {},                      "Guided Backprop"),
    #            ("deep_taylor.bounded",  {"low": -1, "high": 1},  "DeepTaylor"),
    #            ("input_t_gradient",     {},                      "Input * Gradient"),
    #            ("lrp.z",                {},                      "LRP-Z"),
    #            ("lrp.epsilon",          {"epsilon": 1},          "LRP-epsilon"),
    ("lrp.alpha_1_beta_0", {"neuron_selection_mode": "index"}, "LRP-alpha1beta0"),
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
img = nib.load('aal/aal.nii')
img_drawn = nib.load('aal/canny_regions_by_border.nii.gz')
aal_drawn = img_drawn.get_fdata()

x_range_from = 10
x_range_to = 110  # sagittal
y_range_from = 10
y_range_to = 130  # coronal
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
aal_region_names = np.genfromtxt('aal/aal.csv', delimiter=';', usecols=(2), dtype=str, skip_header=1)


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


class Model:

    @staticmethod
    def scale_relevance_map(relevance_map, clipping_threshold):
        """
        Clips the relevance map to given threshold and adjusts it to range -1...1 float.

        :param numpy.ndarray relevance_map:
        :param int clipping_threshold: max value to be plotted, larger values will be set to this value
        :return : The relevance map, clipped to given threshold and adjusted to range -1...1 float.
        :rtype: numpy.ndarray
        """
        if debug: print("Called scale_relevance_map()")
        r_map = np.copy(relevance_map)  # leave original object unmodified.
        # perform intensity normalization
        scale = np.quantile(np.absolute(r_map), 0.99)
        if scale != 0:  # fallback if quantile returns zero: directly use abs max instead
            r_map = (r_map / scale)  # rescale range
        # corresponding to vmax in plt.imshow; vmin=-vmax used here
        # value derived empirically here from the histogram of relevance maps
        r_map[r_map > clipping_threshold] = clipping_threshold  # clipping of positive values
        r_map[r_map < -clipping_threshold] = -clipping_threshold  # clipping of negative values
        r_map = r_map / clipping_threshold  # final range: -1 to 1 float
        return r_map

    def set_model(self, new_model_name):
        """

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

        return

    def set_subj_img(self, img):
        """
        Sets the model input image and creates relevance map and prediction.
        :param numpy.ndarray img: the residualized model input image
        :return: None
        """
        self.subj_img = img
        self.subj_img = np.reshape(self.subj_img, (
            1,) + self.subj_img.shape)  # add first subj index again to mimic original array structure

        # evaluate/predict diag for selected subject
        self.pred = (self.mymodel.predict(self.subj_img)[0, 1] * 100)  # scale probability score to percent
        # derive relevance map from CNN model
        self.relevance_map = self.analyzer.analyze(self.subj_img, neuron_selection=selected_neuron)
        self.relevance_map = np.reshape(self.relevance_map, self.subj_img.shape[1:4])  # drop first index again
        self.relevance_map = scipy.ndimage.filters.gaussian_filter(self.relevance_map,
                                                                   sigma=0.8)  # smooth activity image
        self.relevance_map = Model.scale_relevance_map(self.relevance_map, 3)
        # print(np.max(relevance_map), np.min(relevance_map))

    def set_subj_bg(self, bg):
        """

        :param numpy.ndarray bg: the background image
        :return: None
        """
        self.subj_bg = bg

    def load_nifti(self, base64str, is_zipped):
        """
        Load base64 encoded string read from uploaded file upload into array.
        Also crops and flips the resulting image


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
        y_range_from = 10
        y_range_to = 130
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

    def residualize(self, img_arr, age=73, sex=0.498181818, tiv=1409, field=2.859090909):
        """
        Performs linear regression-based covariates cleaning of given scan.
        Takes ~1 min.

        Default covariate values are average values from ADNI sample.

        :param img_arr: numpy array to residualize
        :param int age: age of subject in years
        :param float sex: 1 = female, 0 = male
        :param float tiv: head volume in cm³
        :param field: the MRI field strength
        :return: the residualized array
        :rtype: numpy.ndarray
        """


        # Perform regression-based covariates cleaning
        res = np.copy(img_arr)  # residualized image

        self.entered_covariates_df = pd.DataFrame({'Age': [age], 'Sex': [sex], 'TIV': [tiv], 'FieldStrength': [field]})
        print(self.entered_covariates_df)
        covariates = self.entered_covariates_df.to_numpy(dtype=np.float32)  # convert data frame to nparray with 32bit types


        # load coefficients for linear models from hdf5
        # TODO: only load hdf5 from disk once at server start and avoid accessing hard drive every time.
        hf = h5py.File('linearmodels.hdf5', 'r')
        hf.keys  # read keys
        lmarray = np.array(hf.get('linearmodels'), dtype=np.float32)  # stores 4 coefficients + 1 intercept per voxel
        hf.close()

        # covCN = covariates[labels['Group'] == 0] # only controls as reference group to estimate effect of covariates
        # print("Controls covariates data frame size : ", covCN.shape)
        lmLoaded = linear_model.LinearRegression()

        for k in range(res.shape[2]):
            if (k % 10 == 0): print('Processing depth slice ', str(k + 1), ' of ', str(res.shape[2]))
            for j in range(res.shape[1]):
                for i in range(res.shape[0]):

                    if any(lmarray[k, j, i, :] != 0):
                        # load fitted linear model from file
                        lmLoaded.coef_ = lmarray[k, j, i, :4]
                        lmLoaded.intercept_ = lmarray[k, j, i, 4]

                        pred = lmLoaded.predict(covariates)  # calculate prediction for all subjects
                        res[i, j, k, 0] = res[
                                              i, j, k, 0] - pred  # % subtract effect of covariates from original values (=calculate residuals)
        print("Residualization successful.")
        self.uploaded_residual = res
        return res

    def __init__(self):
        if (debug): print("Initializing new datamodel object...")

        # Instance attributes (actual values set in set_model(...) and set_subject(...))
        self.analyzer = None
        self.relevance_map = None
        self.selected_model = None # filename of selected model
        self.mymodel = None # actual loaded model
        self.subj_img = None # model input image
        self.subj_bg = None # background image for plotting
        self.pred = None # current scan prediction for Alzheimer's
        self.uploaded_bg_img = None
        self.uploaded_residual = None
        self.entered_covariates_df = None # DataFrame of entered covariates

        # load selected model data from cache or disk:
        self.set_model(selected_model)

        # Call once to initialize first image and variables
        self.set_subject(index_lst[0])  # invoke with first subject

