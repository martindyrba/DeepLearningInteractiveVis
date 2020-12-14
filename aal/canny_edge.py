import numpy as np
import nibabel as nib
img = nib.load('aal.nii')

sig=5 #sigma

x_range_from = 10; x_range_to = 110 #sagittal
y_range_from = 10; y_range_to = 130 #coronal
z_range_from = 5; z_range_to = 105 #axial
aal = img.get_fdata()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
aal = np.transpose(aal, (2, 0, 1)) # reorder dimensions to match coronal view z*x*y in MRIcron etc.
aal = np.flip(aal, (1,2)) # flip coronal and sagittal dimension


a = np.copy(aal)
print(a.shape)
from skimage.feature import canny


i = 1
for x in range(0,100):
	a[x,:,:] = canny(aal[x,:,:], sigma=sig)
		
	print("slice " + str(i) + " done")
	i=i+1


import os
image = nib.Nifti1Image(a, np.eye(4))
image.header.get_xyzt_units()
image.to_filename(os.path.join('canny_regions_sigma_' + str(sig) + '.nii.gz'))
