import numpy as np
import nibabel as nib
img = nib.load('aal.nii')

x_range_from = 10; x_range_to = 110 #sagittal
y_range_from = 10; y_range_to = 130 #coronal
z_range_from = 5; z_range_to = 105 #axial
aal = img.get_fdata()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
aal = np.transpose(aal, (2, 0, 1)) # reorder dimensions to match coronal view z*x*y in MRIcron etc.
aal = np.flip(aal, (1,2)) # flip coronal and sagittal dimension

a = np.copy(aal)
print(a.shape)
from skimage.feature import canny


def neighbors(x,y,z):
	return [aal[x-1,y-1,z],
	aal[x-1,y,z-1],aal[x-1,y,z],aal[x-1,y,z+1],
	aal[x-1,y+1,z],
	aal[x,y-1,z-1],aal[x,y-1,z],aal[x,y-1,z+1],
	aal[x,y,z-1],aal[x,y,z],aal[x,y,z+1],
	aal[x,y+1,z-1],aal[x,y,z+1],
	aal[x+1,y-1,z],
	aal[x+1,y,z-1],aal[x+1,y,z],aal[x+1,y,z+1],
	aal[x+1,y+1,z]]


def at_border(p, nb):
	for k in nb:
		if k!=p:
			return True
	return False
	

for x in range(1,99):
	for y in range(1,99):
		for z in range(1,119):
			n = neighbors(x,y,z)
			if (aal[x,y,z] != 0) and (at_border(aal[x,y,z], n)): #voxel is at border of region
				a[x,y,z]=aal[x,y,z]
			else:
				a[x,y,z]=0


import os
image = nib.Nifti1Image(a, np.eye(4))
image.header.get_xyzt_units()
image.to_filename(os.path.join('canny_regions_by_border.nii.gz'))
