#view.py
#!/usr/bin/env python
# coding: utf-8
firstrun = True #Set to false from main.py after first run.

from bokeh.models.widgets import Select
from bokeh.layouts import column, row, Spacer
from bokeh.plotting import figure, curdoc
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models.annotations import Span
from bokeh.models.widgets import Slider
from bokeh.models.glyphs import Rect
from bokeh.models import Div, Toggle
import numpy as np

from config import scale_factor
import datamodel as m
# Constants for first initialization/getter methods; no write access needed.
from datamodel import sorted_xs, stored_models, selected_model, get_region_name, get_region_ID, aal_drawn 


# In[12]:


subject_select = Select(title="Subjects:", value=sorted_xs[0], options=sorted_xs, width=200)
model_select = Select(title="Model:", value=selected_model, options=stored_models, width=200)
slice_slider_frontal = Slider(start=1, end=m.subj_bg.shape[2], value=50, step=1,
                  title="Coronal slice", width=200)
slice_slider_axial = Slider(start=1, end=m.subj_bg.shape[0], value=50, step=1,
                  title="Axial slice", width=200)
slice_slider_sagittal = Slider(start=1, end=m.subj_bg.shape[1], value=50, step=1,
                  title="Sagittal slice", width=200)
threshold_slider = Slider(start=0, end=1, value=0.4, step=0.05,
                  title="Relevance threshold", width=200)
clustersize_slider = Slider(start=0, end=250, value=50, step=10,
                  title="Minimum cluster size", width=200)
transparency_slider = Slider(start=0, end=1, value=0.3, step=0.05,title="Overlay transparency", width=200)

# initialize the figures
guide_frontal = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
guide_frontal.title.text_font = 'arial'
guide_frontal.title.text_font_style = 'normal'
#guide_frontal.title.text_font_size = '10pt'
guide_frontal.axis.visible = False
guide_frontal.x_range.range_padding = 0
guide_frontal.y_range.range_padding = 0

guide_axial = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
guide_axial.title.text_font = 'arial'
guide_axial.title.text_font_style = 'normal'
#guide_axial.title.text_font_size = '10pt'
guide_axial.axis.visible = False
guide_axial.x_range.range_padding = 0
guide_axial.y_range.range_padding = 0

guide_sagittal = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
guide_sagittal.title.text_font = 'arial'
guide_sagittal.title.text_font_style = 'normal'
#guide_sagittal.title.text_font_size = '10pt'
guide_sagittal.axis.visible = False
guide_sagittal.x_range.range_padding = 0
guide_sagittal.y_range.range_padding = 0

clusthist = figure(plot_width=208, plot_height=70, title='Distribution of cluster sizes:', toolbar_location=None,
              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
clusthist.title.text_font = 'arial'
clusthist.title.text_font_style = 'normal'
clusthist.axis.visible = False
clusthist.x_range.range_padding = 0
clusthist.y_range.range_padding = 0


p_frontal = figure(plot_width=int(np.floor(m.subj_bg.shape[1]*scale_factor)), plot_height=int(np.floor(m.subj_bg.shape[0]*scale_factor)), title='',
          toolbar_location=None,
          active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
p_frontal.axis.visible = False
p_frontal.x_range.range_padding = 0
p_frontal.y_range.range_padding = 0

# The vertical crosshair line on the frontal view that indicates the selected sagittal slice.
frontal_crosshair_from_sagittal = Span(location = slice_slider_sagittal.value-1, dimension='height', line_color='green', line_dash='dashed', line_width=2)

# The horizontal crosshair line on the frontal view that indicates the selected axial slice.
frontal_crosshair_from_axial = Span(location=slice_slider_axial.value-1, dimension='width', line_color='green', line_dash='dashed', line_width=2)
p_frontal.add_layout(frontal_crosshair_from_sagittal)
p_frontal.add_layout(frontal_crosshair_from_axial)



p_axial = figure(plot_width=int(np.floor(m.subj_bg.shape[1]*scale_factor)), plot_height=int(np.floor(m.subj_bg.shape[2]*scale_factor)), title='',
          toolbar_location=None,
          active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
p_axial.axis.visible = False
p_axial.x_range.range_padding = 0
p_axial.y_range.range_padding = 0

axial_crosshair_from_sagittal = Span(location=slice_slider_sagittal.value-1, dimension='height', line_color='green', line_dash='dashed', line_width=2)
axial_crosshair_from_frontal = Span(location=slice_slider_frontal.end-slice_slider_frontal.value + 1, dimension='width', line_color='green', line_dash='dashed', line_width=2)
p_axial.add_layout(axial_crosshair_from_sagittal)
p_axial.add_layout(axial_crosshair_from_frontal)



p_sagittal = figure(plot_width=int(np.floor(m.subj_bg.shape[2]*scale_factor)), plot_height=int(np.floor(m.subj_bg.shape[0]*scale_factor)), title='',
          toolbar_location=None,
          active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
p_sagittal.axis.visible = False
p_sagittal.x_range.range_padding = 0
p_sagittal.y_range.range_padding = 0

sagittal_crosshair_from_frontal = Span(location=slice_slider_frontal.value-1, dimension='height', line_color='green', line_dash='dashed', line_width=2)
sagittal_crosshair_from_axial = Span(location=slice_slider_axial.end-slice_slider_axial.value-1, dimension='width', line_color='green', line_dash='dashed', line_width=2)
p_sagittal.add_layout(sagittal_crosshair_from_frontal)
p_sagittal.add_layout(sagittal_crosshair_from_axial)

toggle_regions = Toggle(label='Show outline of atlas region', button_type='default', width=200)



region_ID = get_region_ID(slice_slider_axial.value-1, slice_slider_sagittal.value-1, slice_slider_frontal.value-1)
selected_region = get_region_name(slice_slider_axial.value-1, slice_slider_sagittal.value-1, slice_slider_frontal.value-1)
region_div = Div(text="Region: "+selected_region, sizing_mode="stretch_both", css_classes=["region_divs"])


def update_region_div():
    global selected_region, region_ID
    region_ID = get_region_ID(slice_slider_axial.value-1, slice_slider_sagittal.value-1, slice_slider_frontal.value-1)
    selected_region = get_region_name(slice_slider_axial.value-1, slice_slider_sagittal.value-1, slice_slider_frontal.value-1)
    region_div.update(text="Region: "+ selected_region)

def update_cluster_divs():
    cluster_index = clust_labelimg[m.subj_bg.shape[0]-slice_slider_axial.value, slice_slider_sagittal.value-1, slice_slider_frontal.value-1]
    if cluster_index == 0:
        cluster_size_div.update(text="Cluster Size: " + "N/A")
        cluster_mean_div.update(text="Mean Intensity: " + "N/A")
        cluster_peak_div.update(text="Peak Intensity: " + "N/A")
    else:
        cluster_size_div.update(text="Cluster Size: " + str(clust_sizes_drawn[cluster_index]) + " vx = " + str(format(clust_sizes_drawn[cluster_index]*3.375, '.0f')) + " mm³")
        cluster_mean_div.update(text="Mean Intensity: " + str(format(clust_mean_intensities[cluster_index], '.2f')))
        cluster_peak_div.update(text="Peak Intensity: " + str(format(clust_peak_intensities[cluster_index], '.2f')))

cluster_size_div = Div(text="Cluster Size: " + "0", css_classes=["cluster_divs"]) #initialize with 0 because clust_labelimg does not exist yet
cluster_mean_div = Div(text="Mean Intensity: " + "0", css_classes=["cluster_divs"])
cluster_peak_div = Div(text="Peak Intensity: " + "0", css_classes=["cluster_divs"])

from config import debug
from datamodel import age, cov_idx, sex, tiv

def update_subject_divs(subj_id):
    if debug: print("Called update_subject_divs().")

    age_div.update(text="Age: %.0f " % age.iloc[cov_idx[subj_id]])
    if (sex.iloc[cov_idx[subj_id]] == 1):
        sex_div.update(text="Sex: " + "female")
    elif (sex.iloc[cov_idx[subj_id]] == 0):
        sex_div.update(text="Sex: " + "male")
    else:
        sex_div.update(text="Sex: " + "N/A")
    #tiv_div.update(text="TIV: " + str(tiv.iloc[cov_idx[subj_id]]))
    tiv_div.update(text="TIV: %.0f cm³" % tiv.iloc[cov_idx[subj_id]])

# see InteractiveVis/static/
age_div = Div(text="Age: " + "N/A", width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), css_classes=["subject_divs"]) #no subject selected at time of initialization
sex_div = Div(text="Sex: " + "N/A", width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), css_classes=["subject_divs"])
tiv_div = Div(text="TIV: " + "N/A", width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), css_classes=["subject_divs"])


# initialize column layout
layout = row(column(subject_select, model_select,
              Spacer(height=40, width=200, sizing_mode='scale_width'),
              threshold_slider, clusthist, clustersize_slider, transparency_slider,
              toggle_regions, region_div,
                column(cluster_size_div, cluster_mean_div, cluster_peak_div)),
                column(
                row(age_div, sex_div, tiv_div, css_classes=["subject_divs"]),
                row(
              column(p_frontal, slice_slider_frontal, guide_frontal),
              column(p_axial, slice_slider_axial, guide_axial),
              column(p_sagittal, slice_slider_sagittal, guide_sagittal))
              )
)



# In[13]:


from PIL import Image
from matplotlib import cm
from skimage.measure import label, regionprops

def apply_thresholds(map, threshold = 0.5, cluster_size = 20):
    if debug: print("Called apply_thresholds().")
    global overlay, sum_pos_frontal, sum_neg_frontal, sum_pos_axial, sum_neg_axial, sum_pos_sagittal, sum_neg_sagittal, clust_sizes, clust_labelimg, clust_sizes_drawn, clust_mean_intensities, clust_peak_intensities # define global variables to store subject data
    overlay = np.copy(map)
    overlay[np.abs(overlay) < threshold] = 0 # completely hide low values
    # cluster_size filtering
    labelimg = np.copy(overlay)
    labelimg[labelimg>0] = 1 # binarize img
    labelimg = label(labelimg, connectivity=2)
    lprops = regionprops(labelimg, intensity_image=overlay)
    clust_sizes = []
    clust_sizes_drawn = np.zeros(len(lprops)+1, dtype=np.uint32) #just those that show up in the end on canvas
    clust_mean_intensities = np.zeros(len(lprops)+1, dtype=np.float64)
    clust_peak_intensities = np.zeros(len(lprops)+1, dtype=np.float64)
    for lab in lprops:
        clust_sizes.append(lab.area)
        clust_sizes_drawn[lab.label]=lab.area
        clust_mean_intensities[lab.label]=lab.mean_intensity
        clust_peak_intensities[lab.label]=lab.max_intensity
        if lab.area<cluster_size:
            labelimg[labelimg==lab.label] = 0 # remove small clusters
            clust_sizes_drawn[lab.label] = 0
            clust_mean_intensities[lab.label] = 0
            clust_peak_intensities[lab.label] = 0
    clust_labelimg = np.copy(labelimg)
    labelimg[labelimg>0] = 1 # create binary mask
    np.multiply(overlay, labelimg, out=overlay)
    tmp = np.copy(overlay)
    tmp[tmp<0] = 0
    sum_pos_frontal = np.sum(tmp, axis=(0,1)) # sum of pos relevance in slices
    sum_pos_axial = np.sum(tmp, axis=(2,1))
    sum_pos_sagittal = np.sum(tmp, axis=(2,0))
    tmp = np.copy(overlay)
    tmp[tmp>0] = 0
    sum_neg_frontal = np.sum(tmp, axis=(0,1)) # sum of neg relevance in slices
    sum_neg_axial = np.sum(tmp, axis=(2,1))
    sum_neg_sagittal = np.sum(tmp, axis=(2,0))
    return overlay # result also stored in global variables: overlay, sum_pos*, sum_neg*




def bg2RGBA(bg):
    img = bg/4 + 0.25 # rescale to range of approx. 0..1 float
    img = np.uint8(cm.gray(img) * 255)
    return img

def overlay2RGBA(map, alpha = 0.5):
    # assume map to be in range of -1..1 with 0 for hidden content
    alpha_mask = np.copy(map)
    alpha_mask[np.abs(alpha_mask) > 0] = alpha # final transparency of visible content
    map = map/2 + 0.5 # range 0-1 float
    ovl = np.uint8(cm.jet(map) * 255) # cm translates range 0 - 255 uint to rgba array
    ovl[:,:,3] = np.uint8(alpha_mask * 255) # replace alpha channel (fourth dim) with calculated values
    return ovl


def region2RGBA(rg, alpha = 0.5):
    global region_ID
    rgcopy = np.copy(rg)
    if region_ID != 0:
        rgcopy[rgcopy != region_ID] = 0
        rgcopy[rgcopy == region_ID] = 1
    else: #crosshair on background
        rgcopy[True] = 0
    alpha_mask = np.copy(rgcopy)
    alpha_mask[alpha_mask == 1] = alpha
    img = np.uint8(cm.autumn(rgcopy)* 255)
    img[:,:,3] = np.uint8(alpha_mask * 255 )
    return img



# In[18]:


# define other callback functions for the sliders
clust_hist_bins = list(range(0, 250+1, 10)) # list from (0, 10, .., 250); range max is slider_max_size+1

def update_guide_frontal():
    global firstrun, pos_area_frontal, pos_line_frontal, neg_area_frontal, neg_line_frontal, hist_frontal
    x = np.arange(0, sum_neg_frontal.shape[0])
    y0 = np.zeros(x.shape, dtype=int)
    if firstrun:
        # initialize/creat plots
        guide_frontal.line(x, y0, color="#000000")
        pos_area_frontal = guide_frontal.varea(x=x, y1=sum_pos_frontal, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area_frontal")
        pos_line_frontal = guide_frontal.line(x, y=sum_pos_frontal, line_width=2, color="#d22a40", name="pos_line_frontal")
        neg_area_frontal = guide_frontal.varea(x=x, y1=sum_neg_frontal, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area_frontal")
        neg_line_frontal = guide_frontal.line(x, y=sum_neg_frontal, line_width=2, color="#36689b", name="neg_line_frontal")
        # calc histogram; clip high values to slider max (=200)
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist_frontal = clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist_frontal")
        #firstrun_frontal = False
    else: # update plots
        curdoc().hold() # disable page updates
        pos_area_frontal.data_source.data = {'x':x, 'y1':sum_pos_frontal, 'y2':y0}
        pos_line_frontal.data_source.data = {'x':x, 'y':sum_neg_frontal}
        neg_area_frontal.data_source.data = {'x':x, 'y1':sum_neg_frontal, 'y2':y0}
        neg_line_frontal.data_source.data = {'x':x, 'y':sum_neg_frontal}
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist_frontal.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}
        curdoc().unhold() # enable page updates again

def update_guide_axial():
    global firstrun, pos_area_axial, pos_line_axial, neg_area_axial, neg_line_axial, hist_axial
    x_mirrored = np.arange(0, sum_neg_axial.shape[0]) #needs to be flipped
    x = x_mirrored[::-1]
    y0 = np.zeros(x.shape, dtype=int)
    if firstrun:
        # initialize/creat plots
        guide_axial.line(x, y0, color="#000000")
        pos_area_axial = guide_axial.varea(x=x, y1=sum_pos_axial, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area_axial")
        pos_line_axial = guide_axial.line(x, y=sum_pos_axial, line_width=2, color="#d22a40", name="pos_line_axial")
        neg_area_axial = guide_axial.varea(x=x, y1=sum_neg_axial, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area_axial")
        neg_line_axial = guide_axial.line(x, y=sum_neg_axial, line_width=2, color="#36689b", name="neg_line_axial")
        # calc histogram; clip high values to slider max (=200)
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist_axial = clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist_axial")
        #firstrun_axial = False
    else: # update plots
        curdoc().hold() # disable page updates
        pos_area_axial.data_source.data = {'x':x, 'y1':sum_pos_axial, 'y2':y0}
        pos_line_axial.data_source.data = {'x':x, 'y':sum_neg_axial}
        neg_area_axial.data_source.data = {'x':x, 'y1':sum_neg_axial, 'y2':y0}
        neg_line_axial.data_source.data = {'x':x, 'y':sum_neg_axial}
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist_axial.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}
        curdoc().unhold() # enable page updates again

def update_guide_sagittal():
    global firstrun, pos_area_sagittal, pos_line_sagittal, neg_area_sagittal, neg_line_sagittal, hist_sagittal
    x = np.arange(0, sum_neg_sagittal.shape[0])
    y0 = np.zeros(x.shape, dtype=int)
    if firstrun:
        # initialize/creat plots
        guide_sagittal.line(x, y0, color="#000000")
        pos_area_sagittal = guide_sagittal.varea(x=x, y1=sum_pos_sagittal, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area_sagittal")
        pos_line_sagittal = guide_sagittal.line(x, y=sum_pos_sagittal, line_width=2, color="#d22a40", name="pos_line_sagittal")
        neg_area_sagittal = guide_sagittal.varea(x=x, y1=sum_neg_sagittal, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area_sagittal")
        neg_line_sagittal = guide_sagittal.line(x, y=sum_neg_sagittal, line_width=2, color="#36689b", name="neg_line_sagittal")
        # calc histogram; clip high values to slider max (=200)
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist_sagittal = clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist_sagittal")
        #firstrun_sagittal = False
    else: # update plots
        curdoc().hold() # disable page updates
        pos_area_sagittal.data_source.data = {'x':x, 'y1':sum_pos_sagittal, 'y2':y0}
        pos_line_sagittal.data_source.data = {'x':x, 'y':sum_neg_sagittal}
        neg_area_sagittal.data_source.data = {'x':x, 'y1':sum_neg_sagittal, 'y2':y0}
        neg_line_sagittal.data_source.data = {'x':x, 'y':sum_neg_sagittal}
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist_sagittal.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}
        curdoc().unhold() # enable page updates again

def plot_frontal():
    if debug: print("Called plot_frontal().")

    bg = m.subj_bg[:,:,slice_slider_frontal.value-1]
    bg = bg2RGBA(bg)
    bg = np.flipud(bg)

    ovl = overlay[:,:,slice_slider_frontal.value-1]
    ovl = overlay2RGBA(ovl, alpha=1-transparency_slider.value)
    ovl = np.flipud(ovl)

    p_frontal.image_rgba(image=[bg,ovl], x=[0,0], y=[0,0], dw=bg.shape[1], dh=bg.shape[0])
    if (toggle_regions.active):
        plot_frontal_region()


def plot_frontal_region():
    if debug: print("Called plot_frontal_region().")
    rg = aal_drawn[:,:,slice_slider_frontal.value-1] #rg = region
    rg = region2RGBA(rg)

    p_frontal.image_rgba(image=[rg], x=[0], y=[0], dw=rg.shape[1], dh=rg.shape[0], level='glyph') #for bokeh render levels see Readme

def plot_axial():
    if debug: print("Called plot_axial().")

    bg = m.subj_bg[m.subj_bg.shape[0]-slice_slider_axial.value,:,:]
    bg = bg2RGBA(bg)
    bg = np.rot90(bg)

    ovl = overlay[m.subj_bg.shape[0]-slice_slider_axial.value,:,:]
    ovl = overlay2RGBA(ovl, alpha=1-transparency_slider.value)
    ovl = np.rot90(ovl)


    p_axial.image_rgba(image=[bg,ovl], x=[0,0], y=[0,0], dw=bg.shape[1] , dh=bg.shape[0]) #note that bg has been rotated here, so coords for dw and dh are different than expected!
    if (toggle_regions.active):
        plot_axial_region()


def plot_axial_region():
    if debug: print("Called plot_axial_region().")

    rg = aal_drawn[slice_slider_axial.value-1,:,:] #rg = region
    rg = region2RGBA(rg)
    rg = np.rot90(rg)

    p_axial.image_rgba(image=[rg], x=[0], y=[0], dw=rg.shape[1], dh=rg.shape[0],  level='glyph')

def plot_sagittal():
    if debug: print("Called plot_sagittal().")

    bg = m.subj_bg[:,slice_slider_sagittal.value-1,:]
    bg = bg2RGBA(bg)
    bg = np.flipud(bg)

    ovl = overlay[:,slice_slider_sagittal.value-1,:]
    ovl = overlay2RGBA(ovl, alpha=1-transparency_slider.value)
    ovl = np.flipud(ovl)


    p_sagittal.image_rgba(image=[bg,ovl], x=[0,0], y=[0,0], dw=bg.shape[1], dh=bg.shape[0])
    if (toggle_regions.active):
        plot_sagittal_region()

def plot_sagittal_region():
    if debug: print("Called plot_sagittal_region().")

    rg = aal_drawn[:,slice_slider_sagittal.value-1,:] #rg = region
    rg = region2RGBA(rg)

    p_sagittal.image_rgba(image=[rg], x=[0], y=[0], dw=rg.shape[1], dh=rg.shape[0],  level='glyph')


def disable_widgets():
    if debug: print("Called disable_widgets().")

    model_select.update(disabled=True)
    subject_select.update(disabled=True)
    slice_slider_frontal.update(disabled=True)
    slice_slider_axial.update(disabled=True)
    slice_slider_sagittal.update(disabled=True)
    threshold_slider.update(disabled=True)
    clustersize_slider.update(disabled=True)
    transparency_slider.update(disabled=True)
    toggle_regions.update(disabled=True)

def enable_widgets():
    if debug: print("Called enable_widgets().")

    model_select.update(disabled=False)
    subject_select.update(disabled=False)
    slice_slider_frontal.update(disabled=False)
    slice_slider_axial.update(disabled=False)
    slice_slider_sagittal.update(disabled=False)
    threshold_slider.update(disabled=False)
    clustersize_slider.update(disabled=False)
    transparency_slider.update(disabled=False)
    toggle_regions.update(disabled=False)
