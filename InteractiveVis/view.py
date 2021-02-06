#!/usr/bin/env python
# coding: utf-8

from bokeh.models.widgets import Select
from bokeh.layouts import column, row, Spacer
from bokeh.plotting import figure, curdoc
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models.annotations import Span, ColorBar
from bokeh.models.widgets import Slider
from bokeh.models.glyphs import Rect
from bokeh.models import Div, Toggle, Label, LinearColorMapper
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.colors import rgb2hex
from skimage.measure import label, regionprops

from config import scale_factor
# Constants for first initialization/getter methods; no write access needed.
from datamodel import sorted_xs, stored_models, selected_model, get_region_name, get_region_ID, aal_drawn, age, cov_idx, sex, tiv
from config import debug, flip_left_right_in_frontal_plot

# Adjusted global color palette for ColorBar annotation, because bokeh does not support 'jet' palette by default:
jet_color_palette = []
for i in range(0 , 256):
    jet_color_palette.append(rgb2hex(cm.jet(i)))


class View():
    
    def update_region_div(self):
        self.region_ID = get_region_ID(self.slice_slider_axial.value-1, self.slice_slider_sagittal.value-1, self.slice_slider_frontal.value-1)
        self.selected_region = get_region_name(self.slice_slider_axial.value-1, self.slice_slider_sagittal.value-1, self.slice_slider_frontal.value-1)
        self.region_div.update(text="Region: " + self.selected_region)

    def update_cluster_divs(self):
        cluster_index = clust_labelimg[self.m.subj_bg.shape[0]-self.slice_slider_axial.value, self.slice_slider_sagittal.value-1, self.slice_slider_frontal.value-1]
        if cluster_index == 0:
            self.cluster_size_div.update(text="Cluster Size: " + "N/A")
            self.cluster_mean_div.update(text="Mean Intensity: " + "N/A")
            self.cluster_peak_div.update(text="Peak Intensity: " + "N/A")
        else:
            self.cluster_size_div.update(text="Cluster Size: " + str(clust_sizes_drawn[cluster_index]) + " vx = " + str(format(clust_sizes_drawn[cluster_index]*3.375, '.0f')) + " mm³")
            self.cluster_mean_div.update(text="Mean Intensity: " + str(format(clust_mean_intensities[cluster_index], '.2f')))
            self.cluster_peak_div.update(text="Peak Intensity: " + str(format(clust_peak_intensities[cluster_index], '.2f')))

    def update_subject_divs(self, subj_id):
        if debug: print("Called update_subject_divs().")
        self.age_div.update(text="Age: %.0f " % age.iloc[cov_idx[subj_id]])
        if (sex.iloc[cov_idx[subj_id]] == 1):
            self.sex_div.update(text="Sex: " + "female")
        elif (sex.iloc[cov_idx[subj_id]] == 0):
            self.sex_div.update(text="Sex: " + "male")
        else:
            self.sex_div.update(text="Sex: " + "N/A")
        #tiv_div.update(text="TIV: " + str(tiv.iloc[cov_idx[subj_id]]))
        self.tiv_div.update(text="TIV: %.0f cm³" % tiv.iloc[cov_idx[subj_id]])


    def apply_thresholds(self, relevance_map, threshold = 0.5, cluster_size = 20):
        if debug: print("Called apply_thresholds().")
        global clust_sizes, clust_labelimg, clust_sizes_drawn, clust_mean_intensities, clust_peak_intensities # define global variables to store subject data
        self.overlay = np.copy(relevance_map)
        self.overlay[np.abs(self.overlay) < threshold] = 0 # completely hide low values
        # cluster_size filtering
        labelimg = np.copy(self.overlay)
        labelimg[labelimg>0] = 1 # binarize img
        labelimg = label(labelimg, connectivity=2)
        lprops = regionprops(labelimg, intensity_image=self.overlay)
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
        np.multiply(self.overlay, labelimg, out=self.overlay)
        tmp = np.copy(self.overlay)
        tmp[tmp<0] = 0
        self.sum_pos_frontal = np.sum(tmp, axis=(0,1)) # sum of pos relevance in slices
        self.sum_pos_axial = np.sum(tmp, axis=(2,1))
        self.sum_pos_sagittal = np.sum(tmp, axis=(2,0))
        tmp = np.copy(self.overlay)
        tmp[tmp>0] = 0
        self.sum_neg_frontal = np.sum(tmp, axis=(0,1)) # sum of neg relevance in slices
        self.sum_neg_axial = np.sum(tmp, axis=(2,1))
        self.sum_neg_sagittal = np.sum(tmp, axis=(2,0))
        self.render_overlay()
        return self.overlay # result also stored in object variables: self.overlay, self.sum_pos*, self.sum_neg*

    @staticmethod
    def bg2RGBA(bg):
        if(debug): print("Called bg2RGBA().")
        
        img = bg/4 + 0.25 # rescale to range of approx. 0..1 float
        img = np.uint8(cm.gray(img) * 255)
        ret = img.view("uint32").reshape(img.shape[:3]) # convert to 3D array of uint32
        return ret

    @staticmethod
    def overlay2RGBA(relevance_map, alpha = 0.5):
        if(debug): print("Called overlay2RGBA().")
        # assume map to be in range of -1..1 with 0 for hidden content
        alpha_mask = np.copy(relevance_map)
        alpha_mask[np.abs(alpha_mask) > 0] = alpha # final transparency of visible content
        relevance_map = relevance_map/2 + 0.5 # range 0-1 float
        ovl = np.uint8(cm.jet(relevance_map) * 255) # cm translates range 0 - 255 uint to rgba array
        ovl[:,:,:,3] = np.uint8(alpha_mask * 255) # replace alpha channel (fourth dim) with calculated values
        ret = ovl.view("uint32").reshape(ovl.shape[:3]) # convert to 3D array of uint32
        return ret

    @staticmethod
    def region2RGBA(rg, region_ID, alpha = 0.5):
        # Other than overlay2RBGA() and bg2RGBA(), this method takes a 2D array 
        # as input instead of a 3D array. That is because it is not useful to preprocess the 
        # entire 3D region outline layer, which would have to be redone every time the crosshair
        # is pointing at another region.
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
        ret = img.view("uint32").reshape(img.shape[:2]) # convert to 2D array of uint32
        return ret
    
    def render_overlay(self):
        self.rendered_overlay = View.overlay2RGBA(self.overlay, alpha=1-self.transparency_slider.value)

    def update_guide_frontal(self):
            x = np.arange(0, self.sum_neg_frontal.shape[0])
            if (self.flip_frontal_view.active): x = np.flip(x)
            y0 = np.zeros(x.shape, dtype=int)
            if self.firstrun:
                # initialize/create plots
                self.guide_frontal.line(x, y0, color="#000000")
                self.pos_area_frontal = self.guide_frontal.varea(x=x, y1=self.sum_pos_frontal, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area_frontal")
                self.pos_line_frontal = self.guide_frontal.line(x, y=self.sum_pos_frontal, line_width=2, color="#d22a40", name="pos_line_frontal")
                self.neg_area_frontal = self.guide_frontal.varea(x=x, y1=self.sum_neg_frontal, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area_frontal")
                self.neg_line_frontal = self.guide_frontal.line(x, y=self.sum_neg_frontal, line_width=2, color="#36689b", name="neg_line_frontal")
                # calc histogram; clip high values to slider max (=200)
                [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
                self.hist_frontal = self.clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist_frontal")
                #firstrun_frontal = False
            else: # update plots
                self.pos_area_frontal.data_source.data = {'x':x, 'y1':self.sum_pos_frontal, 'y2':y0}
                self.pos_line_frontal.data_source.data = {'x':x, 'y':self.sum_neg_frontal}
                self.neg_area_frontal.data_source.data = {'x':x, 'y1':self.sum_neg_frontal, 'y2':y0}
                self.neg_line_frontal.data_source.data = {'x':x, 'y':self.sum_neg_frontal}
                [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
                self.hist_frontal.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}
                

    def update_guide_axial(self):
        x_mirrored = np.arange(0, self.sum_neg_axial.shape[0]) #needs to be flipped
        x = x_mirrored[::-1]
        y0 = np.zeros(x.shape, dtype=int)
        if self.firstrun:
            # initialize/create plots
            self.guide_axial.line(x, y0, color="#000000")
            self.pos_area_axial = self.guide_axial.varea(x=x, y1=self.sum_pos_axial, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area_axial")
            self.pos_line_axial = self.guide_axial.line(x, y=self.sum_pos_axial, line_width=2, color="#d22a40", name="pos_line_axial")
            self.neg_area_axial = self.guide_axial.varea(x=x, y1=self.sum_neg_axial, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area_axial")
            self.neg_line_axial = self.guide_axial.line(x, y=self.sum_neg_axial, line_width=2, color="#36689b", name="neg_line_axial")
            # calc histogram; clip high values to slider max (=200)
            [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
            self.hist_axial = self.clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist_axial")
            #firstrun_axial = False
        else: # update plots
            self.pos_area_axial.data_source.data = {'x':x, 'y1':self.sum_pos_axial, 'y2':y0}
            self.pos_line_axial.data_source.data = {'x':x, 'y':self.sum_neg_axial}
            self.neg_area_axial.data_source.data = {'x':x, 'y1':self.sum_neg_axial, 'y2':y0}
            self.neg_line_axial.data_source.data = {'x':x, 'y':self.sum_neg_axial}
            [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
            self.hist_axial.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}

            
    def update_guide_sagittal(self):
        x = np.arange(0, self.sum_neg_sagittal.shape[0])
        y0 = np.zeros(x.shape, dtype=int)
        if self.firstrun:
            # initialize/create plots
            self.guide_sagittal.line(x, y0, color="#000000")
            self.pos_area_sagittal = self.guide_sagittal.varea(x=x, y1=self.sum_pos_sagittal, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area_sagittal")
            self.pos_line_sagittal = self.guide_sagittal.line(x, y=self.sum_pos_sagittal, line_width=2, color="#d22a40", name="pos_line_sagittal")
            self.neg_area_sagittal = self.guide_sagittal.varea(x=x, y1=self.sum_neg_sagittal, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area_sagittal")
            self.neg_line_sagittal = self.guide_sagittal.line(x, y=self.sum_neg_sagittal, line_width=2, color="#36689b", name="neg_line_sagittal")
            # calc histogram; clip high values to slider max (=200)
            [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
            self.hist_sagittal = self.clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist_sagittal")
            #firstrun_sagittal = False
        else: # update plots
            self.pos_area_sagittal.data_source.data = {'x':x, 'y1':self.sum_pos_sagittal, 'y2':y0}
            self.pos_line_sagittal.data_source.data = {'x':x, 'y':self.sum_neg_sagittal}
            self.neg_area_sagittal.data_source.data = {'x':x, 'y1':self.sum_neg_sagittal, 'y2':y0}
            self.neg_line_sagittal.data_source.data = {'x':x, 'y':self.sum_neg_sagittal}
            [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
            self.hist_sagittal.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}


    def render_backround(self):
        if(debug): print("Called render_background().")
        self.bg = View.bg2RGBA(self.m.subj_bg)


    def plot_frontal(self):
        if debug: print("Called plot_frontal().")
        
        bg = self.bg[:,:,self.slice_slider_frontal.value-1]
        bg = np.flipud(bg)
        ovl = self.rendered_overlay[:,:,self.slice_slider_frontal.value-1]
        ovl = np.flipud(ovl)
        
        if (self.toggle_regions.active):
            rg = aal_drawn[:,:,self.slice_slider_frontal.value-1] #rg = region
            rg = View.region2RGBA(rg, self.region_ID)
            
            # This bokeh function call takes > 100ms, so we approve potentially redundant plotting of background 
            # and overlay if only the region outline would have changed. The other preparation of plotted arrays
            # above this line does not really matter performance wise (e.g. region2RGBA() takes ~1ms per slice).
            self.p_frontal.image_rgba(image=[np.fliplr(img) for img in [bg,ovl,rg]] if self.flip_frontal_view.active else [bg,ovl,rg], x=[0,0,0], y=[0,0,0], dw=bg.shape[1], dh=bg.shape[0])
        else:
            self.p_frontal.image_rgba(image=[np.fliplr(img) for img in [bg,ovl]] if self.flip_frontal_view.active else [bg,ovl], x=[0,0], y=[0,0], dw=bg.shape[1], dh=bg.shape[0])


    def plot_axial(self):
        if debug: print("Called plot_axial().")
        
        bg = self.bg[self.m.subj_bg.shape[0]-self.slice_slider_axial.value,:,:]
        bg = np.rot90(bg)
        ovl = self.rendered_overlay[self.m.subj_bg.shape[0]-self.slice_slider_axial.value,:,:]
        ovl = np.rot90(ovl)
        
        if (self.toggle_regions.active):
            rg = aal_drawn[self.slice_slider_axial.value-1,:,:] #rg = region
            rg = View.region2RGBA(rg, self.region_ID)
            rg = np.rot90(rg)
            self.p_axial.image_rgba(image=[bg,ovl,rg], x=[0,0,0], y=[0,0,0], dw=rg.shape[1], dh=rg.shape[0])
        else:
            self.p_axial.image_rgba(image=[bg,ovl], x=[0,0], y=[0,0], dw=bg.shape[1] , dh=bg.shape[0]) #note that bg has been rotated here, so coords for dw and dh are different than expected!


    def plot_sagittal(self):
        if debug: print("Called plot_sagittal().")
        
        bg = self.bg[:,self.slice_slider_sagittal.value-1,:]
        bg = np.flipud(bg)
        ovl = self.rendered_overlay[:,self.slice_slider_sagittal.value-1,:]
        ovl = np.flipud(ovl)
        
        if (self.toggle_regions.active):
            rg = aal_drawn[:,self.slice_slider_sagittal.value-1,:] #rg = region
            rg = View.region2RGBA(rg, self.region_ID)
            self.p_sagittal.image_rgba(image=[bg,ovl,rg], x=[0,0,0], y=[0,0,0], dw=rg.shape[1], dh=rg.shape[0])
        else:
            self.p_sagittal.image_rgba(image=[bg,ovl], x=[0,0], y=[0,0], dw=bg.shape[1], dh=bg.shape[0])


    def disable_widgets(self):
        if debug: print("Called disable_widgets().")

        self.loading_label.update(visible=True)
        self.model_select.update(disabled=True)
        self.subject_select.update(disabled=True)
        self.slice_slider_frontal.update(disabled=True)
        self.slice_slider_axial.update(disabled=True)
        self.slice_slider_sagittal.update(disabled=True)
        self.threshold_slider.update(disabled=True)
        self.clustersize_slider.update(disabled=True)
        self.transparency_slider.update(disabled=True)
        self.toggle_regions.update(disabled=True)
        self.flip_frontal_view.update(disabled=True)
        

    def enable_widgets(self):
        if debug: print("Called enable_widgets().")

        self.model_select.update(disabled=False)
        self.subject_select.update(disabled=False)
        self.slice_slider_frontal.update(disabled=False)
        self.slice_slider_axial.update(disabled=False)
        self.slice_slider_sagittal.update(disabled=False)
        self.threshold_slider.update(disabled=False)
        self.clustersize_slider.update(disabled=False)
        self.transparency_slider.update(disabled=False)
        self.toggle_regions.update(disabled=False)
        self.flip_frontal_view.update(disabled=False)
        self.loading_label.update(visible=False)
        
    
    def __init__(self, m):
        if debug: print("Initializing new View object...")
        self.curdoc = curdoc
        self.m = m
        self.firstrun = True

        self.subject_select = Select(title="Subjects:", value=sorted_xs[0], options=sorted_xs, width=200)
        self.model_select = Select(title="Model:", value=selected_model, options=stored_models, width=200)
        self.slice_slider_frontal = Slider(start=1, end=m.subj_bg.shape[2], value=50, step=1,
                          title="Coronal slice", width=200)
        self.slice_slider_axial = Slider(start=1, end=m.subj_bg.shape[0], value=50, step=1,
                          title="Axial slice", width=200)
        self.slice_slider_sagittal = Slider(start=1, end=m.subj_bg.shape[1], value=50, step=1,
                          title="Sagittal slice", width=200)
        self.threshold_slider = Slider(start=0, end=1, value=0.4, step=0.05,
                          title="Relevance threshold", width=200)
        self.clustersize_slider = Slider(start=0, end=250, value=50, step=10,
                          title="Minimum cluster size", width=200)
        self.transparency_slider = Slider(start=0, end=1, value=0.3, step=0.05,title="Overlay transparency", width=200)

        # initialize the figures
        self.guide_frontal = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
                      active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.guide_frontal.title.text_font = 'arial'
        self.guide_frontal.title.text_font_style = 'normal'
        #guide_frontal.title.text_font_size = '10pt'
        self.guide_frontal.axis.visible = False
        self.guide_frontal.x_range.range_padding = 0
        self.guide_frontal.y_range.range_padding = 0

        self.guide_axial = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
                      active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.guide_axial.title.text_font = 'arial'
        self.guide_axial.title.text_font_style = 'normal'
        #guide_axial.title.text_font_size = '10pt'
        self.guide_axial.axis.visible = False
        self.guide_axial.x_range.range_padding = 0
        self.guide_axial.y_range.range_padding = 0

        self.guide_sagittal = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
                      active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.guide_sagittal.title.text_font = 'arial'
        self.guide_sagittal.title.text_font_style = 'normal'
        #guide_sagittal.title.text_font_size = '10pt'
        self.guide_sagittal.axis.visible = False
        self.guide_sagittal.x_range.range_padding = 0
        self.guide_sagittal.y_range.range_padding = 0

        self.clusthist = figure(plot_width=208, plot_height=70, title='Distribution of cluster sizes:', toolbar_location=None,
                      active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.clusthist.title.text_font = 'arial'
        self.clusthist.title.text_font_style = 'normal'
        self.clusthist.axis.visible = False
        self.clusthist.x_range.range_padding = 0
        self.clusthist.y_range.range_padding = 0

        self.p_frontal = figure(plot_width=int(np.floor(m.subj_bg.shape[1]*scale_factor)), plot_height=int(np.floor(m.subj_bg.shape[0]*scale_factor)), title='',
                  toolbar_location=None,
                  active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.p_frontal.axis.visible = False
        self.p_frontal.x_range.range_padding = 0
        self.p_frontal.y_range.range_padding = 0
        
        self.flip_frontal_view = Toggle(label='Flip L/R orientation', button_type='default', width=200, active=flip_left_right_in_frontal_plot)
        
        self.orientation_label_shown_left = Label(
                         text='R' if flip_left_right_in_frontal_plot else 'L',
                         render_mode='css', x = 3,
                         y = self.m.subj_bg.shape[0]-13,
                         text_align='left', text_color='white',
                         text_font_size='20px',
                         border_line_color='white', border_line_alpha=0,
                         background_fill_color='black', background_fill_alpha=0,
                         level='overlay', visible=True)
        self.orientation_label_shown_right = Label(
                         text='L' if flip_left_right_in_frontal_plot else 'R',
                         render_mode='css',
                         x = self.m.subj_bg.shape[1]-3,
                         y = self.m.subj_bg.shape[0]-13,
                         text_align='right', text_color='white',
                         text_font_size='20px',
                         border_line_color='white', border_line_alpha=0,
                         background_fill_color='black', background_fill_alpha=0,
                         level='overlay', visible=True)

        self.p_frontal.add_layout(self.orientation_label_shown_left, 'center')
        self.p_frontal.add_layout(self.orientation_label_shown_right, 'center')

        # The vertical crosshair line on the frontal view that indicates the selected sagittal slice.
        self.frontal_crosshair_from_sagittal = Span(location = self.slice_slider_sagittal.value-1, dimension='height', line_color='green', line_dash='dashed', line_width=2)

        # The horizontal crosshair line on the frontal view that indicates the selected axial slice.
        self.frontal_crosshair_from_axial = Span(location=self.slice_slider_axial.value-1, dimension='width', line_color='green', line_dash='dashed', line_width=2)
        self.p_frontal.add_layout(self.frontal_crosshair_from_sagittal)
        self.p_frontal.add_layout(self.frontal_crosshair_from_axial)

        self.p_axial = figure(plot_width=int(np.floor(m.subj_bg.shape[1]*scale_factor)), plot_height=int(np.floor(m.subj_bg.shape[2]*scale_factor)), title='',
                  toolbar_location=None,
                  active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.p_axial.axis.visible = False
        self.p_axial.x_range.range_padding = 0
        self.p_axial.y_range.range_padding = 0

        self.axial_crosshair_from_sagittal = Span(location=self.slice_slider_sagittal.value-1, dimension='height', line_color='green', line_dash='dashed', line_width=2)
        self.axial_crosshair_from_frontal = Span(location=self.slice_slider_frontal.end-self.slice_slider_frontal.value + 1, dimension='width', line_color='green', line_dash='dashed', line_width=2)
        self.p_axial.add_layout(self.axial_crosshair_from_sagittal)
        self.p_axial.add_layout(self.axial_crosshair_from_frontal)

        self.p_sagittal = figure(plot_width=int(np.floor(m.subj_bg.shape[2]*scale_factor)), plot_height=int(np.floor(m.subj_bg.shape[0]*scale_factor)), title='',
                  toolbar_location=None,
                  active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.p_sagittal.axis.visible = False
        self.p_sagittal.x_range.range_padding = 0
        self.p_sagittal.y_range.range_padding = 0

        self.sagittal_crosshair_from_frontal = Span(location=self.slice_slider_frontal.value-1, dimension='height', line_color='green', line_dash='dashed', line_width=2)
        self.sagittal_crosshair_from_axial = Span(location=self.slice_slider_axial.end-self.slice_slider_axial.value-1, dimension='width', line_color='green', line_dash='dashed', line_width=2)
        self.p_sagittal.add_layout(self.sagittal_crosshair_from_frontal)
        self.p_sagittal.add_layout(self.sagittal_crosshair_from_axial)
        
        self.loading_label = Label(
                         text='Processing scan...', render_mode='css',
                         x=self.m.subj_bg.shape[1] // 2,
                         y=self.m.subj_bg.shape[2] // 2,
                         text_align='center', text_color='white',
                         text_font_size='25px', text_font_style='italic',
                         border_line_color='white', border_line_alpha=1.0,
                         background_fill_color='black', background_fill_alpha=0.5,
                         level='overlay', visible=False)

        self.p_axial.add_layout(self.loading_label)
        
        self.render_backround()

        self.toggle_regions = Toggle(label='Show outline of atlas region', button_type='default', width=200)

        self.region_ID = get_region_ID(self.slice_slider_axial.value-1, self.slice_slider_sagittal.value-1, self.slice_slider_frontal.value-1)
        self.selected_region = get_region_name(self.slice_slider_axial.value-1, self.slice_slider_sagittal.value-1, self.slice_slider_frontal.value-1)
        self.region_div = Div(text="Region: "+self.selected_region, sizing_mode="stretch_both", css_classes=["region_divs"])
        
        self.cluster_size_div = Div(text="Cluster Size: " + "0", css_classes=["cluster_divs"]) #initialize with 0 because clust_labelimg does not exist yet
        self.cluster_mean_div = Div(text="Mean Intensity: " + "0", css_classes=["cluster_divs"])
        self.cluster_peak_div = Div(text="Peak Intensity: " + "0", css_classes=["cluster_divs"])
        
        # see InteractiveVis/static/
        self.age_div = Div(text="Age: " + "N/A", width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), css_classes=["subject_divs"]) #no subject selected at time of initialization
        self.sex_div = Div(text="Sex: " + "N/A", width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), css_classes=["subject_divs"])
        self.tiv_div = Div(text="TIV: " + "N/A", width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), css_classes=["subject_divs"])
        
        #Empty dummy figure to add ColorBar to, because annotations (like a ColorBar) must have a parent figure in Bokeh:
        self.p_color_bar = figure(plot_width=100,
            plot_height=int(np.floor(m.subj_bg.shape[0]*scale_factor)),# + 70 + self.guide_sagittal.plot_height,
            title='',
            toolbar_location=None,
            active_drag=None, active_inspect=None, active_scroll=None, active_tap=None, outline_line_alpha = 0.0)
        self.p_color_bar.axis.visible = False
        self.p_color_bar.x_range.range_padding = 0
        self.p_color_bar.y_range.range_padding = 0
        
        self.color_mapper = LinearColorMapper(palette=jet_color_palette, low=-1, high=1)
        self.color_bar = ColorBar(color_mapper = self.color_mapper, title="Relevance")
        self.p_color_bar.add_layout(self.color_bar)

        # initialize column layout
        self.layout = row(
                        column(
                            self.subject_select, self.model_select,
                            Spacer(height=40, width=200, sizing_mode='scale_width'),
                            self.threshold_slider, self.clusthist, self.clustersize_slider, self.transparency_slider,
                            self.toggle_regions, self.region_div,
                            column(self.cluster_size_div, self.cluster_mean_div, self.cluster_peak_div)
                        ),
                        column(
                            row(self.age_div, self.sex_div, self.tiv_div, css_classes=["subject_divs"]),
                            row(
                                column(self.p_frontal, self.slice_slider_frontal, self.guide_frontal, self.flip_frontal_view),
                                column(self.p_axial, self.slice_slider_axial, self.guide_axial),
                                column(self.p_sagittal, self.slice_slider_sagittal, self.guide_sagittal),
                                column(self.p_color_bar)
                            )
                        )
                    )

        self.clust_hist_bins = list(range(0, 250+1, 10)) # list from (0, 10, .., 250); range max is slider_max_size+1        
