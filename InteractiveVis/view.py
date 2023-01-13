#!/usr/bin/env python
# coding: utf-8

from bokeh.models.widgets import Select
from bokeh.layouts import column, row, Spacer
from bokeh.plotting import figure, curdoc
# from bokeh.io import output_notebook, push_notebook, show #unused import
from bokeh.models.annotations import Span, ColorBar
from bokeh.models.widgets import Slider
from bokeh.models import Div, Toggle, Label, LinearColorMapper, ColumnDataSource, FileInput, Spinner, Button
import numpy as np
# from PIL import Image
from matplotlib import cm
from matplotlib.colors import rgb2hex
from skimage.measure import label, regionprops

from config import scale_factor
# Constants for first initialization/getter methods; no write access needed.
from datamodel import sorted_xs, stored_models, selected_model, get_region_name, get_region_id, aal_drawn, age, cov_idx, \
    sex, tiv, field
from config import debug, flip_left_right_in_frontal_plot

# Adjusted global color palette for ColorBar annotation, because bokeh does not support some palettes by default:
overlay_colormap = cm.get_cmap('RdYlGn_r')
color_palette = []
for i in range(0, 256):
    color_palette.append(rgb2hex(overlay_colormap(i)))
relevance_guide_color_pos = "#d22a40" # red
relevance_guide_color_neg = "#429e5a" # green # previous: "#36689b" -> dark blue
cluster_size_histogram_color = "#0984ca"


class View:

    def update_region_div(self):
        """
        Called upon change of slider positions.

        Obtains selected region ID from current slider positions and updates the region Div
        with the region name corresponding to the region ID.
        :return: None
        """
        self.region_ID = get_region_id(self.slice_slider_axial.value - 1,
                                       self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
                                       self.slice_slider_frontal.value - 1)
        self.selected_region = get_region_name(self.slice_slider_axial.value - 1,
                                               self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
                                               self.slice_slider_frontal.value - 1)
        self.region_div.update(text="Region: " + self.selected_region)

    def update_cluster_divs(self):
        """
        Called after changes of the relevance map, selected subject, threshold or slider positions.

        Obtains statistical information about the selected cluster's intensity and displays that.
        :return: None
        """
        self.cluster_index = self.clust_labelimg[self.m.subj_bg.shape[
                                                     0] - self.slice_slider_axial.value, self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1, self.slice_slider_frontal.value - 1]
        if self.cluster_index == 0:
            self.cluster_size_div.update(text="Cluster Size: " + "N/A")
            self.cluster_mean_div.update(text="Mean Intensity: " + "N/A")
            self.cluster_peak_div.update(text="Peak Intensity: " + "N/A")
        else:
            self.cluster_size_div.update(
                text="Cluster Size: " + str(self.clust_sizes_drawn[self.cluster_index]) + " vx = " + str(
                    format(self.clust_sizes_drawn[self.cluster_index] * 3.375, '.0f')) + " mm³")
            self.cluster_mean_div.update(
                text="Mean Intensity: " + str(format(self.clust_mean_intensities[self.cluster_index], '.2f')))
            self.cluster_peak_div.update(
                text="Peak Intensity: " + str(format(self.clust_peak_intensities[self.cluster_index], '.2f')))

    def update_covariate_info(self, subj_id, entered_covariates):
        """
        Called if a new subject has been selected.

        Displays the covariate information age, sex, TIV (brain volume) and field strength of the scan.
        Note that exactly one of the parameters should be None.

        :param int subj_id: ID of selected subject, if an internal scan is selected, otherwise None.
        :param pandas.DataFrame entered_covariates: DataFrame of entered covariates if the uploaded scan is selected,
        otherwise None.
        :return: None
        """
        if debug: print("Called update_covariate_info().")
        if subj_id != None:
            self.age_spinner.update(value=age.iloc[cov_idx[subj_id]] if self.subject_select.value != "User Upload"
                                    else self.entered_covariates.get("Age"))
            if (sex.iloc[cov_idx[subj_id]] == 1):
                self.sex_select.update(value="female")
            elif (sex.iloc[cov_idx[subj_id]] == 0):
                self.sex_select.update(value="male")
            else:
                self.sex_select.update(value="N/A")
            self.tiv_spinner.update(value=tiv.iloc[cov_idx[subj_id]])
            self.field_strength_select.update(value=str(field.iloc[cov_idx[subj_id]]))
        else:
            self.age_spinner.update(value=int(entered_covariates['Age'].values[0]))
            if (entered_covariates['Sex'].values[0] == 1):
                self.sex_select.update(value="female")
            elif (entered_covariates['Sex'].values[0] == 0):
                self.sex_select.update(value="male")
            else:
                self.sex_select.update(value="N/A")
            self.tiv_spinner.update(value=float(entered_covariates['TIV'].values[0]))
            self.field_strength_select.update(value=str(entered_covariates['FieldStrength'].values[0]))

    def apply_thresholds(self, relevance_map, threshold=0.5, cluster_size=20):
        if debug: print("Called apply_thresholds().")
        self.overlay = np.copy(relevance_map)
        self.overlay[np.abs(self.overlay) < threshold] = 0  # completely hide low values
        # cluster_size filtering:
        labelimg = np.copy(self.overlay)
        labelimg[labelimg > 0] = 1  # binarize img
        labelimg = label(labelimg, connectivity=2)
        lprops = regionprops(labelimg, intensity_image=self.overlay)
        self.clust_sizes = []
        self.clust_sizes_drawn = np.zeros(len(lprops) + 1,
                                          dtype=np.uint32)  # just those that show up in the end on canvas
        self.clust_mean_intensities = np.zeros(len(lprops) + 1, dtype=np.float64)
        self.clust_peak_intensities = np.zeros(len(lprops) + 1, dtype=np.float64)
        for lab in lprops:
            self.clust_sizes.append(lab.area)
            self.clust_sizes_drawn[lab.label] = lab.area
            self.clust_mean_intensities[lab.label] = lab.mean_intensity
            self.clust_peak_intensities[lab.label] = lab.max_intensity
            if lab.area < cluster_size:
                labelimg[labelimg == lab.label] = 0  # remove small clusters
                self.clust_sizes_drawn[lab.label] = 0
                self.clust_mean_intensities[lab.label] = 0
                self.clust_peak_intensities[lab.label] = 0
        self.clust_labelimg = np.copy(labelimg)
        labelimg[labelimg > 0] = 1  # create binary mask
        np.multiply(self.overlay, labelimg, out=self.overlay)
        tmp = np.copy(self.overlay)
        tmp[tmp < 0] = 0
        self.sum_pos_frontal = np.sum(tmp, axis=(0, 1))  # sum of pos relevance in slices
        self.sum_pos_axial = np.sum(tmp, axis=(2, 1))
        self.sum_pos_sagittal = np.sum(tmp, axis=(2, 0))
        tmp = np.copy(self.overlay)
        tmp[tmp > 0] = 0
        self.sum_neg_frontal = np.sum(tmp, axis=(0, 1))  # sum of neg relevance in slices
        self.sum_neg_axial = np.sum(tmp, axis=(2, 1))
        self.sum_neg_sagittal = np.sum(tmp, axis=(2, 0))
        self.render_overlay()
        return self.overlay  # result also stored in object variables: self.overlay, self.sum_pos*, self.sum_neg*

    @staticmethod
    def bg2rgba(bg):
        """
        Converts background to RGBA.

        :param numpy.ndarray bg: the background as 3D numpy array.
        :return: the voxel values converted to gray RGBA data.
        :rtype: numpy.ndarray
        """
        if debug: print("Called bg2RGBA().")

        img = bg / 4 + 0.25  # rescale to range of approx. 0..1 float
        img = np.uint8(cm.gray(img) * 255)
        ret = img.view("uint32").reshape(img.shape[:3])  # convert to 3D array of uint32
        return ret

    @staticmethod
    def overlay2rgba(relevance_map, alpha=0.5):
        """
        Converts the 3D relevance map to RGBA.

        :param numpy.ndarray relevance_map: The 3D relevance map.
        :param float alpha: the transparency/the value for the alpha channel.
        :return: the voxel values converted to RGBA data.
        :rtype: numpy.ndarray
        """
        if (debug): print("Called overlay2RGBA().")
        # assume map to be in range of -1..1 with 0 for hidden content
        alpha_mask = np.copy(relevance_map)
        alpha_mask[np.abs(alpha_mask) > 0] = alpha  # final transparency of visible content
        relevance_map = relevance_map / 2 + 0.5  # range 0-1 float
        ovl = np.uint8(overlay_colormap(relevance_map) * 255)  # cm translates range 0 - 255 uint to rgba array
        ovl[:, :, :, 3] = np.uint8(alpha_mask * 255)  # replace alpha channel (fourth dim) with calculated values
        ret = ovl.view("uint32").reshape(ovl.shape[:3])  # convert to 3D array of uint32
        return ret

    @staticmethod
    def region2rgba(rg, region_id, alpha=0.5):
        """
        Paints out the region outline selected via region_ID and converts the resulting 2D array slice to RGBA.

        Other than overlay2RGBA() and bg2RGBA(), this method takes a 2D array
        as input instead of a 3D array. That is because it is not useful to preprocess the
        entire 3D region outline layer, which would have to be redone every time the cross hair
        is pointing at another region.

        :param rg: The 2D region  relevance map.
        :param region_id: the region to plot
        :param alpha: the transparency
        :return: The resulting RGBA array.
        :rtype: numpy.ndarray
        """

        rgcopy = np.copy(rg)
        if region_id != 0:
            rgcopy[rgcopy != region_id] = 0
            rgcopy[rgcopy == region_id] = 1
        else:  # crosshair on background
            rgcopy[True] = 0
        alpha_mask = np.copy(rgcopy)
        alpha_mask[alpha_mask == 1] = alpha
        img = np.uint8(cm.autumn(rgcopy) * 255)
        img[:, :, 3] = np.uint8(alpha_mask * 255)
        ret = img.view("uint32").reshape(img.shape[:2])  # convert to 2D array of uint32
        return ret

    def render_overlay(self):
        """
        Converts the current 3D relevance map to RGBA and assigns it to the corresponding instance attribute,
        ready to be plotted.
        :return: None
        """
        self.rendered_overlay = View.overlay2rgba(self.overlay, alpha=1 - self.transparency_slider.value)

    def update_cluster_sizes_histogram(self):
        if self.firstrun:
            # initialize/create plots
            # calc histogram; clip high values to slider max (=200)
            [histdat, edges] = np.histogram(np.clip(self.clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
            self.hist_frontal = self.clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat,
                                                    left=edges[:-1], right=edges[1:], fill_color=cluster_size_histogram_color,
                                                    line_color=cluster_size_histogram_color, name="hist_cluster_sizes")
        else:  # update plots
            [histdat, edges] = np.histogram(np.clip(self.clust_sizes, a_min=None, a_max=200), bins=self.clust_hist_bins)
            self.hist_frontal.data_source.data = {'bottom': np.zeros(histdat.shape, dtype=int), 'top': histdat,
                                                  'left': edges[:-1], 'right': edges[1:]}

    def update_guide_frontal(self):
        """
        Called if new subject has been selected or the threshold/clustersize sliders have been modified.

        Updates the histogram under the frontal plot.
        :return: None
        """
        x = np.arange(0, self.sum_neg_frontal.shape[0])
        y0 = np.zeros(x.shape, dtype=int)
        if self.firstrun:
            # initialize/create plots
            self.guide_frontal.line(x, y0, color="#000000")
            self.pos_area_frontal = self.guide_frontal.varea(x=x, y1=self.sum_pos_frontal, y2=y0, fill_color=relevance_guide_color_pos,
                                                             fill_alpha=0.8, name="pos_area_frontal")
            self.pos_line_frontal = self.guide_frontal.line(x, y=self.sum_pos_frontal, line_width=2, color=relevance_guide_color_pos,
                                                            name="pos_line_frontal")
            self.neg_area_frontal = self.guide_frontal.varea(x=x, y1=self.sum_neg_frontal, y2=y0, fill_color=relevance_guide_color_neg,
                                                             fill_alpha=0.8, name="neg_area_frontal")
            self.neg_line_frontal = self.guide_frontal.line(x, y=self.sum_neg_frontal, line_width=2, color=relevance_guide_color_neg,
                                                            name="neg_line_frontal")
        else:  # update plots
            self.pos_area_frontal.data_source.data = {'x': x, 'y1': self.sum_pos_frontal, 'y2': y0}
            self.pos_line_frontal.data_source.data = {'x': x, 'y': self.sum_neg_frontal}
            self.neg_area_frontal.data_source.data = {'x': x, 'y1': self.sum_neg_frontal, 'y2': y0}
            self.neg_line_frontal.data_source.data = {'x': x, 'y': self.sum_neg_frontal}

    def update_guide_axial(self):
        """
        Called if new subject has been selected or the threshold/clustersize sliders have been modified.

        Updates the histogram under the axial plot.
        :return: None
        """
        x_mirrored = np.arange(0, self.sum_neg_axial.shape[0])  # needs to be flipped
        x = x_mirrored[::-1]
        y0 = np.zeros(x.shape, dtype=int)
        if self.firstrun:
            # initialize/create plots
            self.guide_axial.line(x, y0, color="#000000")
            self.pos_area_axial = self.guide_axial.varea(x=x, y1=self.sum_pos_axial, y2=y0, fill_color=relevance_guide_color_pos,
                                                         fill_alpha=0.8, name="pos_area_axial")
            self.pos_line_axial = self.guide_axial.line(x, y=self.sum_pos_axial, line_width=2, color=relevance_guide_color_pos,
                                                        name="pos_line_axial")
            self.neg_area_axial = self.guide_axial.varea(x=x, y1=self.sum_neg_axial, y2=y0, fill_color=relevance_guide_color_neg,
                                                         fill_alpha=0.8, name="neg_area_axial")
            self.neg_line_axial = self.guide_axial.line(x, y=self.sum_neg_axial, line_width=2, color=relevance_guide_color_neg,
                                                        name="neg_line_axial")
        else:  # update plots
            self.pos_area_axial.data_source.data = {'x': x, 'y1': self.sum_pos_axial, 'y2': y0}
            self.pos_line_axial.data_source.data = {'x': x, 'y': self.sum_neg_axial}
            self.neg_area_axial.data_source.data = {'x': x, 'y1': self.sum_neg_axial, 'y2': y0}
            self.neg_line_axial.data_source.data = {'x': x, 'y': self.sum_neg_axial}

    def update_guide_sagittal(self):
        """
        Called if new subject has been selected or the threshold/clustersize sliders have been modified.

        Updates the histogram under the sagittal plot.
        :return: None
        """
        x = np.arange(0, self.sum_neg_sagittal.shape[0])
        if self.flip_frontal_view.active: x = np.flip(x)
        y0 = np.zeros(x.shape, dtype=int)
        if self.firstrun:
            # initialize/create plots
            self.guide_sagittal.line(x, y0, color="#000000")
            self.pos_area_sagittal = self.guide_sagittal.varea(x=x, y1=self.sum_pos_sagittal, y2=y0,
                                                               fill_color=relevance_guide_color_pos, fill_alpha=0.8,
                                                               name="pos_area_sagittal")
            self.pos_line_sagittal = self.guide_sagittal.line(x, y=self.sum_pos_sagittal, line_width=2, color=relevance_guide_color_pos,
                                                              name="pos_line_sagittal")
            self.neg_area_sagittal = self.guide_sagittal.varea(x=x, y1=self.sum_neg_sagittal, y2=y0,
                                                               fill_color=relevance_guide_color_neg, fill_alpha=0.8,
                                                               name="neg_area_sagittal")
            self.neg_line_sagittal = self.guide_sagittal.line(x, y=self.sum_neg_sagittal, line_width=2, color=relevance_guide_color_neg,
                                                              name="neg_line_sagittal")
        else:  # update plots
            self.pos_area_sagittal.data_source.data = {'x': x, 'y1': self.sum_pos_sagittal, 'y2': y0}
            self.pos_line_sagittal.data_source.data = {'x': x, 'y': self.sum_neg_sagittal}
            self.neg_area_sagittal.data_source.data = {'x': x, 'y1': self.sum_neg_sagittal, 'y2': y0}
            self.neg_line_sagittal.data_source.data = {'x': x, 'y': self.sum_neg_sagittal}

    def render_backround(self):
        """
        Converts the current 3D background image to RGBA and assigns it to the corresponding instance attribute, ready
        to be plotted.
        :return: None
        """
        if debug: print("Called render_background().")
        self.bg = View.bg2rgba(self.m.subj_bg)

    def plot_frontal(self):
        """
        Called if a new subject has been selected, the overlay heatmap has been modified (e.g. by modifying the
        threshold slider) or if the frontal slider has been modified.

        Selects a 2D slice according to the frontal slider value and changes the figure's underlying ColumnDataSource,
        which will re-plot the frontal figure with background, overlay and region outline (if activated).

        :return: None
        """
        if debug: print("Called plot_frontal().")

        bg = self.bg[:, :, self.slice_slider_frontal.value - 1]
        bg = np.flipud(bg)
        ovl = self.rendered_overlay[:, :, self.slice_slider_frontal.value - 1]
        ovl = np.flipud(ovl)

        if self.toggle_regions.active:
            rg = aal_drawn[:, :, self.slice_slider_frontal.value - 1]  # rg = region
            rg = View.region2rgba(rg, self.region_ID)
        else:
            rg = self.frontal_zeros
        self.frontal_data.data.update(
            image=[np.fliplr(img) for img in [bg, ovl, rg]] if self.flip_frontal_view.active else [bg, ovl, rg])

    def plot_axial(self):
        """
        Called if a new subject has been selected, the overlay heatmap has been modified (e.g. by modifying the
        threshold slider) or if the axial slider has been modified.

        Selects a 2D slice according to the axial slider value and changes the figure's underlying ColumnDataSource,
        which will re-plot the axial figure with background, overlay and region outline (if activated).

        :return: None
        """
        if debug: print("Called plot_axial().")

        bg = self.bg[self.m.subj_bg.shape[0] - self.slice_slider_axial.value, :, :]
        bg = np.rot90(bg)
        ovl = self.rendered_overlay[self.m.subj_bg.shape[0] - self.slice_slider_axial.value, :, :]
        ovl = np.rot90(ovl)

        if self.toggle_regions.active:
            rg = aal_drawn[self.slice_slider_axial.value - 1, :, :]  # rg = region
            rg = View.region2rgba(rg, self.region_ID)
            rg = np.rot90(rg)
        else:
            rg = self.axial_zeros
        self.axial_data.data.update(
            image=[np.fliplr(img) for img in [bg, ovl, rg]] if self.flip_frontal_view.active else [bg, ovl, rg])

    def plot_sagittal(self):
        """
        Called if a new subject has been selected, the overlay heatmap has been modified (e.g. by modifying the
        threshold slider) or if the sagittal slider has been modified.

        Selects a 2D slice according to the sagittal slider value and changes the figure's underlying ColumnDataSource,
        which will re-plot the sagittal figure with background, overlay and region outline (if activated).

        :return: None
        """
        if debug: print("Called plot_sagittal().")

        bg = self.bg[:,
             self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
             :]
        bg = np.flipud(bg)
        ovl = self.rendered_overlay[:,
              self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
              :]
        ovl = np.flipud(ovl)

        if self.toggle_regions.active:
            rg = aal_drawn[:,
                 self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
                 :]  # rg = region
            rg = View.region2rgba(rg, self.region_ID)
        else:
            rg = self.sagittal_zeros
        self.sagittal_data.data.update(image=[bg, ovl, rg])

    def disable_widgets(self):
        """
        Used to disable user interaction with the bokeh widgets, if the application is currently working (e.g. processing a scan).
        :return: None
        """
        if debug: print("Called disable_widgets().")

        self.processing_label.update(visible=True)
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
        """
        Enable user interaction with the widgets again.
        :return: None
        """
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
        self.processing_label.update(visible=False)

    def make_covariates_editable(self):
        """
        Called if a Nifti file has been uploaded and the user should input the covariates.
        :return: None
        """
        if debug: print("Called make_covariates_editable()")
        self.age_spinner.update(disabled=False)
        self.sex_select.update(disabled=False)
        self.tiv_spinner.update(disabled=False)
        self.field_strength_select.update(disabled=False)
        self.prepare_button.update(disabled=False)

    def freeze_covariates(self):
        """
        Called if an internal scan has been selected. In that case covariates should
        not be editable (only for scans uploaded by the user).

        :return: None
        """
        if debug: print("Called freeze_covariates()")
        self.age_spinner.update(disabled=True)
        self.sex_select.update(disabled=True)
        self.tiv_spinner.update(disabled=True)
        self.field_strength_select.update(disabled=True)
        self.prepare_button.update(disabled=True)

    def __init__(self, m):
        if debug: print("Initializing new View object...")
        self.curdoc = curdoc # reference to the current Bokeh document
        self.m = m
        self.firstrun = True
        self.error_flag = True
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
        self.transparency_slider = Slider(start=0, end=1, value=0.3, step=0.05, title="Overlay transparency", width=200)

        # Initialize the figures:
        self.guide_frontal = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:',
                                    toolbar_location=None,
                                    active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.guide_frontal.title.text_font = 'arial'
        self.guide_frontal.title.text_font_style = 'normal'
        # guide_frontal.title.text_font_size = '10pt'
        self.guide_frontal.axis.visible = False
        self.guide_frontal.x_range.range_padding = 0
        self.guide_frontal.y_range.range_padding = 0

        self.guide_axial = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:',
                                  toolbar_location=None,
                                  active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.guide_axial.title.text_font = 'arial'
        self.guide_axial.title.text_font_style = 'normal'
        # guide_axial.title.text_font_size = '10pt'
        self.guide_axial.axis.visible = False
        self.guide_axial.x_range.range_padding = 0
        self.guide_axial.y_range.range_padding = 0

        self.guide_sagittal = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:',
                                     toolbar_location=None,
                                     active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.guide_sagittal.title.text_font = 'arial'
        self.guide_sagittal.title.text_font_style = 'normal'
        # guide_sagittal.title.text_font_size = '10pt'
        self.guide_sagittal.axis.visible = False
        self.guide_sagittal.x_range.range_padding = 0
        self.guide_sagittal.y_range.range_padding = 0

        self.clusthist = figure(plot_width=208, plot_height=70, title='Distribution of cluster sizes:',
                                toolbar_location=None,
                                active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.clusthist.title.text_font = 'arial'
        self.clusthist.title.text_font_style = 'normal'
        self.clusthist.axis.visible = False
        self.clusthist.x_range.range_padding = 0
        self.clusthist.y_range.range_padding = 0

        self.p_frontal = figure(plot_width=int(np.floor(m.subj_bg.shape[1] * scale_factor)),
                                plot_height=int(np.floor(m.subj_bg.shape[0] * scale_factor)), title='',
                                toolbar_location=None,
                                active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.p_frontal.axis.visible = False
        self.p_frontal.x_range.range_padding = 0
        self.p_frontal.y_range.range_padding = 0

        self.flip_frontal_view = Toggle(label='Flip L/R orientation', button_type='default', width=200,
                                        active=flip_left_right_in_frontal_plot)

        self.orientation_label_shown_left = Label(
            text='R' if flip_left_right_in_frontal_plot else 'L',
            render_mode='css', x=3,
            y=self.m.subj_bg.shape[0] - 13,
            text_align='left', text_color='white',
            text_font_size='20px',
            border_line_color='white', border_line_alpha=0,
            background_fill_color='black', background_fill_alpha=0,
            level='overlay', visible=True)
        self.orientation_label_shown_right = Label(
            text='L' if flip_left_right_in_frontal_plot else 'R',
            render_mode='css',
            x=self.m.subj_bg.shape[1] - 3,
            y=self.m.subj_bg.shape[0] - 13,
            text_align='right', text_color='white',
            text_font_size='20px',
            border_line_color='white', border_line_alpha=0,
            background_fill_color='black', background_fill_alpha=0,
            level='overlay', visible=True)

        self.p_frontal.add_layout(self.orientation_label_shown_left, 'center')
        self.p_frontal.add_layout(self.orientation_label_shown_right, 'center')

        # The vertical crosshair line on the frontal view that indicates the selected sagittal slice.
        self.frontal_crosshair_from_sagittal = Span(location=self.slice_slider_sagittal.value - 1, dimension='height',
                                                    line_color='green', line_width=1, render_mode="css")

        # The horizontal crosshair line on the frontal view that indicates the selected axial slice.
        self.frontal_crosshair_from_axial = Span(location=self.slice_slider_axial.value - 1, dimension='width',
                                                 line_color='green', line_width=1, render_mode="css")
        self.p_frontal.add_layout(self.frontal_crosshair_from_sagittal)
        self.p_frontal.add_layout(self.frontal_crosshair_from_axial)

        self.p_axial = figure(plot_width=int(np.floor(m.subj_bg.shape[1] * scale_factor)),
                              plot_height=int(np.floor(m.subj_bg.shape[2] * scale_factor)), title='',
                              toolbar_location=None,
                              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.p_axial.axis.visible = False
        self.p_axial.x_range.range_padding = 0
        self.p_axial.y_range.range_padding = 0

        self.axial_crosshair_from_sagittal = Span(location=self.slice_slider_sagittal.value - 1, dimension='height',
                                                  line_color='green', line_width=1, render_mode="css")
        self.axial_crosshair_from_frontal = Span(
            location=self.slice_slider_frontal.end - self.slice_slider_frontal.value + 1, dimension='width',
            line_color='green', line_width=1, render_mode="css")
        self.p_axial.add_layout(self.axial_crosshair_from_sagittal)
        self.p_axial.add_layout(self.axial_crosshair_from_frontal)

        self.p_sagittal = figure(plot_width=int(np.floor(m.subj_bg.shape[2] * scale_factor)),
                                 plot_height=int(np.floor(m.subj_bg.shape[0] * scale_factor)), title='',
                                 toolbar_location=None,
                                 active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
        self.p_sagittal.axis.visible = False
        self.p_sagittal.x_range.range_padding = 0
        self.p_sagittal.y_range.range_padding = 0

        self.sagittal_crosshair_from_frontal = Span(location=self.slice_slider_frontal.value - 1, dimension='height',
                                                    line_color='green', line_width=1, render_mode="css")
        self.sagittal_crosshair_from_axial = Span(
            location=self.slice_slider_axial.end - self.slice_slider_axial.value - 1, dimension='width',
            line_color='green', line_width=1, render_mode="css")
        self.p_sagittal.add_layout(self.sagittal_crosshair_from_frontal)
        self.p_sagittal.add_layout(self.sagittal_crosshair_from_axial)

        self.processing_label = Label(
            text='Processing scan...', render_mode='css',
            x=self.m.subj_bg.shape[1] // 2,
            y=self.m.subj_bg.shape[2] // 2,
            text_align='center', text_color='white',
            text_font_size='25px', text_font_style='italic',
            border_line_color='white', border_line_alpha=1.0,
            background_fill_color='black', background_fill_alpha=0.5,
            level='overlay', visible=False)

        self.p_axial.add_layout(self.processing_label)

        self.render_backround()

        # create empty plot objects with empty ("fully transparent") ColumnDataSources):
        self.frontal_zeros = np.zeros_like(np.flipud(self.bg[:, :, self.slice_slider_frontal.value - 1]))
        self.axial_zeros = np.zeros_like(
            np.rot90(self.bg[self.m.subj_bg.shape[0] - self.slice_slider_axial.value, :, :]))
        self.sagittal_zeros = np.zeros_like(np.flipud(self.bg[:,
                                                      self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
                                                      :]))
        self.frontal_zeros[True] = 255  # value for a fully transparent pixel
        self.axial_zeros[True] = 255
        self.sagittal_zeros[True] = 255

        self.frontal_data = ColumnDataSource(
            data=dict(image=[self.frontal_zeros, self.frontal_zeros, self.frontal_zeros], x=[0, 0, 0], y=[0, 0, 0]))
        self.axial_data = ColumnDataSource(
            data=dict(image=[self.axial_zeros, self.axial_zeros, self.axial_zeros], x=[0, 0, 0], y=[0, 0, 0]))
        self.sagittal_data = ColumnDataSource(
            data=dict(image=[self.sagittal_zeros, self.sagittal_zeros, self.sagittal_zeros], x=[0, 0, 0], y=[0, 0, 0]))

        self.p_frontal.image_rgba(image="image", x="x", y="y", dw=self.frontal_zeros.shape[1],
                                  dh=self.frontal_zeros.shape[0], source=self.frontal_data)
        self.p_axial.image_rgba(image="image", x="x", y="y", dw=self.axial_zeros.shape[1], dh=self.axial_zeros.shape[0],
                                source=self.axial_data)
        self.p_sagittal.image_rgba(image="image", x="x", y="y", dw=self.sagittal_zeros.shape[1],
                                   dh=self.sagittal_zeros.shape[0], source=self.sagittal_data)
        self.toggle_transparency = Toggle(label='Hide relevance overlay', button_type='default', width=200)
        self.toggle_regions = Toggle(label='Show outline of atlas region', button_type='default', width=200)

        self.region_ID = get_region_id(self.slice_slider_axial.value - 1,
                                       self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
                                       self.slice_slider_frontal.value - 1)
        self.selected_region = get_region_name(self.slice_slider_axial.value - 1,
                                               self.slice_slider_sagittal.end - self.slice_slider_sagittal.value if self.flip_frontal_view.active else self.slice_slider_sagittal.value - 1,
                                               self.slice_slider_frontal.value - 1)
        self.region_div = Div(text="Region: " + self.selected_region, sizing_mode="stretch_both",
                              css_classes=["region_divs"])

        self.cluster_size_div = Div(text="Cluster Size: " + "0", css_classes=[
            "cluster_divs"])  # initialize with 0 because clust_labelimg does not exist yet
        self.cluster_mean_div = Div(text="Mean Intensity: " + "0", css_classes=["cluster_divs"])
        self.cluster_peak_div = Div(text="Peak Intensity: " + "0", css_classes=["cluster_divs"])

        # see InteractiveVis/static/ for default formatting/style definitions
        self.age_spinner = Spinner(title="Age:", placeholder="years", mode="int", low=55, high=99, width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), disabled=True) #no subject selected at time of initialization
        self.sex_select = Select(title="Sex:", value="N/A", options=["male", "female", "N/A"], width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), disabled=True)
        self.tiv_spinner = Spinner(title="TIV:", placeholder="cm³", mode="float", low=1000, high=2100, width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), disabled=True)
        self.field_strength_select = Select(title="Field Strength [T]:", value="1.5", options=["1.5", "3.0"], width=int(np.floor(m.subj_bg.shape[1]*scale_factor)//2 -10), disabled=True)

        # Empty dummy figure to add ColorBar to, because annotations (like a ColorBar) must have a
        # parent figure in Bokeh:
        self.p_color_bar = figure(plot_width=100,
                                  plot_height=int(np.floor(m.subj_bg.shape[0] * scale_factor)),
                                  title='',
                                  toolbar_location=None,
                                  active_drag=None, active_inspect=None, active_scroll=None, active_tap=None,
                                  outline_line_alpha=0.0)
        self.p_color_bar.axis.visible = False
        self.p_color_bar.x_range.range_padding = 0
        self.p_color_bar.y_range.range_padding = 0

        self.color_mapper = LinearColorMapper(palette=color_palette, low=-1, high=1)
        self.color_bar = ColorBar(color_mapper=self.color_mapper, title="Relevance")
        self.p_color_bar.add_layout(self.color_bar)
        self.scan_upload = FileInput(accept='.nii.gz, .nii')
        self.prepare_button = Button(label="Start processing and evaluate scan", disabled=True)
        def dummy():
            pass
        self.prepare_button.on_click(dummy) # TODO: remove this once on_click is working when setting callback only from the model class (bug in Bokeh 2.2.x ?)

        # Initialize column layout:
        self.layout = row(
            column(
                self.subject_select, self.model_select,
                Spacer(height=40, width=200, sizing_mode='scale_width'),
                self.threshold_slider, self.clusthist, self.clustersize_slider, self.transparency_slider,
                self.toggle_transparency, self.toggle_regions, self.region_div,
                column(self.cluster_size_div, self.cluster_mean_div, self.cluster_peak_div)
            ),
            column(
                row(self.age_spinner, self.sex_select, self.tiv_spinner, self.field_strength_select, self.scan_upload, css_classes=["subject_divs"]),
                row(self.prepare_button),
                row(
                    column(self.p_frontal, self.slice_slider_frontal, self.guide_frontal, self.flip_frontal_view),
                    column(self.p_axial, self.slice_slider_axial, self.guide_axial),
                    column(self.p_sagittal, self.slice_slider_sagittal, self.guide_sagittal),
                    column(self.p_color_bar)
                )
            )
        )

        self.clust_hist_bins = list(
            range(0, 250 + 1, 10))  # list from (0, 10, .., 250); range max is slider_max_size+1
