#!/usr/bin/env python
# coding: utf-8
# controller module

from config import debug
from datamodel import index_lst, sorted_xs, Model
import view
from bokeh.events import Tap, Pan, MouseWheel
import numpy as np


def click_frontal_callback(event):
    """
    Called if user has clicked or dragged the mouse on frontal plot.

    Normalizes the coordinates in the given event and sets the sagittal and axial sliders accordingly, which
    will redraw plots and update cross hair positions.
    :param event: click event with x and y coordinates
    :return: None
    """
    if debug: print("Called click_frontal_callback().")
    global dontplot  # helper variable to not plot the same image twice

    # normalising click coordinates
    if event.x < 1:
        x = 1
    elif event.x > v.slice_slider_sagittal.end:
        x = v.slice_slider_sagittal.end
    else:
        x = int(round(event.x))

    if event.y < 1:
        y = 1
    elif event.y > v.slice_slider_axial.end:
        y = v.slice_slider_axial.end
    else:
        y = int(round(event.y))

    dontplot = True
    v.slice_slider_sagittal.update(value=x)
    dontplot = False
    v.slice_slider_axial.update(value=y)
    if not v.toggle_regions.active: v.plot_sagittal()


def click_axial_callback(event):
    """
    Called if user has clicked or dragged the mouse on axial plot.

    Normalizes the coordinates in the given event and sets the frontal and sagittal sliders accordingly, which
    will redraw plots and update cross hair positions.
    :param event: click event with x and y coordinates
    :return: None
    """
    if debug: print("Called click_axial_callback().")
    global dontplot

    # normalising click coordinates
    if event.x < 1:
        x = 1
    elif event.x > v.slice_slider_sagittal.end:
        x = v.slice_slider_sagittal.end
    else:
        x = int(round(event.x))

    if event.y < 1:
        y = 1
    elif event.y > v.slice_slider_frontal.end:
        y = v.slice_slider_frontal.end
    else:
        y = int(round(event.y))

    dontplot = True
    v.slice_slider_frontal.update(value=v.slice_slider_frontal.end - y + 1)
    dontplot = False
    v.slice_slider_sagittal.update(value=x)
    if not v.toggle_regions.active: v.plot_frontal()


def click_sagittal_callback(event):
    """
    Called if user has clicked or dragged the mouse on sagittal plot.

    Normalizes the coordinates in the given event and sets the frontal and axial sliders accordingly, which
    will redraw plots and update cross hair positions.
    :param event: click event with x and y coordinates
    :return: None
    """
    if debug: print("Called click_sagittal_callback().")
    global dontplot

    # normalising click coordinates
    if event.x < 1:
        x = 1
    elif event.x > v.slice_slider_frontal.end:
        x = v.slice_slider_frontal.end
    else:
        x = int(round(event.x))

    if event.y < 1:
        y = 1
    elif event.y > v.slice_slider_axial.end:
        y = v.slice_slider_axial.end
    else:
        y = int(round(event.y))

    dontplot = True
    v.slice_slider_frontal.update(value=x)
    dontplot = False
    v.slice_slider_axial.update(value=y)
    if not v.toggle_regions.active: v.plot_frontal()


def select_subject_worker():
    if debug: print("Called select_subject_worker().")
    v.curdoc().hold()
    if not v.firstrun:  # Avoid duplicate set_subject() call when application first starts.
        if (v.subject_select.value != "User Upload"):
            if debug: print("Using internal scan.....")
            m.set_subject(index_lst[sorted_xs.index(v.subject_select.value)])  # this parameter is subj_id
        else:
            if debug: print("Using uploaded scan.....")
            m.set_subj_img(m.uploaded_residual)
            m.set_subj_bg(m.uploaded_bg_img)
    if (v.subject_select.value != "User Upload"):
        v.update_covariate_info(index_lst[sorted_xs.index(
            v.subject_select.value)], None)  # called with subj_id; corresponding RID/sid would be m.grps.iloc[m.index_lst[m.sorted_xs.index(v.subject_select.value)], 1]
        v.freeze_covariates()
    else:
        v.update_covariate_info(None, m.entered_covariates_df)
        v.make_covariates_editable()

    v.p_frontal.title.text = "Scan predicted as %0.2f%% Alzheimer\'s" % m.pred
    v.p_axial.title.text = " "
    v.p_sagittal.title.text = " "
    v.render_backround()
    v.apply_thresholds(m.relevance_map, threshold=v.threshold_slider.value, cluster_size=v.clustersize_slider.value)

    v.update_guide_frontal()
    v.update_guide_axial()
    v.update_guide_sagittal()

    v.plot_frontal()
    v.plot_axial()
    v.plot_sagittal()
    v.update_cluster_divs()
    v.enable_widgets()
    v.curdoc().unhold()


def select_subject_callback(attr, old, new):
    """
    Called if new subject has been selected.

    Disables the widgets for user input and calls the worker method for selecting a new subject (which will e.g. re-plot
    the figures etc.).

    :param attr: not used
    :param str old: the old value, not used
    :param str new: the new value of the subject selected (i.e. the name), not used
    :return: None
    """
    if debug: print("Called select_subject_callback().")

    v.disable_widgets()
    v.freeze_covariates()

    # This branching is necessary because the 'next tick' occurs only after the entire main script is run.
    # In that case firstrun would already be set to False and the v.update_guide_*() calls would not initialize properly.
    if v.firstrun:
        select_subject_worker()
    else:
        v.curdoc().add_next_tick_callback(select_subject_worker)


def select_model_worker():
    """
    Sets the new selected model and calls "select_subject_worker".

    :return: None
    """
    if debug: print("Called select_model_worker().")

    # Might take a long time if model is not cached yet.
    m.set_model(v.model_select.value)
    print("Finished selecting new model.")
    v.curdoc().add_next_tick_callback(select_subject_worker)


def select_model_callback(attr, old, new):
    """
    Only disables widgets for user interaction and calls select_model_worker-method.

    :param attr: not used
    :param old: previously selected model, not used
    :param new: the new selected model name ("newmodel/newmodel_cb_....") in the model selection box, not used
    :return: None
    """
    v.disable_widgets()

    if debug: print("Called select_model_callback().")
    v.curdoc().add_next_tick_callback(select_model_worker)


def apply_thresholds_callback(attr, old, new):
    """
    Called if threshold_slider or clustersize_slider value been modified.
    Re-plots the figures and guides with updated overlay.

    :param attr: not used
    :param old: not used
    :param new: not used
    :return: None
    """
    if debug: print("Called apply_thresholds_callback().")

    v.apply_thresholds(m.relevance_map, threshold=v.threshold_slider.value, cluster_size=v.clustersize_slider.value)
    v.update_guide_frontal()
    v.update_guide_axial()
    v.update_guide_sagittal()
    v.update_cluster_divs()
    v.plot_frontal()
    v.plot_axial()
    v.plot_sagittal()


def set_slice_frontal_mouse_wheel(event):
    """
    Called when mouse wheel is used on the frontal slice view.
    
    :param MouseWheel event: float delta contains the (signed) scroll speed.
    :return: None
    """
    if event.delta > 0 and v.slice_slider_frontal.value > v.slice_slider_frontal.start:
        v.slice_slider_frontal.value -= 1
    elif event.delta < 0 and v.slice_slider_frontal.value < v.slice_slider_frontal.end:
        v.slice_slider_frontal.value += 1
    print("Mouse Scroll Event Fired on Frontal")

def set_slice_axial_mouse_wheel(event):
    """
    Called when mouse wheel is used on the axial slice view.
    
    :param MouseWheel event: float delta contains the (signed) scroll speed.
    :return: None
    """
    if event.delta > 0 and v.slice_slider_axial.value < v.slice_slider_axial.end:
        v.slice_slider_axial.value += 1 
    elif event.delta < 0 and v.slice_slider_axial.value > v.slice_slider_axial.start:
        v.slice_slider_axial.value -= 1 
    print("Mouse Scroll Event Fired on Axial")

def set_slice_sagittal_mouse_wheel(event):
    """
    Called when mouse wheel is used on the sagittal slice view.
    
    :param MouseWheel event: float delta contains the (signed) scroll speed.
    :return: None
    """
    if event.delta > 0 and v.slice_slider_sagittal.value < v.slice_slider_sagittal.end:
        v.slice_slider_sagittal.value += 1 
    elif event.delta < 0 and v.slice_slider_sagittal.value > v.slice_slider_sagittal.start:
        v.slice_slider_sagittal.value -= 1 
    print("Mouse Scroll Event Fired on Sagittal")
 

def set_slice_frontal_callback(attr, old, new):
    """
    Called if the slice_slider_frontal has been modified (directly by user or or indirectly via click event)

    Usually re-plots the frontal figure (and the others if region outline is active). Also relocates the cross hairs
    indicating the selected frontal slice.

    :param attr: not used
    :param old: not used
    :param int new: the frontal slider value
    :return: None
    """
    if debug: print("Called set_slice_frontal_callback().")

    v.axial_crosshair_from_frontal.update(location=v.slice_slider_frontal.end - new + 1)
    v.sagittal_crosshair_from_frontal.update(location=new)
    v.update_region_div()
    v.update_cluster_divs()
    if not dontplot:  # is set True if called from click_*_callback()
        v.plot_frontal()
    if v.toggle_regions.active and not dontplot:
        v.plot_axial()
        v.plot_sagittal()


def set_slice_axial_callback(attr, old, new):
    """
    Called if the slice_slider_axial has been modified (directly by user or or indirectly via click event)

    Usually re-plots the axial figure (and the others if region outline is active). Also relocates the cross hairs
    indicating the selected axial slice.

    :param attr: not used
    :param old: not used
    :param int new: the axial slider value
    :return: None
    """
    if debug: print("Called set_slice_axial_callback().")

    v.frontal_crosshair_from_axial.update(location=new)
    v.sagittal_crosshair_from_axial.update(location=new)
    v.update_region_div()
    v.update_cluster_divs()
    if not dontplot:
        v.plot_axial()
    if v.toggle_regions.active and not dontplot:
        v.plot_frontal()
        v.plot_sagittal()


def set_slice_sagittal_callback(attr, old, new):
    """
    Called if the slice_slider_sagittal has been modified (directly by user or or indirectly via click event)

    Usually re-plots the sagittal figure (and the others if region outline is active). Also relocates the cross hairs
    indicating the selected sagittal slice.

    :param attr: not used
    :param old: not used
    :param int new: the sagittal slider value
    :return: None
    """
    if debug: print("Called set_slice_sagittal_callback().")

    v.frontal_crosshair_from_sagittal.update(location=new)
    v.axial_crosshair_from_sagittal.update(location=new)
    v.update_region_div()
    v.update_cluster_divs()
    if not dontplot:
        v.plot_sagittal()
    if v.toggle_regions.active and not dontplot:
        v.plot_frontal()
        v.plot_axial()


def set_transparency_callback(attr, old, new):
    """
    Called if the transparency_slider has been modified by the user.

    Re-plots the figures with updated overlay (with selected transparency).

    :param attr: not used
    :param old: not used
    :param new: not used
    :return: None
    """
    if debug: print("Called set_transparency_callback().")
    v.render_overlay()
    v.plot_frontal()
    v.plot_axial()
    v.plot_sagittal()


def click_show_regions_callback(attr):
    """
    Called if the "Show region outline" button has been pressed.

    Re-plots all figures.

    :param attr:  not used
    :return: None
    """
    if debug: print("Called click_show_regions_callback().")
    v.plot_frontal()
    v.plot_axial()
    v.plot_sagittal()


def flip_frontal_callback(attr):
    """
    Called if "Flip L/R orientation" button has been pressed.

    Updates the orientation labels, figures and sliders with the mirrored versions.

    :param attr: not used
    :return: None
    """
    if debug: print("Called flip_frontal_callback().")
    v.orientation_label_shown_left.update(text="R" if v.flip_frontal_view.active else "L")
    v.orientation_label_shown_right.update(text="L" if v.flip_frontal_view.active else "R")
    v.update_region_div()
    v.slice_slider_sagittal.update(value=np.abs(v.slice_slider_sagittal.end - v.slice_slider_sagittal.value) + 1)
    v.plot_frontal()
    v.plot_axial()
    v.update_guide_sagittal()


def upload_scan_callback(attr, old, new):
    """
    Called if a new file has been uploaded.
    Attempts to load the file as array and enable covariate widgets for entering values
    for residualization.
    Also sets initial values for the covariates.

    :param attr: not used
    :param old: not used
    :param new: the uploaded file encoded as base64 string
    :return: None
    """
    if debug: print("Called upload_scan_callback()")
    is_zipped = v.scan_upload.filename.endswith('.gz')
    # Set initial average values:
    v.age_spinner.update(value=73)
    v.sex_select.update(value='N/A')
    v.tiv_spinner.update(value=1400)
    v.field_strength_select.update(value='3.0')

    m.load_nifti(new, is_zipped)
    v.make_covariates_editable()


def residualize_worker():
    """
    This exists just to pack all work for residualization in one callback method for bokeh's add_next_tick_callback(...)

    :return: None
    """
    print("Called residualize_worker()")
    sex_identifier = 0.5
    if (v.sex_select.value == "female"):
        sex_identifier = 1.0
    elif (v.sex_select.value == "male"):
        sex_identifier = 0.0

    m.residualize(m.uploaded_bg_img, v.age_spinner.value, sex_identifier, v.tiv_spinner.value, float(v.field_strength_select.value))
    # Add user upload to selection box:
    v.subject_select.update(options=(["User Upload"]+sorted_xs), value="User Upload")
    select_subject_worker()


def enter_covariates_callback():
    """
    Called if the 'Start residualization and view scan' button has been pressed.
    Disables widgets, shows 'loading label' and performs residualization work.

    :return: None
    """
    if debug: print("Called enter_covariates_callback")
    v.disable_widgets()

    v.curdoc().add_next_tick_callback(residualize_worker)


dontplot = False
m = Model()  # construct new datamodel object for storing selected subject/cnn model per session
# construct new View object for every session since bokeh models (i.e. sliders, figures, ...)
# cannot be shared across client sessions:
v = view.View(m)

# handle clicking events
v.p_frontal.on_event(Tap, click_frontal_callback)
v.p_frontal.on_event(Pan, click_frontal_callback)
v.p_axial.on_event(Tap, click_axial_callback)
v.p_axial.on_event(Pan, click_axial_callback)
v.p_sagittal.on_event(Tap, click_sagittal_callback)
v.p_sagittal.on_event(Pan, click_sagittal_callback)
# handle Mouse Wheel events
v.p_frontal.on_event(MouseWheel, set_slice_frontal_mouse_wheel)
v.p_axial.on_event(MouseWheel, set_slice_axial_mouse_wheel)
v.p_sagittal.on_event(MouseWheel, set_slice_sagittal_mouse_wheel)

# for jupyter notebook:
# show(layout)
# alternatively, add layout to the document (for bokeh server)
v.curdoc().add_root(v.layout)
v.curdoc().title = 'Online AD brain viewer'

# Set callbacks for events:
v.toggle_regions.on_click(click_show_regions_callback)
v.subject_select.on_change('value', select_subject_callback)

v.curdoc().hold()
select_subject_callback('', '', '')  # call once
# v.curdoc().unhold() #Redundant, is already unhold in select_subject_worker() call.
v.firstrun = False

v.slice_slider_frontal.on_change('value', set_slice_frontal_callback)
v.slice_slider_axial.on_change('value', set_slice_axial_callback)
v.slice_slider_sagittal.on_change('value', set_slice_sagittal_callback)
v.flip_frontal_view.on_click(flip_frontal_callback)
v.threshold_slider.on_change('value', apply_thresholds_callback)
v.clustersize_slider.on_change('value', apply_thresholds_callback)
v.transparency_slider.on_change('value', set_transparency_callback)

v.model_select.on_change('value', select_model_callback)

v.residualize_button.on_click(enter_covariates_callback)
v.scan_upload.on_change("value", upload_scan_callback)

# automatically close bokeh after browser window was closed
# def close_session(session_context):
#    sys.exit()  # Stop the server
# v.curdoc().on_session_destroyed(close_session)
