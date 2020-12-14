# Interactive Visualization Web App

## Setup

Make sure that all dependencies are installed in your Python environment.
Go to the parent directory DeepLearningInteractiveVis and execute:
`pip install -r requirements.txt`

If required, modify the header of [config.py](config.py) to load the correct data and CNN model.

## Running

The interactive bokeh application `InteractiveVis` can be run for inspecting the created relevance maps overlaid on the original input images.

Here, point the Anaconda/command line prompt to the parent directory DeepLearningInteractiveVis and then run bokeh using:

`bokeh serve InteractiveVis --show`
