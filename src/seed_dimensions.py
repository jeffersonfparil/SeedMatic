print("#######################")
print("Measure seed dimensions")
print("#######################")

import os, sys, argparse, re            ### I/O libraries
import numpy as np                      ### Numerical library, mostly for vector, matrix and array operations
import multiprocessing                  ### Parallel processing library
from functools import partial           ### Allows iterating across the first argument while the other arguments are defined - used in designating the input image file to a corresponding thread for execution.
import tqdm                             ### Progress bar library which allows tracking of parallel processes.
import warnings                         ### Warnings library to ignore warnings and render the progress bar readable.
warnings.filterwarnings("ignore")
from functions import *                 ### seedGermCV logic file containing the image processing and analysis functions , i.e. "seedGermCV/src/functions.py"


input_directory = "/home/jeff/Documents/SeedMatic/misc"
extension_name = "jpg"
output_directory = "/home/jeff/Documents/SeedMatic/misc/test_out"
os.chdir(input_directory)


vec_fnames = np.sort([os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.startswith("At-seeds-")])
# vec_fnames = ['../res/At-seeds-Col_1-03.JPG', '../res/At-seeds-Oy_0-04.JPG']
for fname in vec_fnames:
    OUT = fun_seed_dimensions(fname, 
                shoot_area_limit=[5000, np.Infinity],
                max_deviation=500,
                plot=False,
                dir_output=output_directory,
                suffix_out="",
                write_out=True,
                plot_out=True)

# plt.hist(OUT.area); plt.show()
# plt.hist(OUT.length); plt.show()
# plt.hist(OUT.width); plt.show()
