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

def main():
    ### extract user inputs
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_directory", required=True,
        help="Input directory of images.")
    ap.add_argument("-e", "--extension_name", type=str, default="jpg",
        help="Extension name of input images.")
    ap.add_argument("-o", "--output_directory", type=str, default=".",
        help="Output directory.")
    ap.add_argument("-a", "--seed_area_minimum", type=int, default=5000,
        help="Minimum contour area which we classify as seed.")
    ap.add_argument("-A", "--seed_area_maximum", type=int, default=np.Infinity,
        help="Maximum contour area which we classify as seed.")
    ap.add_argument("-d", "--max_convex_hull_deviation", type=int, default=500,
        help="Maximum deviation from the convex hull perimeter for which the contour is classified as a single seed.")
    ap.add_argument("-s", "--suffix_out", type=str, default="",
        help="Optional suffix of the output files.")
    ap.add_argument("-w", "--write_out", type=str, default="True",
        help="Output seed dimensions csv file per input image?")
    ap.add_argument("-P", "--plot_out", type=str, default="True",
        help="Output seed dimensions image segmentation jpeg file per input image?")
    ap.add_argument("-W", "--plot_width", type=int, default=5,
        help="Plot width in x100 pixels.")
    ap.add_argument("-H", "--plot_height", type=int, default=5,
        help="Plot length in x100 pixels.")
    ap.add_argument("-c", "--concatenate_output", type=str, default="False",
        help="Concatenate output csv files.")
    ap.add_argument("-f", "--concatenate_output_filename", type=str, default="merged_output.csv",
        help="Filename of the concatenated output csv file.")
    ### parse user inputs
    args = vars(ap.parse_args())
    input_directory = args["input_directory"] #.input_directory
    extension_name = args["extension_name"] #.extension_name
    output_directory = args["output_directory"] #.output_directory
    if output_directory == ".":
        output_directory = os.path.join(input_directory, "OUTPUT")
    try:
        os.mkdir(output_directory)
    except:
        0
    seed_area_minimum = args["seed_area_minimum"]
    seed_area_maximum = args["seed_area_maximum"]
    max_convex_hull_deviation = args["max_convex_hull_deviation"]
    suffix_out = args["suffix_out"]
    if args["write_out"] == "True":
        write_out = True
    else:
        write_out = False
    if args["plot_out"] == "True":
        plot_out = True
    else:
        plot_out = False
    plot_width = args["plot_width"]
    plot_height = args["plot_height"]
    if args["concatenate_output"] == "True":
        concatenate_output = True
    else:
        concatenate_output = False
    concatenate_output_filename = args["concatenate_output_filename"]
    ##################################################################
    ### TEST
    # input_directory = "/home/jeff/Documents/SeedMatic/misc"
    # extension_name = "jpg"
    # output_directory = "/home/jeff/Documents/SeedMatic/misc/test_out"
    # seed_area_minimum = 5000
    # seed_area_maximum = np.Infinity
    # max_convex_hull_deviation = 500
    # suffix_out = ""
    # write_out = True
    # plot_out = True
    # plot_width = 20
    # plot_height = 20
    # concatenate_output = False
    # concatenate_output_filename = "/home/jeff/Documents/SeedMatic/misc/test_out/merged_output.csv"
    # time python seed_dimensions.py -i /home/jeff/Documents/SeedMatic/misc/images-seeds -e jpg -o /home/jeff/Documents/SeedMatic/misc/test_out
    ##################################################################
    print("##############################################################################")
    print("Input parameters:")
    print("##############################################################################")
    print("input_directory: ", input_directory)
    print("extension_name: ", extension_name)
    print("output_directory: ", output_directory)
    print("seed_area_minimum: ", str(seed_area_minimum))
    print("seed_area_maximum: ", str(seed_area_maximum))
    print("max_convex_hull_deviation: ", str(max_convex_hull_deviation))
    print("suffix_out: ", suffix_out)
    print("write_out: ", str(write_out))
    print("plot_out: ", str(plot_out))
    print("plot_width: ", str(plot_width))
    print("plot_height: ", str(plot_height))
    print("concatenate_output: ", str(concatenate_output))
    if concatenate_output:
        print("concatenate_output_filename: ", output_directory + "/" + concatenate_output_filename)
    print("##############################################################################")
    ### extract image filenames
    vec_filenames = np.sort([os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(extension_name)])
    ### setup parallele processing
    n_cores = multiprocessing.cpu_count() - 1
    parallel = multiprocessing.Pool(n_cores)
    ### parallel processing
    for result in tqdm.tqdm(parallel.imap_unordered(partial(fun_seed_dimensions,
                                                                seed_area_limit=[seed_area_minimum, seed_area_maximum],
                                                                max_convex_hull_deviation=max_convex_hull_deviation,
                                                                plot=False,
                                                                dir_output=output_directory,
                                                                suffix_out=suffix_out,
                                                                write_out=write_out,
                                                                plot_out=plot_out,
                                                                plot_width=plot_width,
                                                                plot_height=plot_height), vec_filenames), total=len(vec_filenames)):
        if concatenate_output:
            try:
                df_out = df_out.append(result)
            except:
                df_out = result
        else:
            x = np.nan ### do nothing
    ### close parallel processes
    parallel.close()
    ### write-out merge output
    if concatenate_output:
        df_out.to_csv(concatenate_output_filename, index=False)
        print("####################################################")
        print("Please find the output file: ")
        print('"' + concatenate_output_filename + '"')
        print("####################################################")
    ### output
    return(0)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()