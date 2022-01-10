print("#######################")
print("Measure seed dimensions")
print("#######################")

import os, argparse                     ### I/O libraries
import numpy as np                      ### Numerical library, mostly for vector, matrix and array operations
import multiprocessing                  ### Parallel processing library
from functools import partial           ### Allows iterating across the first argument while the other arguments are defined - used in designating the input image file to a corresponding thread for execution.
import tqdm                             ### Progress bar library which allows tracking of parallel processes.
from functions import *                 ### SeedMatic logic file containing the image processing and analysis functions , i.e. "seedGermCV/src/functions.py"
import warnings                         ### Warnings library to ignore warnings and render the progress bar readable.
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")
from gooey import Gooey, GooeyParser


@Gooey
def main():
    ### extract user inputs
    parser = GooeyParser(description="Seed dimensions measurement") 
    parser.add_argument('input_directory', widget="DirChooser")
    parser.add_argument('extension_name', default="jpg")
    parser.add_argument('output_directory', widget="DirChooser", default="<input_directory>/OUTPUT")
    parser.add_argument("-a", "--seed_area_minimum", type=int, default=5000,
        help="Minimum contour area which we classify as seed.")
    parser.add_argument("-A", "--seed_area_maximum", default=np.Infinity,
        help="Maximum contour area which we classify as seed.")
    parser.add_argument("-d", "--max_convex_hull_deviation", type=int, default=500,
        help="Maximum deviation from the convex hull perimeter for which the contour is classified as a single seed.")
    parser.add_argument("-s", "--suffix_out", type=str, default="",
        help="Optional suffix of the output files.")
    parser.add_argument("-w", "--write_out", type=str, default="True",
        choices=["True", "False"], widget='Dropdown',
        help="Output seed dimensions csv file per input image?")
    parser.add_argument("-P", "--plot_out", type=str, default="True",
        choices=["True", "False"], widget='Dropdown',
        help="Output seed dimensions image segmentation jpeg file per input image?")
    parser.add_argument("-W", "--plot_width", type=int, default=5,
        help="Plot width in x100 pixels.")
    parser.add_argument("-H", "--plot_height", type=int, default=5,
        help="Plot length in x100 pixels.")
    parser.add_argument("-c", "--concatenate_output", type=str, default="True",
        choices=["True", "False"], widget='Dropdown',
        help="Concatenate output csv files.")
    parser.add_argument("-f", "--concatenate_output_filename", type=str, default="<output_directory>/merged_output.csv",
        help="Filename of the concatenated output csv file.")
    ### parse user inputs
    args = parser.parse_args()
    input_directory = args.input_directory #.input_directory
    extension_name = args.extension_name #.extension_name
    output_directory = args.output_directory #.output_directory
    if output_directory == "<input_directory>/OUTPUT":
        output_directory = os.path.join(input_directory, "OUTPUT")
    try:
        os.mkdir(output_directory)
    except:
        0
    seed_area_minimum = args.seed_area_minimum
    seed_area_maximum = args.seed_area_maximum
    if seed_area_maximum == 'inf':
        seed_area_maximum = np.Infinity
    max_convex_hull_deviation = args.max_convex_hull_deviation
    suffix_out = args.suffix_out
    if args.write_out == "True":
        write_out = True
    else:
        write_out = False
    if args.plot_out == "True":
        plot_out = True
    else:
        plot_out = False
    plot_width = args.plot_width
    plot_height = args.plot_height
    if args.concatenate_output == "True":
        concatenate_output = True
    else:
        concatenate_output = False
    concatenate_output_filename = args.concatenate_output_filename
    if concatenate_output_filename == "<output_directory>/merged_output.csv":
        concatenate_output_filename = os.path.join(output_directory, "merged_output.csv")
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
        print("concatenate_output_filename: ", concatenate_output_filename)
    print("##############################################################################")
    ### extract image filenames
    vec_filenames = np.sort([os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(extension_name)])
    if len(vec_filenames)==0:
        print("No input image files found in: " + input_directory)
        print("with extension name: " + extension_name)
        return(1)
        exit
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
