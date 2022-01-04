print("###################################################")
print("Estiamte seed germination rate from coloured images")
print("###################################################")

########################
### Import libraries ###
########################
import os, sys, argparse, re            ### I/O libraries
import numpy as np                      ### Numerical library, mostly for vector, matrix and array operations
import multiprocessing                  ### Parallel processing library
from functools import partial           ### Allows iterating across the first argument while the other arguments are defined - used in designating the input image file to a corresponding thread for execution.
import tqdm                             ### Progress bar library which allows tracking of parallel processes.
import warnings                         ### Warnings library to ignore warnings and render the progress bar readable.
warnings.filterwarnings("ignore")
from functions import *                 ### seedGermCV logic file containing the image processing and analysis functions , i.e. "seedGermCV/src/functions.py"

###############################################
### Define the user input parsing function  ###
###############################################
### convert string input into a list
def parse_str_to_list(str_input, vec_delimiters=[',', ';', ':'], vec_brackets=['\[', '\]', '\(', '\)', '\{', '\}'], input_name="test", list_type="int", list_len=2):
    ####################
    ## TEST
    # str_input = "[123, 456]"
    # str_input = "blue,0,0,1"
    # vec_delimiters=[',', ';', ':']
    # vec_brackets=['\[', '\]', '\(', '\)', '\{', '\}']
    # input_name="test"
    # list_type="int"
    # list_len=2
    ####################
    ### convert all delimeters into commas
    string = str_input
    for d in vec_delimiters:
        string = re.sub(d, ",", string)
    ### remove brackets
    for b in vec_brackets:
        string = re.sub(b, "", string)
    ### split into a list
    vec_input = string.split(',')
    ### remove whitespace
    vec_input = [x.strip(' ') for x in vec_input]
    ### convert elemets into the required type
    if list_type=="int":
        vec_input = [int(eval(x)) for x in vec_input]
    elif list_type=="float":
        vec_input = [float(eval(x)) for x in vec_input]
    elif list_type=="bool":
        vec_input = [bool(eval(x)) for x in vec_input]
    elif list_type=="str":
        vec_input = [str(x) for x in vec_input]
    else:
        print("Input error in: --" + input_name )
        print(str_input + ": invalid input.")
        print("Please check the help documentation.")
        exit()
    ### check length
    if len(vec_input) != list_len:
        print("Input error in: --" + input_name )
        print(str_input + ": invalid input.")
        print("Expecting " + str(list_len) + " elements but got " + str(len(vec_input)) + " elements instead.")
        exit()
    ### output
    return(vec_input)

### convert a vector of string input into a dictionary
def parse_vec_str_to_dic(vec_str_input, dic_elem_len=3, dic_type="float", input_name="test"):
    ####################
    ### TEST
    # vec_str_input = ["grayscale", "0.2125", "0.7154", "0.0721", "red", "1.0", "0.0", "0.0", "green", "0.0", "1.0", "0.0"]
    # dic_elem_len = 3
    # input_name = "test"
    ####################
    ### extract number of rows and columns based on the input strings
    n_cols = dic_elem_len + 1 ### number of elements of each dictionary entry plus the entry name or identifier
    n_rows = int(len(vec_str_input) / n_cols)
    ### check length of input strings
    if len(vec_str_input) != (n_cols * n_rows):
        print("Input error in: --" + input_name )
        print(vec_str_input + ": invalid input.")
        print("Please check the help documentation.")
        exit()
    ### generate matrix of input strings where the first column will be the dictionary identifiers
    mat_str_input = np.reshape(vec_str_input, (n_rows, n_cols))
    ### extract the dictionary identifiers from the first column
    vec_id = mat_str_input[:,0]
    if dic_type=="int":
        ### initialise the dictionary
        dic_out = {vec_id[0]: [int(x) for x in mat_str_input[0,1:(n_cols+1)]]}
        ### append to the dictionary
        for i in range(1, n_rows):
            id = vec_id[i]
            dic_out[id] = [int(x) for x in mat_str_input[i,1:(n_cols+1)]]
    elif dic_type=="float":
        ### initialise the dictionary
        dic_out = {vec_id[0]: [float(x) for x in mat_str_input[0,1:(n_cols+1)]]}
        ### append to the dictionary
        for i in range(1, n_rows):
            id = vec_id[i]
            dic_out[id] = [float(x) for x in mat_str_input[i,1:(n_cols+1)]]
    elif dic_type=="str":
        ### initialise the dictionary
        dic_out = {vec_id[0]: [str(x) for x in mat_str_input[0,1:(n_cols+1)]]}
        ### append to the dictionary
        for i in range(1, n_rows):
            id = vec_id[i]
            dic_out[id] = [str(x) for x in mat_str_input[i,1:(n_cols+1)]]
    else:
        print("Type " + dic_type + " not supported.")
        exit()
    ### output
    return(dic_out)

### seed germ function wrapper
def wrapper_fun_frac_shoot_emergence(vec_input, fname,  dir_output, write_out, plot_out, plate_shape, dic_flattenTypes, debug):
            blur_threshold = vec_input[0]
            vec_plate_radius_or_height_limit = vec_input[1]
            vec_RGB_mode_expected = vec_input[2]
            dic_fracThesholds = vec_input[3]
            shoot_area_limit = vec_input[4]
            vec_green_hue_limit = vec_input[5]
            vec_green_sat_limit = vec_input[6]
            vec_green_val_limit = vec_input[7]
            seed_area_limit = vec_input[8]
            vec_seed_hue_limit = vec_input[9]
            vec_seed_sat_limit = vec_input[10]
            vec_seed_val_limit = vec_input[11]
            shoot_axis_ratio_min_diff = vec_input[12]
            seed_axis_ratio_min_diff = vec_input[13]
            suffix_out = "_".join(map(str, vec_input))
            out = fun_frac_shoot_emergence(fname=fname, 
                                        dir_output=dir_output,
                                        write_out=write_out,
                                        plot_out=plot_out,
                                        suffix_out=suffix_out,
                                        blur_threshold=blur_threshold,
                                        plate_shape=plate_shape,
                                        vec_plate_radius_or_height_limit=vec_plate_radius_or_height_limit,
                                        vec_plate_width_limit=vec_plate_radius_or_height_limit,
                                        vec_RGB_mode_expected=vec_RGB_mode_expected,
                                        dic_flattenTypes=dic_flattenTypes,
                                        dic_fracThesholds=dic_fracThesholds,
                                        shoot_area_limit=shoot_area_limit,
                                        vec_green_hue_limit=vec_green_hue_limit,
                                        vec_green_sat_limit=vec_green_sat_limit,
                                        vec_green_val_limit=vec_green_val_limit,
                                        seed_area_limit=seed_area_limit,
                                        vec_seed_hue_limit=vec_seed_hue_limit,
                                        vec_seed_sat_limit=vec_seed_sat_limit,
                                        vec_seed_val_limit=vec_seed_val_limit,
                                        shoot_axis_ratio_min_diff=shoot_axis_ratio_min_diff,
                                        seed_axis_ratio_min_diff=seed_axis_ratio_min_diff,
                                        debug=debug)
            return(out)

#################################
### Define the main function  ###
#################################
def main():
    ### extract user inputs
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_directory", required=True,
        help="Input directory of images.")
    ap.add_argument("-e", "--extension_name", type=str, default="jpg",
        help="Extension name of input images.")
    ap.add_argument("-o", "--output_directory", type=str, default=".",
        help="Output directory.")
    ap.add_argument("-b", "--blur_threshold", type=int, default=1000,
        help="Minimum Laplacian variance threshold below which the image is deemed blurry.")
    ap.add_argument("-j", "--plate_shape", type=str, default="round",
        help="Plate shape. Choose from 'round' or 'rectangular'.")
    ap.add_argument("-p", "--vec_plate_radius_or_height_limit", type=str, default="[1000, 2000]",
        help="Minimum and maximum expected plate radius or height in pixels. Enter as '-p 1000,2000' or -p \"[1000, 2000]\", etc...")
    ap.add_argument("-z", "--vec_plate_width_limit", type=str, default="None",
        help="Minimum and maximum expected plate width in pixels. Enter as '-p 1000,2000' or -p \"[1000, 2000]\", etc...")
    ap.add_argument("-C", "--central_round_plate", type=str, default="True",
        help="Find the central round plate if True; else find the best fitting round plate.")
    ap.add_argument("-m", "--vec_RGB_mode_expected", type=str, default="[45, 55, 75]",
        help="Expected colour of the most common pixel after black. Express in RGB values ranging from 0 to 255. Enter as '-m 45,55,75' or -m \"[45, 75,55]\" etc...")
    ap.add_argument("-f", "--flatten_additional", action='append', default=['grayscale', '0.2125', '0.7154', '0.0721', 'red', '1.0', '0.0', '0.0', 'green', '0.0', '1.0', '0.0'],
        help="Additional dictionary item/s for image flattening. Includes the key name and the three RGB value coefficients. Enter as '-f blue,0.0,0.0,1.0 -f red_green,0.5,0.5,0.0 etc...")
    ap.add_argument("-t", "--flatten_additional_area_thresh", action='append', default=['grayscale', '0.2', 'red', '0.2', 'green', '0.2'],
        help="Additional dictionary item/s for image flattening maximum area threshold. Includes the key name the same as the one used in --flatten_additional and maximum expected fraction of area kept after filtering. Enter as '-t blue,0.2 -t red_green,0.5 etc...")
    ap.add_argument("-s", "--shoot_area_limit", type=str, default="[100, 10000]",
        help="Minimum and maximum expected area of shoots in pixels. Enter as '-s 100,10000' or -s \"[100, 10000]\", etc...")
    ap.add_argument("-u", "--vec_green_hue_limit", type=str, default="[60/360, 150/360]",
        help="Minimum and maximum expected hue of green shoots ranging from 0 to 1. Enter as '-u 0.17,0.42' or -u \"[60/360, 150/360]\", etc...")
    ap.add_argument("-a", "--vec_green_sat_limit", type=str, default="[0.25, 1.00]",
        help="Minimum and maximum expected saturation values of green shoots ranging from 0 to 1. Enter as '-a 0.25,1.00' or -a \"[0.25, 1.00]\", etc...")
    ap.add_argument("-v", "--vec_green_val_limit", type=str, default="[0.25, 1.00]",
        help="Minimum and maximum expected values of green shoots ranging from 0 to 1. Enter as '-v 0.25,1.00' or -v \"[0.25, 1.00]\", etc...")
    ap.add_argument("-l", "--seed_area_limit", type=str, default="[100, 1000]",
        help="Minimum and maximum expected area of seeds in pixels. Enter as '-l 100,1000' or -l \"[100, 1000]\", etc...")
    ap.add_argument("-w", "--vec_seed_hue_limit", type=str, default="[50/360, 180/360]",
        help="Minimum and maximum expected hue of seeds ranging from 0 to 1. Enter as '-w 0.14,0.50' or -w \"[50/360, 180/360]\", etc...")
    ap.add_argument("-x", "--vec_seed_sat_limit", type=str, default="[0.20, 1.00]",
        help="Minimum and maximum expected saturation values of seeds ranging from 0 to 1. Enter as '-x 0.20,1.00' or -x \"[0.20, 1.00]\", etc...")
    ap.add_argument("-y", "--vec_seed_val_limit", type=str, default="[0.30, 1.00]",
        help="Minimum and maximum expected values of seeds ranging from 0 to 1. Enter as '-y 0.30,1.00' or -y \"[0.30, 1.00]\", etc...")
    ap.add_argument("-c", "--shoot_axis_ratio_min_diff", type=float, default=0.5,
        help="Minimum absolute difference between 0.5 and the ratio between the shoots' major and minor axes.")
    ap.add_argument("-g", "--seed_axis_ratio_min_diff", type=float, default=0.2,
        help="Minimum absolute difference between 0.5 and the ratio between the seeds' major and minor axes.")
    ap.add_argument("-k", "--explore_parameter_ranges", type=str, default="False",
        help="Explore range of input parameters to find the most suitable. RECOMMENDATION: Use a small subset of photographs with known counts, e.g. 0.0, 1.0, and 0.5 germination rates.")
    ap.add_argument("-q", "--explore_parameter_ranges_lengths", type=str, default="[1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]",
        help="Length of the range of input parameters to explore. Needs to be 14 in total for the 14 parameters we're exploring, i.e. test_blur_threshold, test_vec_plate_radius_or_height_limit, test_vec_RGB_mode_expected, test_dic_fracThesholds, test_shoot_area_limit, test_vec_green_hue_limit, test_vec_green_sat_limit, test_vec_green_val_limit, test_seed_area_limit, test_vec_seed_hue_limit, test_vec_seed_sat_limit, test_vec_seed_val_limit, test_shoot_axis_ratio_min_diff, test_seed_axis_ratio_min_diff.")
    ap.add_argument("-r", "--compute_areas", type=str, default="False",
        help="Estimate shoot and combined seed/root areas.")
    ap.add_argument("-d", "--debug", type=str, default="False",
        help="Debug.")
    ### parse user inputs
    args = vars(ap.parse_args())
    ### define input and output directories
    input_directory = args["input_directory"] #.input_directory
    extension_name = args["extension_name"] #.extension_name
    output_directory = args["output_directory"] #.output_directory
    if output_directory == ".":
        output_directory = os.path.join(input_directory, "OUTPUT")
    try:
        os.mkdir(output_directory)
    except:
        0
    ### define user input parameters
    blur_threshold = args["blur_threshold"]
    plate_shape = args["plate_shape"]
    vec_plate_radius_or_height_limit = parse_str_to_list(str_input=args["vec_plate_radius_or_height_limit"], input_name="vec_plate_radius_or_height_limit", list_type="int", list_len=2)
    vec_plate_width_limit = args["vec_plate_width_limit"]
    if vec_plate_width_limit=="None":
        vec_plate_width_limit=None
    else:
        vec_plate_width_limit = parse_str_to_list(str_input=args["vec_plate_width_limit"], input_name="vec_plate_width_limit", list_type="int", list_len=2)
    if (args["central_round_plate"]=="False") or (args["central_round_plate"]=="F") or (args["central_round_plate"]=="FALSE") or (args["central_round_plate"]=="false") or (args["central_round_plate"]=="f"):
        central_round_plate = False
    else:
        central_round_plate = True
    vec_RGB_mode_expected = args["vec_RGB_mode_expected"]
    if vec_RGB_mode_expected=="None":
        vec_RGB_mode_expected = None
    else:
        vec_RGB_mode_expected = parse_str_to_list(str_input=vec_RGB_mode_expected, input_name="vec_RGB_mode_expected", list_type="int", list_len=3)
    ### create a dictionary for dic_flattenTypes
    vec_flatten = args["flatten_additional"][0:12]
    if len(args["flatten_additional"]) > 12:
        vec_flatten_additional = []
        for i in range(12, len(args["flatten_additional"])):
            vec_flatten_additional = vec_flatten_additional + parse_str_to_list(str_input=args["flatten_additional"][i], input_name="flatten_additional", list_type="str", list_len=4)
        vec_flatten = vec_flatten + vec_flatten_additional
    dic_flattenTypes = parse_vec_str_to_dic(vec_str_input=vec_flatten, dic_elem_len=3, dic_type="float", input_name="flatten_additional")
    ### create a dictionary for dic_fracThesholds
    vec_threshold = args["flatten_additional_area_thresh"][0:6]
    if len(args["flatten_additional_area_thresh"]) > 6:
        vec_dic_fracThesholds = []
        for i in range(6, len(args["flatten_additional_area_thresh"])):
            vec_dic_fracThesholds = vec_dic_fracThesholds + parse_str_to_list(str_input=args["flatten_additional_area_thresh"][i], input_name="dic_fracThesholds", list_type="str", list_len=2)
        vec_threshold = vec_threshold + vec_dic_fracThesholds
    dic_fracThesholds = parse_vec_str_to_dic(vec_str_input=vec_threshold, dic_elem_len=1, dic_type="float", input_name="dic_fracThesholds")
    shoot_area_limit = parse_str_to_list(str_input=args["shoot_area_limit"], input_name="shoot_area_limit", list_type="int", list_len=2)
    vec_green_hue_limit = parse_str_to_list(str_input=args["vec_green_hue_limit"], input_name="vec_green_hue_limit", list_type="float", list_len=2)
    vec_green_sat_limit = parse_str_to_list(str_input=args["vec_green_sat_limit"], input_name="vec_green_sat_limit", list_type="float", list_len=2)
    vec_green_val_limit = parse_str_to_list(str_input=args["vec_green_val_limit"], input_name="vec_green_val_limit", list_type="float", list_len=2)
    seed_area_limit = parse_str_to_list(str_input=args["seed_area_limit"], input_name="seed_area_limit", list_type="int", list_len=2)
    vec_seed_hue_limit = parse_str_to_list(str_input=args["vec_seed_hue_limit"], input_name="vec_seed_hue_limit", list_type="float", list_len=2)
    vec_seed_sat_limit = parse_str_to_list(str_input=args["vec_seed_sat_limit"], input_name="vec_seed_sat_limit", list_type="float", list_len=2)
    vec_seed_val_limit = parse_str_to_list(str_input=args["vec_seed_val_limit"], input_name="vec_seed_val_limit", list_type="float", list_len=2)
    shoot_axis_ratio_min_diff = args["shoot_axis_ratio_min_diff"]
    seed_axis_ratio_min_diff = args["seed_axis_ratio_min_diff"]
    if (args["explore_parameter_ranges"]=="False") or (args["explore_parameter_ranges"]=="F") or (args["explore_parameter_ranges"]=="FALSE") or (args["explore_parameter_ranges"]=="false") or (args["explore_parameter_ranges"]=="f"):
        explore_parameter_ranges = False
    else:
        explore_parameter_ranges = True
        temp_explore_parameter_ranges_lengths = parse_str_to_list(str_input=args["explore_parameter_ranges_lengths"], input_name="explore_parameter_ranges_lengths", list_type="str", list_len=14*2)
        explore_parameter_ranges_lengths =  parse_vec_str_to_dic(vec_str_input=temp_explore_parameter_ranges_lengths, input_name="explore_parameter_ranges_lengths", dic_type="int", dic_elem_len=1)
    if (args["compute_areas"]=="False") or (args["compute_areas"]=="F") or (args["compute_areas"]=="FALSE") or (args["compute_areas"]=="false") or (args["compute_areas"]=="f"):
        compute_areas = False
    else:
        compute_areas = True
    if (args["debug"]=="False") or (args["debug"]=="F") or (args["debug"]=="FALSE") or (args["debug"]=="false") or (args["debug"]=="f"):
        debug = False
    else:
        debug = True
    ##################################################################
    ### TEST
    # input_directory = "/home/jeff/Documents/seedGermCV/res/Arabidopsis"
    # extension_name = "JPG"
    # output_directory = os.path.join(input_directory, "OUTPUT")
    # blur_threshold = 1000
    # plate_shape = "round"
    # vec_plate_radius_or_height_limit = [1000, 2000]
    # vec_RGB_mode_expected = [45, 55, 75]
    # dic_flattenTypes = {'grayscale':[0.2125, 0.7154, 0.0721], 'red':[1.0, 0.0, 0.0], 'green':[0.0, 1.0, 0.0]}
    # dic_fracThesholds = {'grayscale':[0.20], 'red':[0.20], 'green':[0.20]}
    # shoot_area_limit = [100, 10000]
    # vec_green_hue_limit = [60/360, 150/360]
    # vec_green_sat_limit = [0.25, 1.00]
    # vec_green_val_limit = [0.25, 1.00]
    # seed_area_limit = [100, 1000]
    # vec_seed_hue_limit = [50/360, 180/360]
    # vec_seed_sat_limit = [0.20, 1.00]
    # vec_seed_val_limit = [0.30, 1.00]
    # shoot_axis_ratio_min_diff = 0.5
    # seed_axis_ratio_min_diff = 0.2
    # explore_parameter_ranges = False
    # explore_parameter_ranges_lengths = {"blur_threshold": [1], "vec_plate_radius_or_height_limit": [5], "vec_RGB_mode_expected": [1], "dic_fracThesholds": [1], "shoot_area_limit": [1], "vec_green_hue_limit": [1], "vec_green_sat_limit": [1], "vec_green_val_limit": [1], "seed_area_limit": [1], "vec_seed_hue_limit": [1], "vec_seed_sat_limit": [1], "vec_seed_val_limit": [1], "shoot_axis_ratio_min_diff": [1], "seed_axis_ratio_min_diff": [1]}
    # debug = False
    ##################################################################
    print("##############################################################################")
    print("Input parameters:")
    print("##############################################################################")
    print("input_directory: ", str(input_directory))
    print("extension_name: ", str(extension_name))
    print("output_directory: ", str(output_directory))
    print("blur_threshold: ", str(blur_threshold))
    print("plate_shape: ", str(plate_shape))
    print("vec_plate_radius_or_height_limit: ", str(vec_plate_radius_or_height_limit))
    print("vec_plate_width_limit: ", str(vec_plate_width_limit))
    print("vec_RGB_mode_expected: ", str(vec_RGB_mode_expected))
    print("dic_flattenTypes: ", str(dic_flattenTypes))
    print("dic_fracThesholds: ", str(dic_fracThesholds))
    print("shoot_area_limit: ", str(shoot_area_limit))
    print("vec_green_hue_limit: ", str(vec_green_hue_limit))
    print("vec_green_sat_limit: ", str(vec_green_sat_limit))
    print("vec_green_val_limit: ", str(vec_green_val_limit))
    print("seed_area_limit: ", str(seed_area_limit))
    print("vec_seed_hue_limit: ", str(vec_seed_hue_limit))
    print("vec_seed_sat_limit: ", str(vec_seed_sat_limit))
    print("vec_seed_val_limit: ", str(vec_seed_val_limit))
    print("shoot_axis_ratio_min_diff: ", str(shoot_axis_ratio_min_diff))
    print("seed_axis_ratio_min_diff: ", str(seed_axis_ratio_min_diff))
    print("compute_areas: ", str(compute_areas))
    print("##############################################################################")
    ### extract image filenames
    vec_filenames = np.sort([os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith("."+extension_name)])
    ### define static parameters
    write_out = True ### save one-line output csv file of shoot count, seed/seedling count and germination rate
    plot_out = True
    suffix_out = ""


    if explore_parameter_ranges:
        print("###############################################################")
        print("Exploring ranges of input parameters")
        print("We recommend using a very small subset of your images for this.")
        print("Use images with known counts.")
        print("For example plates with 0%, 50%, and 100% germination rates.")
        print("###############################################################")
        ### define range generator function
        def fun_generate_range(x, dic, key, dtype=int, decplac=2):
            num = dic[key][0]
            divisor = num
            out = np.round(np.linspace(np.array(x)/divisor, x, num=num, endpoint=True, dtype=dtype, axis=0), decplac)
            return(out.tolist())
        ### generate  ranges
        test_blur_threshold = fun_generate_range(x=blur_threshold, dic=explore_parameter_ranges_lengths, key="blur_threshold", dtype=int)
        test_vec_plate_radius_or_height_limit = fun_generate_range(x=vec_plate_radius_or_height_limit, dic=explore_parameter_ranges_lengths, key="vec_plate_radius_or_height_limit", dtype=int)
        if vec_RGB_mode_expected is None:
            test_vec_RGB_mode_expected = [None]
        else:    
            test_vec_RGB_mode_expected = fun_generate_range(x=vec_RGB_mode_expected, dic=explore_parameter_ranges_lengths, key="vec_RGB_mode_expected", dtype=int)
        temp_dic_fracThesholds = fun_generate_range(x=list(dic_fracThesholds.values()), dic=explore_parameter_ranges_lengths, key="dic_fracThesholds", dtype=float)
        test_dic_fracThesholds = []
        for i in range(len(temp_dic_fracThesholds)):
            dic = {}
            keys = list(dic_fracThesholds.keys())
            for j in range(len(temp_dic_fracThesholds[0])):
                key = keys[j]
                dic[key] = temp_dic_fracThesholds[i][j]
            test_dic_fracThesholds.append(dic)
        test_shoot_area_limit = fun_generate_range(x=shoot_area_limit, dic=explore_parameter_ranges_lengths, key="shoot_area_limit", dtype=int)
        test_vec_green_hue_limit = fun_generate_range(x=vec_green_hue_limit, dic=explore_parameter_ranges_lengths, key="vec_green_hue_limit", dtype=float)
        test_vec_green_sat_limit = fun_generate_range(x=vec_green_sat_limit, dic=explore_parameter_ranges_lengths, key="vec_green_sat_limit", dtype=float)
        test_vec_green_val_limit = fun_generate_range(x=vec_green_val_limit, dic=explore_parameter_ranges_lengths, key="vec_green_val_limit", dtype=float)
        test_seed_area_limit = fun_generate_range(x=seed_area_limit, dic=explore_parameter_ranges_lengths, key="seed_area_limit", dtype=int)
        test_vec_seed_hue_limit = fun_generate_range(x=vec_seed_hue_limit, dic=explore_parameter_ranges_lengths, key="vec_seed_hue_limit", dtype=float)
        test_vec_seed_sat_limit = fun_generate_range(x=vec_seed_sat_limit, dic=explore_parameter_ranges_lengths, key="vec_seed_sat_limit", dtype=float)
        test_vec_seed_val_limit = fun_generate_range(x=vec_seed_val_limit, dic=explore_parameter_ranges_lengths, key="vec_seed_val_limit", dtype=float)
        test_shoot_axis_ratio_min_diff = fun_generate_range(x=shoot_axis_ratio_min_diff, dic=explore_parameter_ranges_lengths, key="shoot_axis_ratio_min_diff", dtype=float)
        test_seed_axis_ratio_min_diff = fun_generate_range(x=seed_axis_ratio_min_diff, dic=explore_parameter_ranges_lengths, key="seed_axis_ratio_min_diff", dtype=float)

        test_input_vec = []
        for a in test_blur_threshold:
            for b in test_vec_plate_radius_or_height_limit:
                for c in test_vec_RGB_mode_expected:
                    for d in test_dic_fracThesholds:
                        for e in test_shoot_area_limit:
                            for f in test_vec_green_hue_limit:
                                for g in test_vec_green_sat_limit:
                                    for h in test_vec_green_val_limit:
                                        for i in test_seed_area_limit:
                                            for j in test_vec_seed_hue_limit:
                                                for k in test_vec_seed_sat_limit:
                                                    for l in test_vec_seed_val_limit:
                                                        for m in test_shoot_axis_ratio_min_diff:
                                                            for n in test_seed_axis_ratio_min_diff:
                                                                vec = [a, b, c, d, e, f, g, h, i, j, k, l, m, n]
                                                                test_input_vec.append(vec)
        print("Testing " + str(len(test_input_vec)) + " combinations of input parameters on each image input.")
        print("And we have " + str(len(vec_filenames)) + " input images to test.")

        # ### test
        # wrapper_fun_frac_shoot_emergence(vec_input=test_input_vec[0],
        #                                 fname="/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-no-marker-46-49.jpg", 
        #                                 dir_output="/home/jeff/Documents/seedGermCV/res/Arabidopsis/OUTPUT",
        #                                 write_out=write_out,
        #                                 plot_out=plot_out,
        #                                 plate_shape=plate_shape,
        #                                 dic_flattenTypes=dic_flattenTypes,
        #                                 debug=debug)

        ### testing partial
        n_cores = multiprocessing.cpu_count() - 1
        parallel = multiprocessing.Pool(n_cores)
        parallel_out = []
        for fname in vec_filenames:
            for x in tqdm.tqdm(parallel.imap_unordered(partial(wrapper_fun_frac_shoot_emergence, fname=fname, 
                                                                                                dir_output=output_directory,
                                                                                                write_out=write_out,
                                                                                                plot_out=plot_out,
                                                                                                plate_shape=plate_shape,
                                                                                                dic_flattenTypes=dic_flattenTypes,
                                                                                                debug=debug),
                                test_input_vec), total=len(test_input_vec)):
                parallel_out.append(x)
        parallel.close()
        parallel.join()
    else:
        #############################################
        ### parallel processing with progress bar ###
        #############################################
        n_cores = multiprocessing.cpu_count() - 1
        parallel = multiprocessing.Pool(n_cores)
        parallel_out = []
        for result in tqdm.tqdm(parallel.imap_unordered(partial(fun_frac_shoot_emergence, dir_output=output_directory,
                                                                    write_out=write_out,
                                                                    plot_out=plot_out,
                                                                    suffix_out=suffix_out,
                                                                    blur_threshold=blur_threshold,
                                                                    plate_shape=plate_shape,
                                                                    vec_plate_radius_or_height_limit=vec_plate_radius_or_height_limit,
                                                                    vec_plate_width_limit=vec_plate_width_limit,
                                                                    central_round_plate=central_round_plate,
                                                                    vec_RGB_mode_expected=vec_RGB_mode_expected,
                                                                    dic_flattenTypes=dic_flattenTypes,
                                                                    dic_fracThesholds=dic_fracThesholds,
                                                                    shoot_area_limit=shoot_area_limit,
                                                                    vec_green_hue_limit=vec_green_hue_limit,
                                                                    vec_green_sat_limit=vec_green_sat_limit,
                                                                    vec_green_val_limit=vec_green_val_limit,
                                                                    seed_area_limit=seed_area_limit,
                                                                    vec_seed_hue_limit=vec_seed_hue_limit,
                                                                    vec_seed_sat_limit=vec_seed_sat_limit,
                                                                    vec_seed_val_limit=vec_seed_val_limit,
                                                                    shoot_axis_ratio_min_diff=shoot_axis_ratio_min_diff,
                                                                    seed_axis_ratio_min_diff=seed_axis_ratio_min_diff,
                                                                    compute_areas=compute_areas,
                                                                    debug=debug), vec_filenames), total=len(vec_filenames)):
            parallel_out.append(result)
        parallel.close()
        parallel.join()
    ##############
    ### output ###
    ##############
    return(parallel_out)

###############
### Execute ###
###############
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

### TESTS:
# DIR=/data/TransHeat/gen2/germination_photos/seedGermCV-tan-TESTS
# DIR_SRC=${DIR}/seedGermCV/src
# time \
# for DATE in $(ls ${DIR} | grep "^2021")
# do
#     DIR_INPUT=${DIR}/${DATE}
#     DIR_OUTPUT=${DIR_INPUT}/OUTPUT
#     python ${DIR_SRC}/main.py -i ${DIR_INPUT} -e JPG -o ${DIR_OUTPUT}
# done

# DIR_SRC=/home/jeff/Documents/seedGermCV/src
# # DIR_INPUT=/home/jeff/Documents/seedGermCV/res/Arabidopsis
# # DIR_OUTPUT=/home/jeff/Documents/seedGermCV/res/Arabidopsis/OUTPUT
# # EXTENSION_NAME=jpg
# DIR_INPUT=/home/jeff/Downloads/misc/A5_Tray1_Day5
# DIR_OUTPUT="."
# EXTENSION_NAME=JPG
# time python ${DIR_SRC}/main.py -i ${DIR_INPUT} -e ${EXTENSION_NAME} -o ${DIR_OUTPUT}

# # DIR_SRC="/home/jeff/Documents/seedGermCV/src"
# # DIR_INPUT="/home/jeff/Downloads/weedomics_large_files/1.c.2.-Avadex/Batch-A-20211012/TEST-seedGermCV-D10"
# # DIR_OUTPUT="/home/jeff/Downloads/weedomics_large_files/1.c.2.-Avadex/Batch-A-20211012/TEST-seedGermCV-D10"
# DIR_SRC=/data/weedomics/1.c_60_populations_6_pre_herbicides/seedGermCV/src
# DIR=/data/weedomics/1.c_60_populations_6_pre_herbicides/1.c.2.-Avadex
# for d1 in $(ls $DIR | grep "^Batch")
# do
# for d2 in $(ls ${DIR}/${d1} | grep "^D")
# do
# DIR_INPUT=${DIR}/${d1}/${d2}
# DIR_OUTPUT=${DIR_INPUT}/OUTPUT
# EXTENSION_NAME="jpg"
# time \
# python3 ${DIR_SRC}/main.py \
#         -i ${DIR_INPUT} \
#         -e ${EXTENSION_NAME} \
#         -o ${DIR_OUTPUT} \
#         --vec_plate_radius_or_height_limit "[800,1500]" \
#         --vec_plate_width_limit None \
#         --central_round_plate False \
#         --vec_RGB_mode_expected None \
#         --shoot_area_limit="[100, 1000000]" \
#         --vec_green_hue_limit="[60/360, 140/360]" \
#         --vec_green_sat_limit="[0.25, 1.00]" \
#         --vec_green_val_limit="[0.25, 1.00]" \
#         --seed_area_limit="[100, 1000000]" \
#         --vec_seed_hue_limit="[0/360, 45/360]" \
#         --vec_seed_sat_limit="[0.30, 1.0]" \
#         --vec_seed_val_limit="[0.00, 1.00]" \
#         --shoot_axis_ratio_min_diff=0.5 \
#         --seed_axis_ratio_min_diff=0.2 \
#         --compute_areas True
# done
# done
