# SeedMatic
Seed phenotying and germination measurements using coloured photographs.

|**Laboratory**|**Build Status**|**License**|
|:---:|:---:|:---:|
| <a href="https://adaptive-evolution.biosciences.unimelb.edu.au/"><img src="https://adaptive-evolution.biosciences.unimelb.edu.au/Adaptive%20Evolution%20Logo%20mod.png" width="150"></a> | <a href="https://github.com/jeffersonfparil/SeedMatic/actions"><img src="https://github.com/jeffersonfparil/SeedMatic/actions/workflows/python.yml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

## Aims
Measure the following seed phenotypes:
- germination rate,
- projected area,
- volume, and
- dimensions (e.g. length and width).

These phenotypes can affect the following ecologically and agriculturally important seed traits:
- dispersion,
- viability,
- longevity,
- dormancy, and
- yield.

## Usage

### Seed dimensions measurement module:
```
seed_dimensions.py  -i INPUT_DIRECTORY
                   [-e EXTENSION_NAME]
                   [-o OUTPUT_DIRECTORY]
                   [-a SEED_AREA_MINIMUM]
                   [-A SEED_AREA_MAXIMUM]
                   [-d MAX_CONVEX_HULL_DEVIATION]
                   [-s SUFFIX_OUT]
                   [-w WRITE_OUT]
                   [-P PLOT_OUT]
                   [-W PLOT_WIDTH]
                   [-H PLOT_HEIGHT]
                   [-c CONCATENATE_OUTPUT]
                   [-f CONCATENATE_OUTPUT_FILENAME]
                   [-h HELP]
```
**Compulsary argument:**
- *-i, --input_directory*: Input directory of images.

**Optional arguments:**
- *-e, --extension_name*: Extension name of input images [default='jpg'].
- *-o, --output_directory*: Output directory [default='<input_directory>/OUTPUT'].
- *-a, --seed_area_minimum*: Minimum contour area which we classify as seed [default=5000].
- *-A, --seed_area_maximum*: Maximum contour area which we classify as seed [default=inf].
- *-d, --max_convex_hull_deviation*: Maximum deviation from the convex hull perimeter for which the contour is classified as a single seed [default=500].
- *-s, --suffix_out*: Optional suffix of the output files [default=''].
- *-w, --write_out*: Output seed dimensions csv file per input image? [default=True]
- *-P, --plot_out*: Output seed dimensions image segmentation jpeg file per input image? [default=True]
- *-W, --plot_width*: Plot width in x100 pixels. [default=5]
- *-H, --plot_height*: Plot length in x100 pixels. [default=5]
- *-c, --concatenate_output*: Concatenate output csv files. [default=True]
- *-f, --concatenate_output_filename*: Filename of the concatenated output csv file. [default='<output_directory>/merged_output.csv']
- *-h*: Show help message and exit.


### Seed germination assessment module:
```
seed_germination.py  -i INPUT_DIRECTORY
                    [-e EXTENSION_NAME]
                    [-o OUTPUT_DIRECTORY]
                    [-b BLUR_THRESHOLD]
                    [-j PLATE_SHAPE]
                    [-p VEC_PLATE_RADIUS_OR_HEIGHT_LIMIT]
                    [-z VEC_PLATE_WIDTH_LIMIT]
                    [-C CENTRAL_ROUND_PLATE]
                    [-m VEC_RGB_MODE_EXPECTED]
                    [-f FLATTEN_ADDITIONAL]
                    [-t FLATTEN_ADDITIONAL_AREA_THRESH]
                    [-s SHOOT_AREA_LIMIT]
                    [-u VEC_GREEN_HUE_LIMIT]
                    [-a VEC_GREEN_SAT_LIMIT]
                    [-v VEC_GREEN_VAL_LIMIT]
                    [-l SEED_AREA_LIMIT]
                    [-w VEC_SEED_HUE_LIMIT]
                    [-x VEC_SEED_SAT_LIMIT]
                    [-y VEC_SEED_VAL_LIMIT]
                    [-c SHOOT_AXIS_RATIO_MIN_DIFF]
                    [-g SEED_AXIS_RATIO_MIN_DIFF]
                    [-k EXPLORE_PARAMETER_RANGES]
                    [-q EXPLORE_PARAMETER_RANGES_LENGTHS]
                    [-r COMPUTE_AREAS]
                    [-d DEBUG]
                    [-h HELP]
```
**Compulsary argument:**
- *-i, --input_directory*: Input directory of images. (default: None)

**Optional arguments:**
- *-e, --extension_name*: Extension name of input images. (default: 'jpg')
- *-o, --output_directory*: Output directory. (default: '<input_directory>/OUTPUT')
- *-b, --blur_threshold*: Minimum Laplacian variance threshold below which the image is deemed blurry. (default: 1000)
- *-j, --plate_shape*: Plate shape. Choose from 'round' or 'rectangular'. (default: 'round')
- *-p, --vec_plate_radius_or_height_limit*: Minimum and maximum expected plate radius or height in pixels. Enter as '-p 1000,2000' or -p "[1000,
                        2000]", etc... (default: [1000, 2000])
- *-z, --vec_plate_width_limit*: Minimum and maximum expected plate width in pixels. Enter as '-p 1000,2000' or -p "[1000, 2000]", etc...
                        (default: None)
- *-C, --central_round_plate*: Find the central round plate if True; else find the best fitting round plate. (default: True)
- *-m, --vec_RGB_mode_expected*: Expected colour of the most common pixel after black. Express in RGB values ranging from 0 to 255. Enter
                        as '-m 45,55,75' or -m "[45, 75,55]" etc... (default: [45, 55, 75])
- *-f, --flatten_additional*: Additional dictionary item/s for image flattening. Includes the key name and the three RGB value
                        coefficients. Enter as '-f blue,0.0,0.0,1.0 -f red_green,0.5,0.5,0.0 etc... (default: ['grayscale',
                        '0.2125', '0.7154', '0.0721', 'red', '1.0', '0.0', '0.0', 'green', '0.0', '1.0', '0.0'])
- *-t, --flatten_additional_area_thresh*: Additional dictionary item/s for image flattening maximum area threshold. Includes the key name the same
                        as the one used in --flatten_additional and maximum expected fraction of area kept after filtering. Enter
                        as '-t blue,0.2 -t red_green,0.5 etc... (default: ['grayscale', '0.2', 'red', '0.2', 'green', '0.2'])
- *-s, --shoot_area_limit*: Minimum and maximum expected area of shoots in pixels. Enter as '-s 100,10000' or -s "[100, 10000]",
                        etc... (default: [100, 10000])
- *-u, --vec_green_hue_limit*: Minimum and maximum expected hue of green shoots ranging from 0 to 1. Enter as '-u 0.17,0.42' or -u
                        "[60/360, 150/360]", etc... (default: [60/360, 150/360])
- *-a, --vec_green_sat_limit*: Minimum and maximum expected saturation values of green shoots ranging from 0 to 1. Enter as '-a
                        0.25,1.00' or -a "[0.25, 1.00]", etc... (default: [0.25, 1.00])
- *-v, --vec_green_val_limit*: Minimum and maximum expected values of green shoots ranging from 0 to 1. Enter as '-v 0.25,1.00' or -v
                        "[0.25, 1.00]", etc... (default: [0.25, 1.00])
- *-l, --seed_area_limit*: Minimum and maximum expected area of seeds in pixels. Enter as '-l 100,1000' or -l "[100, 1000]", etc...
                        (default: [100, 1000])
- *-w, --vec_seed_hue_limit*: Minimum and maximum expected hue of seeds ranging from 0 to 1. Enter as '-w 0.14,0.50' or -w "[50/360,
                        180/360]", etc... (default: [50/360, 180/360])
- *-x, --vec_seed_sat_limit*: Minimum and maximum expected saturation values of seeds ranging from 0 to 1. Enter as '-x 0.20,1.00' or
                        -x "[0.20, 1.00]", etc... (default: [0.20, 1.00])
- *-y, --vec_seed_val_limit*: Minimum and maximum expected values of seeds ranging from 0 to 1. Enter as '-y 0.30,1.00' or -y "[0.30,
                        1.00]", etc... (default: [0.30, 1.00])
- *-c, --shoot_axis_ratio_min_diff*: Minimum absolute difference between 0.5 and the ratio between the shoots' major and minor axes. (default:
                        0.5)
- *-g, --seed_axis_ratio_min_diff*: Minimum absolute difference between 0.5 and the ratio between the seeds' major and minor axes. (default:
                        0.2)
- *-k, --explore_parameter_ranges*: Explore range of input parameters to find the most suitable. RECOMMENDATION: Use a small subset of
                        photographs with known counts, e.g. 0.0, 1.0, and 0.5 germination rates. (default: False)
- *-q, --explore_parameter_ranges_lengths*: Length of the range of input parameters to explore. Needs to be 14 in total for the 14 parameters we're
                        exploring, i.e. test_blur_threshold, test_vec_plate_radius_or_height_limit, test_vec_RGB_mode_expected,
                        test_dic_fracThesholds, test_shoot_area_limit, test_vec_green_hue_limit, test_vec_green_sat_limit,
                        test_vec_green_val_limit, test_seed_area_limit, test_vec_seed_hue_limit, test_vec_seed_sat_limit,
                        test_vec_seed_val_limit, test_shoot_axis_ratio_min_diff, test_seed_axis_ratio_min_diff. (default: [1, 3,
                        2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1])
- *-r, --compute_areas*: Estimate shoot and combined seed/root areas. (default: False)
- *-d, --debug*: Debug. (default: False)
- *-h*: Show help message and exit.


## Examples

### Seed dimensions measurement module:
```
python src/seed_dimensions.py \
                -i res/ \
                -e jpeg
```

### Seed germination assessment module:
```
python src/seed_germination.py \
        -i res/ \
        -e JPG \
        --vec_plate_radius_or_height_limit "[500,800]" \
        --vec_plate_width_limit None \
        --central_round_plate True \
        --vec_RGB_mode_expected None \
        --shoot_area_limit="[100, 1000000]" \
        --vec_green_hue_limit="[60/360, 140/360]" \
        --vec_green_sat_limit="[0.25, 1.00]" \
        --vec_green_val_limit="[0.25, 1.00]" \
        --seed_area_limit="[100, 1000000]" \
        --vec_seed_hue_limit="[0/360, 45/360]" \
        --vec_seed_sat_limit="[0.30, 1.0]" \
        --vec_seed_val_limit="[0.00, 1.00]" \
        --shoot_axis_ratio_min_diff=0.5 \
        --seed_axis_ratio_min_diff=0.2 \
        --compute_areas True
```
