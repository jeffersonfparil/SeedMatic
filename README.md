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
- yield in grain crops for example.

## Usage
```
python src/seed_dimensions.py -h
python src/seed_germination.py -h
```

## Examples
```
echo "##########################################"
echo "Testing seed dimensions measurement module"
echo "##########################################"
time \
python src/seed_dimensions.py \
                -i res/ \
                -e jpeg \
                -W 20 \
                -H 20
echo "##########################################"
echo "Testing seed germination assessment module"
echo "##########################################"
time \
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
