#!/bin/bash
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
python3 ${DIR_SRC}/main.py \
        -i res/ \
        -e JPG \
        --vec_plate_radius_or_height_limit "[800,1500]" \
        --vec_plate_width_limit None \
        --central_round_plate False \
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