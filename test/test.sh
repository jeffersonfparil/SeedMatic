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
if [ $(cat res/OUTPUT/At-seeds-Col_1-03-germination_data.csv | wc -l) -gt 1 ]
then
        echo "PASSED: seed dimension measurement module"
        RET1=0
else
        echo "FAILED: seed dimension measurement module"
        RET1=1
fi

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
if [ $(cat res/OUTPUT/At-germination-no-marker-1-germination_data.csv | wc -l) -gt 0 ]
then
        echo "PASSED: seed germination assessment module"
        RET2=0
else
        echo "FAILED: seed germination assessment module"
        RET2=1
fi

### error codes: 0 for success, 1 for one error, and 2 for 2 errors
return $(echo "$RET1 + $RET2" | bc)
