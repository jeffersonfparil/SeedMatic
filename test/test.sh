#!/bin/bash
echo "##################################"
echo "Test"
echo "##################################"
time python src/seed_dimensions.py \
                -i res/ \
                -e jpeg \
                -W 20 \
                -H 20
