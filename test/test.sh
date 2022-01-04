#!/bin/bash
echo "##################################"
echo "Test"
echo "##################################"
git clone https://github.com/jeffersonfparil/SeedMatic.git
cd SeedMatic/src
time python seed_dimensions.py \
                -i ../res \
                -e jpeg \
                -W 20 \
                -H 20

