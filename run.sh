#!/bin/bash
for problem in \
    p00_scalar_add \
    p01_vector_add \
    p02_scale_shift \
    p03_relu_mul
do
    echo "===== $problem ====="
    python3.12 submit.py "$problem" --compiler compile.py
done