#!/bin/bash

tests=(
    "1000 2000 1000 1"
    "1000 2000 1000 2"
    "1000 2000 1000 4"
    "1000 2000 1000 8"
    "1000 2000 1000 16"

    "1000 1000 1000 1"
    "2000 1000 1000 2"
    "4000 1000 1000 4"
    "8000 1000 1000 8"
    "16000 1000 1000 16"
)

for test in "${tests[@]}"; do
    echo "Running a test with parameters: $test"
    ./matrix_prod $test
done
