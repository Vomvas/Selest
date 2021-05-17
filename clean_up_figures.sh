#!/bin/bash

echo "Cleaning up benchmark results and figures..."

rm -rf Fig_*/*.pdf
rm -rf Fig_5/*/bench_outputs/*
rm -rf Table_*/table_*.txt
rm -rf Table_*/bench_outputs/*

echo "Figures and results deleted."
