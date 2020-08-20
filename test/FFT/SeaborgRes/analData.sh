#!/bin/bash
#
# usage: ./analData.sh dataSet1 TestFFT
# 
# $1 is the filename for the results
# $2 is the base filename for the data 
#       
rm -rf $1
touch $1
echo "# Loops FFTTypw TotalTime Dim Processors NX=NY NZ ERROR" > $1
for file in `ls $2*.out`
    do
        grep Result $file | awk '{ print $2 " " $4$5$6 " " $10 " " $14 " " $16 " " $18 " " $22 " " $24}' >> $1
    done
