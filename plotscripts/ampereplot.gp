#! /usr/bin/gnuplot
set terminal svg size 800,600 font "CMU Serif,22"
set logscale x 2
set logscale y 2
set key box
set output "ampere.svg"
set ylabel "Magnetic Flux density[T]"
set xlabel "Distance to wire [m]"
#set title "Ampere's Law"
set grid lw 2
set ytics format "2^{%L}"
set xtics format "2^{%L}"
plot 'ampere_line.txt' u (abs($1)):(abs($2)) w points title "Measurements", 1.256e-6/(2*3.14159)*(1/x) w lines lw 2 title "Radial falloff"
