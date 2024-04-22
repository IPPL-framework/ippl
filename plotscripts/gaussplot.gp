#! /usr/bin/gnuplot
set terminal svg size 900,700 font "CMU Serif, 25"
set logscale x 2
set logscale y 2
set key box
set output "gauss.svg"
set ylabel "Electric Field [V/m]"
set xlabel "Distance to charge [m]"
#set title "Gauss's Law"
set grid lw 2
set ytics format "2^{%L}"
set xtics format "2^{%L}"
plot 'gauss_line.txt' u (abs($1)):(abs($2)) w points title "Measurements" pt 2, (1 / (4 * 3.1416 * 8.854e-12 * x * x)) w lines lw 3 title "Quadratic falloff"
