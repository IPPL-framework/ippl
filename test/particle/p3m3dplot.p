set terminal postscript
set output "plot.eps"
set yrange [0:0.05]
plot "field.txt" using 2:3  lc rgb "blue" title "P3M", (x<6)?x/(6*6*6):1/(x*x) title "analytical"

#splot "field.txt" using 4:5:(abs(1-$3*($2*$2))>0.25?$6:0/0)
