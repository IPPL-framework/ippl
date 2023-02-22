set title "Disorder Induced Heating of cold plasma sphere - Mesh: 256-32"
set xlabel "time [ms]"
set datafile separator ","
set ylabel "x-Emittance"
set ylabel "x-Emittance"



# set terminal png size 20, 10
# set output 'test.png'

# set linetype 1 dashtype 1 
set for [i=3:9] linetype i dashtype 2 linewidth 2
# 1-5 possibledastypes
#1-9 possible linetpyes types to configure


set grid x,y
# set terminal wxt size 1100,1500
# set terminal wxt size 1500, 1000
set key top right


TOP= "set tmargin at screen 0.98;\
set bmargin at screen 0.68;\
unset xlabel;\
set format x '';\
set format y '%g'  "

MID = "set tmargin at screen 0.68;\
set bmargin at screen 0.38;\
set format y '%g'"

BOT= "set tmargin at screen 0.38;\
set bmargin at screen 0.08 ;\
set format x '%g' ;\
set format y '%g' ;\
set xlabel 'time [ms]' "

LEFT="set lmargin at screen 0.08; \
set rmargin at screen 0.53;"

RIGHT="set format y '' ;\
unset ylabel;\
set lmargin at screen 0.53; \
set rmargin at screen 0.98;"



do for [i=0:10000] {


set multiplot layout 2 ,1
set multiplot layout 3, 2
set title offset 0, -7

@TOP
@LEFT
set yrange [0:2]
set ylabel "emittance/(cm^2/ms)"    
set title "emittance vmax=6e4, no diffusion"
plot \
    "./data_v6_nc/FieldLangevin_1.csv"             using 2:5 with lines title                 "no collisions" ,\
    "./data_v6.e3_dr/FieldLangevin_1.csv"          using 2:5 with lines title                 "drag*=e-3" ,\
    "./data_v6.e4_dr/FieldLangevin_1.csv"          using 2:5 with lines title                 "drag*=e-4" ,\
    "./data_v6.e5_dr/FieldLangevin_1.csv"          using 2:5 with lines title                 "drag*=e-5" ,\
    "./data_v6.e6_dr/FieldLangevin_1.csv"          using 2:5 with lines title                 "drag*=e-6" ,\
    "./data_v6.e7_dr/FieldLangevin_1.csv"          using 2:5 with lines title                 "drag*=e-7" ,\
    

    # "./data_v6.e1_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff" ,\

set title offset 0,-3
set title "emittance vmax=6e4, no drag"
@RIGHT
plot \
    "./data_v6_nc/FieldLangevin_1.csv"             using 2:5 with lines title                 "no collisions" ,\
    "./data_v6.e2_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff*=e2" ,\
    "./data_v6.e3_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff*=e3" ,\
    "./data_v6.e4_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff*=e4" ,\
    "./data_v6.e5_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff*=e5" ,\
    "./data_v6.e6_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff*=e6" ,\
    "./data_v6.e7_di/FieldLangevin_1.csv"          using 2:5 with lines title                 "diff*=e7" ,\



@MID
@LEFT
set yrange [0:2.5e8]
set ylabel "temperature/((cm/ms)^23k_B)"
    
set title "temperature vmax=6e4, no diffusion "     
plot \
    "./data_v6_nc/FieldLangevin_1.csv"             using 2:3 with lines title                 "no collisions" ,\
    "./data_v6.e3_dr/FieldLangevin_1.csv"          using 2:3 with lines title                 "drag*e-3" ,\
    "./data_v6.e4_dr/FieldLangevin_1.csv"          using 2:3 with lines title                 "drag*e-4" ,\
    "./data_v6.e5_dr/FieldLangevin_1.csv"          using 2:3 with lines title                 "drag*e-5" ,\
    "./data_v6.e6_dr/FieldLangevin_1.csv"          using 2:3 with lines title                 "drag*e-6" ,\
    "./data_v6.e7_dr/FieldLangevin_1.csv"          using 2:3 with lines title                 "drag*e-7" ,\

    

    # "./data_v6.e1_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff" ,\

@RIGHT
set title "temperature vmax=6e4, no drag"    
plot \
    "./data_v6_nc/FieldLangevin_1.csv"             using 2:3 with lines title                 "no collisions" ,\
    "./data_v6.e2_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff*=e2" ,\
    "./data_v6.e3_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff*=e3" ,\
    "./data_v6.e4_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff*=e4" ,\
    "./data_v6.e5_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff*=e5" ,\
    "./data_v6.e6_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff*=e6" ,\
    "./data_v6.e7_di/FieldLangevin_1.csv"          using 2:3 with lines title                 "diff*=e7" ,\





set key bottom right
@BOT
@LEFT
set yrange [0.00075:0.0011]
set ylabel "rRMS/(cm)"
set title "rRMS vmax=6e4, no diffusion"
plot \
    "./data_v6_nc/All_FieldLangevin_1.csv"             using 38:29 with lines title                 "no collisions" ,\
    "./data_v6.e3_dr/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "drag*e-3" ,\
    "./data_v6.e4_dr/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "drag*e-4" ,\
    "./data_v6.e5_dr/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "drag*e-5" ,\
    "./data_v6.e6_dr/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "drag*e-6" ,\
    "./data_v6.e7_dr/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "drag*e-7" ,\

    
    
    # "./data_v6.e1_di/FieldLangevin_1.csv"          using 38:29 with lines title                 "diff" ,\

@RIGHT
set title "rRMS vmax=6e4, no drag"
plot \
    "./data_v6_nc/All_FieldLangevin_1.csv"             using 38:29 with lines title                "no collisions" ,\
    "./data_v6.e2_di/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "rRMS;diff*=e2" ,\
    "./data_v6.e3_di/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "rRMS;diff*=e3" ,\
    "./data_v6.e4_di/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "rRMS;diff*=e4" ,\
    "./data_v6.e5_di/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "rRMS;diff*=e5" ,\
    "./data_v6.e6_di/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "rRMS;diff*=e6" ,\
    "./data_v6.e7_di/All_FieldLangevin_1.csv"          using 38:29 with lines title                 "rRMS;diff*=e7" ,\
    # "./data_v6_nc/All_FieldLangevin_1.csv"             using 38:8 with lines title                 "no collisions, rmax" ,\
    # "./data_v6.e5_di/All_FieldLangevin_1.csv"          using 38:8 with lines title                  "rmaxS diff*e5" ,\
    # "./data_v6.e6_di/All_FieldLangevin_1.csv"          using 38:8 with lines title                  "rmaxS diff*e6" ,\
    # "./data_v6.e7_di/All_FieldLangevin_1.csv"          using 38:8 with lines title                  "rmaxS diff*e7" ,\

    
    
    

    unset multiplot
    pause 30   
}










# 1 n
# 2 vmax
# 5 vmin
# 8 rmax
# 11 rmin
# 14 vrms
# 17 T
# 20 eps
# 23 eps2
# 26 rvrms
# 29 rrms
# 32 rmean
# 35 vmean
# 38 time
# 39ex field eergy
# 40ex max norm
