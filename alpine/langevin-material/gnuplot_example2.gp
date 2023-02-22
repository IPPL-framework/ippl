# set title "  ANALYT(hess(G)) - hess(NUM(G)) \n  ANALYT(grad(H)) - grad(NUM(H)) \n \
# ignore one additinal border cell in error clalc \n  n=1 vth=1 vmax=5sigma"


# set title "  ANALYT(hess(G)) - hess(ANALYT(G)) \n ANALYT(grad(H)) - grad(ANALYT(H)) \n \
Ghost 0 \n n=1 vth=1 vmax=5sigma"
set xlabel "meshpoints per axis"
set datafile separator ","
set grid x, y
set xrange[1:260]
f2(x) = 1/(x*x)
f1(x) = 1/(x)
L(x) = log10(x)

set logscale x

 

set ylabel "-log_{10}(error)"
p  L(f2(x))  w l title "order 2", L(f1(x)) w l title "order 1",\
"1150-VICO-spectral.csv"        u 1:(log10($2)) w lp title "H L2-Error",\
"1150-VICO-spectral.csv"        u 1:(log10($7)) w lp title "F_j avg L2-Error"


# set logscale y
# set ylabel "error"
# p f2(x)    w l title "order 2", f1(x) w l title "order 1",\
# "D1150.csv"        u 1:2 w lp title "H L2-Error",\
# "D1150.csv"        u 1:7 w lp title "Hx  L2-Error"


