v = Sqrt[vx^2 + vy^2 + vz^2]
f[vx_,vy_,vz_] = n/v*Erf[v/(Sqrt[2] * vth)]
Simplify[D[f[vx,vy,vz],vx]]
CForm[Simplify[D[f[vx,vy,vz],vx]]]
