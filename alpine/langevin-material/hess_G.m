v = Sqrt[vx^2 + vy^2 + vz^2]
e1[vx_,vy_,vz_] = Exp[-v^2/(2*vth^2)]/Sqrt[Pi]
e2[vx_,vy_,vz_] = Erf[v/(Sqrt[2] * vth)]
e3[vx_,vy_,vz_] = (vth/(Sqrt[2]*v)) + (v/(Sqrt[2]*vth))
f[vx_,vy_,vz_] = e1[vx,vy,vz] + e2[vx,vy,vz] * (e3[vx,vy,vz])
// Simplify[D[f[vx,vy,vz], {vx,2}]]
Simplify[D[f[vx,vy,vz], vx, vx]]
Simplify[D[f[vx,vy,vz],vy,vx]]
CForm[Simplify[D[f[vx,vy,vz], vx, vx]]]
CForm[Simplify[D[f[vx,vy,vz],vy,vx]]]
// makes a difference if pi or Pi? -> can better simplify!!
