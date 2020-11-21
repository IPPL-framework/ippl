p3m3d 
-----

Usage:
	      interaction radius
                     /
          grid size /  particles    distribution
           / |  \  /    /             /
  ./p3m3d 16 16 16 5. 1000 [uniform|random|point] --commlib mpi --info 9 | tee field.txt
 

using the "point" distribution will only place one particle


P3m3d creates a x*y*z grid and calculates the field for either a point charge (when the "point" parameter is used)
or a spherical distribution of charges with at total charge of 1.