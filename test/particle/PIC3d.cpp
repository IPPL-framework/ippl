//
// Test PIC3d
//   This test program sets up a simple sine-wave electric field in 3D,
//   creates a population of particles with random q/m values (charge-to-mass
//   ratio) and velocities, and then tracks their motions in the static
//   electric field using cloud-in-cell interpolation.
//
//   Usage:
//     mpirun -np 2 PIC3d 128 128 128 10000 10 OOP --commlib mpi --info 0
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"
// #include <string>
// #include <fstream>
// #include <vector>
// #include <iostream>
// #include <set>

#include <random>

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::detail::ParticleLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef Cell                                       Center_t;
//typedef CenteredFieldLayout<Dim, Mesh_t, Center_t> FieldLayout_t;
typedef FieldLayout<Dim> FieldLayout_t;


template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Vector_t, Dim> VField_t;


//enum BC_t {OOO,OOP,PPP};

double pi = acos(-1.0);
const double dt = 1.0;          // size of timestep

/*
void dumpVTK(Field<Vektor<double,3>,3> &EFD, NDIndex<3> lDom, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "pic3d" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
    vtkout << "ORIGIN 0 0 0" << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "POINT_DATA " << nx*ny*nz << std::endl;

    vtkout << "VECTORS E-Field float" << std::endl;
    for (int z=lDom[2].first(); z<lDom[2].last(); z++) {
        for (int y=lDom[1].first(); y<lDom[1].last(); y++) {
            for (int x=lDom[0].first(); x<lDom[0].last(); x++) {
                Vektor<double, 3> tmp = EFD[x][y][z].get();
                vtkout << tmp(0) << "\t"
                       << tmp(1) << "\t"
                       << tmp(2) << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}


void dumpVTK(Field<double,3> &EFD, NDIndex<3> lDom, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    //SERIAL at the moment
    //if (Ippl::myNode() == 0) {

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "toyfdtd" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
    vtkout << "ORIGIN 0 0 0" << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "POINT_DATA " << nx*ny*nz << std::endl;

    vtkout << "SCALARS E-Field float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z=lDom[2].first(); z<=lDom[2].last(); z++) {
        for (int y=lDom[1].first(); y<=lDom[1].last(); y++) {
            for (int x=lDom[0].first(); x<=lDom[0].last(); x++) {
                vtkout << EFD[x][y][z].get() << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}
*/



template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    Field<Vector<double, Dim>, Dim> EFD_m;
    Field<double,Dim> EFDMag_m;


    Vector<int, Dim> nr_m;

    e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;


public:
    ParticleAttrib<double>     qm; // charge-to-mass ratio
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position
    typename ippl::ParticleBase<PLayout>::particle_position_type B;  // magnetic field at particle position

//     /*
//       In case we have OOP or PPP boundary conditions
//       we must define the domain, i.e can not be deduced from the
//       particles as in the OOO case.
//     */
//
    ChargedParticles(PLayout& pl,
//                      BC_t bc, InterPol_t interpol,
                     Vector_t hr, Vector_t rmin, Vector_t rmax, e_dim_tag decomp[Dim]
                     /*, bool gCells*/)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    {
//         // register the particle attributes
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(B);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    void setupBCs() {
//         if (bco_m == OOO)
//             setBCAllOpen();
//         else if (bco_m == PPP)
            setBCAllPeriodic();
//         else
//             setBCOOP();
    }

//     inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }
//
//     inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }
//
//     inline const FieldLayout_t& getFieldLayout() const {
//         return dynamic_cast<FieldLayout_t&>( this->getLayout().getLayout().getFieldLayout());
//     }
//
//     inline FieldLayout_t& getFieldLayout() {
//         return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
//     }
//
     void gatherCIC(/*int iteration*/) {

        gather(this->E, EFD_m, this->R);
 		scatterCIC();
        //NDIndex<Dim> lDom = getFieldLayout().getLocalNDIndex();
         //dumpVTK(EFDMag_m,lDom,nr_m[0],nr_m[1],nr_m[2],iteration,hr_m[0],hr_m[1],hr_m[2]);
     }

     //void scatterCIC(Field_t& field_temp) {
     void scatterCIC() {
         Inform m("scatter ");
         double initialQ = 1.0;//qm.sum();
         EFDMag_m = 0.0;
         scatter(qm, EFDMag_m, this->R);
         //scatter(qm, field_temp, this->R);
         double Q_grid = EFDMag_m.sum(1);
         //double Q_grid = field_temp.sum(1);
         m << "Q grid = " << Q_grid << endl;
         m << "Error = " << initialQ-Q_grid << endl;
     }
//
//     void myUpdate() {
//
//         double hz   = hr_m[2];
//         double zmin = rmin_m[2];
//         double zmax = rmax_m[2];
//
//         if (bco_m != PPP) {
//             bounds(this->R, rmin_m, rmax_m);
//
//             NDIndex<Dim> domain = this->getFieldLayout().getDomain();
//
//             for (unsigned int i=0; i<Dim; i++)
//                 nr_m[i] = domain[i].length();
//
//             for (unsigned int i=0; i<Dim; i++)
//                 hr_m[i] = (rmax_m[i] - rmin_m[i]) / (nr_m[i] - 1.0);
//
//             if (bco_m == OOP) {
//                 rmin_m[2] = zmin;
//                 rmax_m[2] = zmax;
//                 hr_m[2] = hz;
//             }
//
//             getMesh().set_meshSpacing(&(hr_m[0]));
//             getMesh().set_origin(rmin_m);
//
//             if(withGuardCells_m) {
//                 EFD_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);
//                 EFDMag_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
//             }
//             else {
//                 EFD_m.initialize(getMesh(), getFieldLayout(), vbc_m);
//                 EFDMag_m.initialize(getMesh(), getFieldLayout(), bc_m);
//             }
//         }
//         else {
//             if(fieldNotInitialized_m) {
//                 fieldNotInitialized_m=false;
//                 getMesh().set_meshSpacing(&(hr_m[0]));
//                 getMesh().set_origin(rmin_m);
//                 if(withGuardCells_m) {
//                     EFD_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);
//                     EFDMag_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
//                 }
//                 else {
//                     EFD_m.initialize(getMesh(), getFieldLayout(), vbc_m);
//                     EFDMag_m.initialize(getMesh(), getFieldLayout(), bc_m);
//                 }
//             }
//         }
//         this->update();
//     }
//
     
     void initFields() {
         Inform m("initFields ");

         NDIndex<Dim> domain = EFD_m.getDomain();

         for (unsigned int i=0; i<Dim; i++)
             nr_m[i] = domain[i].length();

         int nx = nr_m[0];
         int ny = nr_m[1];
         int nz = nr_m[2];

         double phi0 = 0.1*nx;
         double pi = acos(-1.0);

         m << "rmin= " << rmin_m << " rmax= " << rmax_m << " h= " << hr_m << " n= " << nr_m << endl;


         typename VField_t::LField_t::view_type& view = EFD_m(0).getView();

         Kokkos::parallel_for("Assign EFD_m[0]", 
                              Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1),
                                                                      view.extent(2)}),
                              KOKKOS_LAMBDA(const int i, const int j, const int k){

                                view(i, j, k)[0] = -2.0*pi*phi0/nx * 
                                                    cos(2.0*pi*(i+0.5)/nx) *
                                                    cos(4.0*pi*(j+0.5)/ny) * cos(pi*(k+0.5)/nz);
                              
                              });
         
         Kokkos::parallel_for("Assign EFD_m[1]", 
                              Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1),
                                                                      view.extent(2)}),
                              KOKKOS_LAMBDA(const int i, const int j, const int k){

                                view(i, j, k)[1] = 4.0*pi*phi0/ny * 
                                                   sin(2.0*pi*(i+0.5)/nx) * sin(4.0*pi*(j+0.5)/ny);
                              
                              });
         
         Kokkos::parallel_for("Assign EFD_m[2]", 
                              Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1),
                                                                      view.extent(2)}),
                              KOKKOS_LAMBDA(const int i, const int j, const int k){

                                view(i, j, k)[2] = 4.0*pi*phi0/ny * 
                                                   sin(2.0*pi*(i+0.5)/nx) * sin(4.0*pi*(j+0.5)/ny);
                              
                              });

         EFDMag_m = dot(EFD_m, EFD_m);
     }

     Vector_t getRMin() { return rmin_m;}
     Vector_t getRMax() { return rmax_m;}
     Vector_t getHr() { return hr_m;}

//     void savePhaseSpace(std::string fn, int idx) {
//
//         int tag = Ippl::Comm->next_tag(IPPL_APP_TAG4, IPPL_APP_CYCLE);
// 	std::vector<double> tmp;
//         tmp.clear();
//
//         // every node ckecks if he has to dump particles
//         for (unsigned i=0; i<this->getLocalNum(); i++) {
//             tmp.push_back(this->ID[i]);
//             tmp.push_back(this->R[i](0));
//             tmp.push_back(this->R[i](1));
//             tmp.push_back(this->R[i](2));
//             tmp.push_back(this->P[i](0));
//             tmp.push_back(this->P[i](1));
//             tmp.push_back(this->P[i](2));
//         }
//
//         if(Ippl::myNode() == 0) {
//  	    std::ofstream ostr;
//             std::string Fn;
//             char numbuf[6];
//             sprintf(numbuf, "%05d", idx);
//             Fn = fn  + std::string(numbuf) + ".dat";
//             ostr.open(Fn.c_str(), std::ios::out );
//             ostr.precision(15);
//             ostr.setf(std::ios::scientific, std::ios::floatfield);
//
//             ostr << " x, px, y, py t, pt, id, node" << std::endl;
//
//             unsigned int dataBlocks=0;
//             double x,y,z,px,py,pz,id;
//             unsigned  vn;
//
//             for (unsigned i=0; i < tmp.size(); i+=7)
//                 ostr << tmp[i+1] << " " << tmp[i+4] << " " << tmp[i+2]  << " "
//                      << tmp[i+5] << " " << tmp[i+3] << " " << tmp[i+6]  << " "
//                      << tmp[i]   << " " << Ippl::myNode() << std::endl;
//
//             int notReceived =  Ippl::getNodes() - 1;
//             while (notReceived > 0) {
//                 int node = COMM_ANY_NODE;
//                 Message* rmsg =  Ippl::Comm->receive_block(node, tag);
//                 if (rmsg == 0)
//                     ERRORMSG("Could not receive from client nodes in main." << endl);
//                 notReceived--;
//                 rmsg->get(&dataBlocks);
//                 rmsg->get(&vn);
//                 for (unsigned i=0; i < dataBlocks; i+=7) {
//                     rmsg->get(&id);
//                     rmsg->get(&x);
//                     rmsg->get(&y);
//                     rmsg->get(&z);
//                     rmsg->get(&px);
//                     rmsg->get(&py);
//                     rmsg->get(&pz);
//                     ostr << x  << "\t " << px  << "\t " << y  << "\t "
//                          << py << "\t " << z << "\t " << pz << "\t "
//                          << id   << "\t " << vn << std::endl;
//                 }
//                 delete rmsg;
//             }
//             ostr.close();
//         }
//         else {
//             unsigned dataBlocks = 0;
//             Message* smsg = new Message();
//             dataBlocks = tmp.size();
//             smsg->put(dataBlocks);
//             smsg->put(Ippl::myNode());
//             for (unsigned i=0; i < tmp.size(); i++)
//                 smsg->put(tmp[i]);
//             bool res = Ippl::Comm->send(smsg, 0, tag);
//             if (! res)
//                 ERRORMSG("Ippl::Comm->send(smsg, 0, tag) failed " << endl;);
//         }
//     }
//
private:
//
//     inline void setBCAllOpen() {
//         for (unsigned i=0; i < 2*Dim; i++) {
//             this->getBConds()[i] = ParticleNoBCond;
//             bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
//             vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
//         }
//     }
//
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }
//
//     inline void setBCOOP() {
//         for (unsigned i=0; i < 2*Dim - 2; i++) {
//             bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
//             vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
//             this->getBConds()[i] = ParticleNoBCond;
//         }
//         for (unsigned i= 2*Dim - 2; i < 2*Dim; i++) {
//             bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
//             vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
//             this->getBConds()[i] = ParticlePeriodicBCond;
//         }
//     }
//

};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    if (argc != 6) {
        msg << "PIC3d [mx] [mx] [my] [#particles] [#time steps]"
            << endl;
        return -1;
    }

    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    const unsigned int totalP = std::atoi(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    
    msg << "Particle test PIC3d "
        << endl
        << "nt " << nt << " Np= "
        << totalP << " grid = " << nr
        << endl;
//     BC_t myBC;
//     if (std::string(argv[7])==std::string("OOO")) {
//         myBC = OOO; // open boundary
//         msg << "BC == OOO" << endl;
//     }
//     else if (std::string(argv[7])==std::string("OOP")) {
//         myBC = OOP; // open boundary in x and y, periodic in z
//         msg << "BC == OOP" << endl;
//     }
//     else {
//         myBC = PPP; // all periodic
//         msg << "BC == PPP" << endl;
//     }
//

    std::unique_ptr<Mesh_t> mesh;
    std::unique_ptr<FieldLayout_t> FL;

    using bunch_type = ChargedParticles<PLayout_t>;

    std::unique_ptr<bunch_type>  P;
    std::unique_ptr<PLayout_t> PL;

    NDIndex<Dim> domain;
    
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = Index(nr[i]);
    }
    
    e_dim_tag decomp[Dim];    
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = SERIAL;
    }

    // create mesh and layout objects for this problem domain
    double dx = 1.0 / double(nr[0]);
    double dy = 1.0 / double(nr[1]);
    double dz = 1.0 / double(nr[2]);
    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {0, 0, 0};
    mesh = std::make_unique<Mesh_t>(domain, hr, origin);
    //FL   = std::make_unique<FieldLayout_t>(*mesh, decomp);
    FL   = std::make_unique<FieldLayout_t>(domain, decomp, 1);
    PL   = std::make_unique<PLayout_t>(); //(*FL, *mesh);

    //ippl::UniformCartesian<double, 3> mesh_temp(domain, hr, origin);
    //FieldLayout<3> layout(domain,decomp, 1);
    //Field_t field_temp;
    //field_temp.initialize(mesh_temp, layout);

    /*
     * In case of periodic BC's define
     * the domain with hr and rmin
     */
    //Vector_t hr(1.0);
    Vector_t rmin(0.0);
    Vector_t rmax(1.0);

    P = std::make_unique<bunch_type>(*PL,/*myBC,*/hr,rmin,rmax,decomp);

    // initialize the particle object: do all initialization on one node,
    // and distribute to others

    unsigned long int nloc = totalP / Ippl::getNodes();

    P->create(nloc);

    std::mt19937_64 eng;//(42);
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();
    typename ParticleAttrib<double>::HostMirror Q_host = P->qm.getHostMirror();

    double q = 1.0/totalP;

    for (unsigned long int i = 0; i< nloc; i++) {
        for (int d = 0; d<3; d++) {
            R_host(i)[d] =  unif(eng); //* nr[d];
        }
        //Vector_t r = {unif(eng), unif(eng), unif(eng)};
        //R_host(i) = r;
        Q_host(i) = q;
    }

    Kokkos::deep_copy(P->R.getView(), R_host);
    Kokkos::deep_copy(P->qm.getView(), Q_host);
    P->P = 0.0;

    ippl::PRegion<double> region0(0.0, 1.0);
    ippl::PRegion<double> region1(0.0, 1.0);
    ippl::PRegion<double> region2(0.0, 1.0);

    ippl::NDRegion<double, Dim>  pr;
    pr = ippl::NDRegion<double, Dim>(region0, region1, region2);

    msg << "particles created and initial conditions assigned " << endl;
    P->EFD_m.initialize(*mesh, *FL);
    P->EFDMag_m.initialize(*mesh, *FL);
//
//     // redistribute particles based on spatial layout
//     P->myUpdate();
//
//     msg << "initial update and initial mesh done .... Q= " << sum(P->qm) << endl;
//     msg << P->getMesh() << endl;
//     msg << P->getFieldLayout() << endl;
//
    msg << "scatter test" << endl;
    //P->scatterCIC(field_temp);
    P->scatterCIC();
    
    P->initFields();
    msg << "P->initField() done " << endl;
     // begin main timestep loop
     msg << "Starting iterations ..." << endl;
     for (unsigned int it=0; it<nt; it++) {
         //P->gatherStatistics();
         // advance the particle positions
         // basic leapfrogging timestep scheme.  velocities are offset
         // by half a timestep from the positions.
         P->R = P->R + dt * P->P;

         //Apply particle BCs
         P->getLayout().applyBC(P->R, pr);

         // update particle distribution across processors
         //P->myUpdate();

         // gather the local value of the E field
         P->gatherCIC(/*it*/);

         // advance the particle velocities
         P->P = P->P + dt * P->qm * P->E;
         msg << "Finished iteration " << it << " - min/max r and h " << P->getRMin()
             << P->getRMax() << P->getHr() << endl;
     }
//     Ippl::Comm->barrier();
    msg << "Particle test PIC3d: End." << endl;

    return 0;
}
