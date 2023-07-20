#ifndef CONNECTOR_H
#define CONNECTOR_H

#include "PContainer/PContainer.hpp"

namespace Connector {

  enum CONNECTION_TYPE {VTK, HDF5, CSV, INSITU};

  
template <typename T, unsigned Dim = 3>
class Connector {

  const char* testName_m;
  const CONNECTION_TYPE connType_m;
  
public:
  
  Connector(const char* TestName, const CONNECTION_TYPE connType) :
    testName_m(TestName),
    connType_m(connType)
  {

  }
    ~Connector() {

  }

  virtual int open(std::string fn) = 0;

  virtual int close() = 0;

  virtual int status() = 0;

  virtual int write() = 0;

  virtual int read() = 0;

  const char* getName() { return testName_m;}

  const char* getConnTypeName() {

    switch (connType_m)
      {
      case CONNECTION_TYPE::VTK : return "VTK";
      case CONNECTION_TYPE::HDF5 : return "HDF5 not yet avaidable";
      case CONNECTION_TYPE::CSV : return "CSV";
      case CONNECTION_TYPE::INSITU : return "INSITY not yet avaidable";
      };

    /* soon we can do this :)
       #include <reflexpr>   (not yet in C++20
       #include <iostream>
       return std::meta::get_base_name_v<
          std::meta::get_element_m<
          std::meta::get_enumerators_m<reflexpr(connType_m)>,
          0>
       >;
    */
  }
  
};


template <typename T, unsigned Dim = 3>
class VTKConnector : public Connector<T, Dim> {

public:
  VTKConnector(const char* TestName) :
    Connector<T, Dim>(TestName, VTK)
  {

  }

   ~VTKConnector() {

  }

  void dumpVTK(VField_t<T, 3>& E, int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {

    typename VField_t<T, 3>::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << Connector<T, Dim>::testName_m << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                       << host_view(x, y, z)[2] << endl;
            }
        }
    }
  }

  void dumpVTK(Field_t<3>& rho, int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {
    
    typename Field_t<3>::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";
    
    Kokkos::deep_copy(host_view, rho.getView());
    
    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << Connector<T, Dim>::testName_m << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;
    
    vtkout << "SCALARS Rho float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = 0; z < nz + 2; z++) {
      for (int y = 0; y < ny + 2; y++) {
	for (int x = 0; x < nx + 2; x++) {
	  vtkout << host_view(x, y, z) << endl;
	}
      }
    }
  }
};

template <typename T, unsigned Dim = 3>
class StatisticsConnector : public Connector<T, Dim> {

  size_type totalNumParts_m;

public:
  StatisticsConnector(const char* TestName, size_type totalNumParts) :
    Connector<T, Dim>(TestName,CSV),
    totalNumParts_m(totalNumParts)
  {

  }

   ~StatisticsConnector() {

  }

  int open(std::string fn) { return 0; } 

  int close() { return 0; } 

  int status() { return 0; } 

  int write() { return 0; } 

  int read() { return 0; } 


  void gatherFieldStatistics (auto vel, auto rhs, auto vField, Vector_t<T,Dim> hr, const size_type localNum, double time) {
    auto Vview = vel.getView();

    double kinEnergy = 0.0;
    double potEnergy = 0.0;

    potEnergy = 0.5 * hr[0] * hr[1] * hr[2] * rhs.sum();
	
    Kokkos::parallel_reduce(
			    "Particle Kinetic Energy", localNum,
			    KOKKOS_LAMBDA(const int i, double& valL) {
			      double myVal = dot(Vview(i), Vview(i)).apply();
			      valL += myVal;
			    },
			    Kokkos::Sum<double>(kinEnergy));

    kinEnergy *= 0.5;
    double gkinEnergy = 0.0;

    MPI_Reduce(&kinEnergy, &gkinEnergy, 1, MPI_DOUBLE, MPI_SUM, 0,
	       ippl::Comm->getCommunicator());

    const int nghostE = vField.getNghost();
    auto Eview        = vField.getView();
    Vector_t<T, Dim> normE;

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    for (unsigned d = 0; d < Dim; ++d) {
      T temp = 0.0;
      ippl::parallel_reduce(
			    "Vector E reduce", ippl::getRangePolicy(Eview, nghostE),
			    KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
			      // ippl::apply accesses the view at the given indices and obtains a
			      // reference; see src/Expression/IpplOperations.h
			      T myVal = std::pow(ippl::apply(Eview, args)[d], 2);
			      valL += myVal;
			    },
			    Kokkos::Sum<T>(temp));
      T globaltemp          = 0.0;
      MPI_Datatype mpi_type = get_mpi_datatype<T>(temp);
      MPI_Reduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, 0, ippl::Comm->getCommunicator());
      normE[d] = std::sqrt(globaltemp);
    }

    if (ippl::Comm->rank() == 0) {
      std::stringstream fname;
      fname << "data/ParticleField_";
      fname << ippl::Comm->size();
      fname << ".csv";
      
      Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
      csvout.precision(10);
      csvout.setf(std::ios::scientific, std::ios::floatfield);
      
      if (time == 0.0) {
	csvout << "time, Potential energy, Kinetic energy, Total energy, Rho_norm2, "
	  "Ex_norm2, Ey_norm2, Ez_norm2";
	for (unsigned d = 0; d < Dim; d++) {
	  csvout << "E" << d << "norm2, ";
	}

	csvout << endl;
      }
      double rhoNorm_m = 0.0; /// FixMe
      csvout << time << " " << potEnergy << " " << gkinEnergy << " "
	     << potEnergy + gkinEnergy << " " << rhoNorm_m << " ";
      for (unsigned d = 0; d < Dim; d++) {
	csvout << normE[d] << " ";
      }
      csvout << endl;
    }
    
    ippl::Comm->barrier();
  }

  
  void gatherLocalDomainStatistics(const FieldLayout_t<Dim>& fl, const unsigned int step) {
    if (ippl::Comm->rank() == 0) {
      const typename FieldLayout_t<Dim>::host_mirror_type domains = fl.getHostLocalDomains();
      std::ofstream myfile;
      myfile.open("data/domains" + std::to_string(step) + ".txt");
      for (unsigned int i = 0; i < domains.size(); ++i) {
	for (unsigned d = 0; d < Dim; d++) {
	  myfile << domains[i][d].first() << " " << domains[i][d].last() << " ";
	}
	myfile << "\n";
      }
      myfile.close();
    }
    ippl::Comm->barrier();
  }

  
  void gatherLoadBalancingStatistics(size_type localNum, double time) {

    std::vector<double> imb(ippl::Comm->size());
    double equalPart = (double)totalNumParts_m / ippl::Comm->size();
    double dev       = (std::abs((double)localNum - equalPart) / totalNumParts_m) * 100.0;
    MPI_Gather(&dev, 1, MPI_DOUBLE, imb.data(), 1, MPI_DOUBLE, 0,
	       ippl::Comm->getCommunicator());

    if (ippl::Comm->rank() == 0) {
      std::stringstream fname;
      fname << "data/LoadBalance_";
      fname << ippl::Comm->size();
      fname << ".csv";

      Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
      csvout.precision(5);
      csvout.setf(std::ios::scientific, std::ios::floatfield);

      if (time == 0.0) {
	csvout << "time, rank, imbalance percentage" << endl;
      }

      for (int r = 0; r < ippl::Comm->size(); ++r) {
	csvout << time << " " << r << " " << imb[r] << endl;
      }
    }

    ippl::Comm->barrier();
  }
  /*

    Application specific routines, could go into seperate classes
    but for now I keep them here

   */

  void dumpLandau(auto Vfield, Vector_t<double,Dim> hr, double time) {
    
    const int nghostE = Vfield.getNghost();
    auto vFieldView   = Vfield.getView();
    
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    double localEx2 = 0, localExNorm = 0;
    ippl::parallel_reduce(
			  "Ex stats", ippl::getRangePolicy(vFieldView, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(vFieldView, args)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;

                double norm = Kokkos::fabs(ippl::apply(vFieldView, args)[0]);
                if (norm > ENorm) {
                    ENorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

    double globaltemp = 0.0;
    MPI_Reduce(&localEx2, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0,
	       ippl::Comm->getCommunicator());
    double fieldEnergy =
      std::reduce(hr.begin(), hr.end(), globaltemp, std::multiplies<double>());

    double ExAmp = 0.0;
    MPI_Reduce(&localExNorm, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());

    if (ippl::Comm->rank() == 0) {
      std::stringstream fname;
      fname << "data/FieldLandau_";
      fname << ippl::Comm->size();
      fname << ".csv";

      Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
      csvout.precision(10);
      csvout.setf(std::ios::scientific, std::ios::floatfield);

      if (time == 0.0) {
	csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
      }
      
      csvout << time << " " << fieldEnergy << " " << ExAmp << endl;
    }
    
    ippl::Comm->barrier();
  }

  void dumpBumponTail(auto Vfield, Vector_t<double,Dim> hr, double time) {
    
    const int nghostE = Vfield.getNghost();
    auto Eview        = Vfield.getView();
    double fieldEnergy, EzAmp;

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    double temp            = 0.0;
    ippl::parallel_reduce(
			  "Ex inner product", ippl::getRangePolicy(Eview, nghostE),
			  KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
			    // ippl::apply accesses the view at the given indices and obtains a
			    // reference; see src/Expression/IpplOperations.h
			    double myVal = std::pow(ippl::apply(Eview, args)[Dim - 1], 2);
			    valL += myVal;
			  },
			  Kokkos::Sum<double>(temp));
    double globaltemp = 0.0;
    MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
    fieldEnergy = std::reduce(hr.begin(), hr.end(), globaltemp, std::multiplies<double>());

    double tempMax = 0.0;
    ippl::parallel_reduce(
			  "Ex max norm", ippl::getRangePolicy(Eview, nghostE),
			  KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
			    // ippl::apply accesses the view at the given indices and obtains a
			    // reference; see src/Expression/IpplOperations.h
			    double myVal = std::fabs(ippl::apply(Eview, args)[Dim - 1]);
			    if (myVal > valL) {
			      valL = myVal;
			    }
			  },
			  Kokkos::Max<double>(tempMax));
    EzAmp = 0.0;
    MPI_Reduce(&tempMax, &EzAmp, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());

    if (ippl::Comm->rank() == 0) {
      std::stringstream fname;
      fname << "data/FieldBumponTail_";
      fname << ippl::Comm->size();
      fname << ".csv";

      Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
      csvout.precision(10);
      csvout.setf(std::ios::scientific, std::ios::floatfield);
      
      if (time == 0.0) {
	csvout << "time, Ez_field_energy, Ez_max_norm" << endl;
      }

      csvout << time << " " << fieldEnergy << " " << EzAmp << endl;
    }

    ippl::Comm->barrier();
  }
};

template <typename T, unsigned Dim = 3>
class PhaseSpaceConnector : public Connector<T, Dim> {

  size_type totalNumParts_m;

public:
  PhaseSpaceConnector(const char* TestName, size_type totalNumParts) :
    Connector<T, Dim>(TestName,CSV),
    totalNumParts_m(totalNumParts)
  {

  }

   ~PhaseSpaceConnector() {

  }

  int open(std::string fn) { return 0; } 

  int close() { return 0; } 

  int status() { return 0; } 

  int write() { return 0; } 

  int read() { return 0; } 

  void dumpParticleData(auto RHost, auto VHost, size_type localNum) {
    
    std::stringstream pname;
    pname << "data/ParticleIC_";
    pname << ippl::Comm->rank();
    pname << ".csv";
    Inform pcsvout(NULL, pname.str().c_str(), Inform::OVERWRITE, ippl::Comm->rank());
    pcsvout.precision(10);
    pcsvout.setf(std::ios::scientific, std::ios::floatfield);
    pcsvout << "R_x, R_y, R_z, V_x, V_y, V_z" << endl;
    for (size_type i = 0; i < localNum; i++) {
      for (unsigned d = 0; d < Dim; d++) {
	pcsvout << RHost(i)[d] << " ";
      }
      for (unsigned d = 0; d < Dim; d++) {
	pcsvout << VHost(i)[d] << " ";
      }
      pcsvout << endl;
    }
    ippl::Comm->barrier();
 }
  
  
};



  

} // namespace
#endif
