#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"



TEST(Solver, SetLinearBilinearForm) {
  constexpr unsigned Dim = 3;
  using Tlhs = double;
  using T = Tlhs;
  using ElementType = std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>, std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>, ippl::HexahedralElement<Tlhs>>>;
  using Mesh_t   = ippl::UniformCartesian<double, Dim>;
  using Field_t  = ippl::Field<double, Dim, Mesh_t, Cell>;
  using QuadratureType = ippl::GaussJacobiQuadrature<Tlhs, 5, ElementType>;
  using SpaceType = ippl::NedelecSpace<double, Dim, 1, ElementType, QuadratureType, Field_t, Field_t>;
  using MeshType = typename Field_t::Mesh_t;
  
  ElementType refElement_m;
  QuadratureType quadrature_m(refElement_m, 0.0, 0.0);

  
  const unsigned numNodesPerDim = 20;
  const unsigned numCellsPerDim = numNodesPerDim - 1;

  const T domain_start = 0.0;
  const T domain_end = 1.0;
  // Domain: [-1, 1]
  const ippl::Vector<unsigned, Dim> nodesPerDimVec(numNodesPerDim);
  ippl::NDIndex<Dim> domain(nodesPerDimVec);
  ippl::Vector<T, Dim> cellSpacing((domain_end - domain_start) / static_cast<T>(numCellsPerDim));
  ippl::Vector<T, Dim> origin(domain_start);
  Mesh_t mesh(domain, cellSpacing, origin);

  // specifies decomposition; here all dimensions are parallel
  std::array<bool, Dim> isParallel;
  isParallel.fill(true);

  ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel);
  
  
  SpaceType space(mesh, refElement_m, quadrature_m);
  ippl::FEMSolver<Field_t, Field_t, SpaceType, Mesh_t, QuadratureType> solver(space, mesh, layout, quadrature_m);

  solver.setBilinear([](size_t i, size_t j, ippl::Vector<double,3> x) {
    return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
  });


  solver.setLinear([](size_t i, size_t j, ippl::Vector<double,3> x) {
    return 2*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  });

}



int main(int argc, char* argv[]) {
  int success = 1;
  ippl::initialize(argc, argv);
  {
    ::testing::InitGoogleTest(&argc, argv);
    success = RUN_ALL_TESTS();
  }
  ippl::finalize();
  return success;
}