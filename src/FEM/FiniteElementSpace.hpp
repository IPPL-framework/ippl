
namespace ippl {
    template <typename T, unsigned Dim, typename QuadratureType>
    FiniteElementSpace<T, Dim>::FiniteElementSpace(const UniformCartesian<T, Dim>& mesh,
                                                   QuadratureType& quadrature, unsigned degree = 1)
        : degree_m(degree) {
        // This constructor creates a uniform cartesian finite element mesh with lines in 1D,
        // quadrilaterals in 2D or hexahedral elements in 3D
        switch (Dim) {
            case 1:
                throw std::invalid_argument("1D meshes are not implemented yet");
                // element_m = new LineElement<T>();
                break;
            case 2:
                throw std::invalid_argument("2D meshes are not implemented yet");
                // element_m = new QuadrilateralElement<T>();
                break;
            case 3:
                element_m = new HexahedralElement<T>();
                break;
            default:
                throw std::invalid_argument(
                    "Invalid dimension template argument 'Dim', only 1, 2, 3 are supported");
                break;
        }

        // TODO
    }
}  // namespace ippl