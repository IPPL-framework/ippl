#include "FEM/Element.h"

namespace ippl {

template <typename T, unsigned Dim, Element<Dim> ElementType>
FEMMesh<T, Dim, ElementType>::FEMMesh() : Mesh<T, Dim>(), volume_m(0.0) {}

}  // namespace ippl