


namespace ippl {
    namespace detail {
        template<typename T, unsigned Dim, class Mesh, class Cell>
        int BCondBase<T, Dim, Mesh, Cell>::allComponents = -9999;


        template<typename T, unsigned Dim, class Mesh, class Cell>
        BCondBase<T, Dim, Mesh, Cell>::BCondBase(unsigned int face, int i)
        : face_m(face), m_changePhysical(false)
        {
            // For only one specified component index (including the default case of
            // BCondBase::allComponents meaning apply to all components of T, just
            // assign the Component value for use in pointer offsets into
            // single-component-index types in applicative templates elsewhere:
            component_m = i;
        }
    }


    template<typename T, unsigned Dim, class Mesh, class Cell>
    void ConstantFace<T, Dim, Mesh, Cell>::write(std::ostream& out) const
    {
        out << "ConstantFace"
            << ", Face=" << this->face_m
            << ", Constant=" << this->offset_m;
    }
}