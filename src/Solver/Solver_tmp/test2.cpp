#include <iostream>
#include <type_traits>

enum Interpl_t {
    NGP_t,
    CIC_t
};

template <unsigned Dim, class M, class C>
class Field {
public:
    Field() = default;
};

template <typename T, typename U, class... Args>
class ParticleAttrib {
public:

    ParticleAttrib() : t_m(2), u_m(3)
    {}

    template <unsigned Dim, class M, class C>
    void scatterNGP(Field<Dim, M, C>&) {
        std::cout << "NGP" << std::endl;
    }

    template <unsigned Dim, class M, class C>
    void scatterCIC(Field<Dim, M, C>&) {
        std::cout << "CIC" << std::endl;
    }

private:
    T t_m;
    U u_m;
};

template <Interpl_t I, unsigned Dim, class M, class C,
          typename T, typename U, class... Args,
          std::enable_if_t<I == NGP_t, bool> = true>
void scatter(ParticleAttrib<T, U, Args...>& t, Field<Dim, M, C>& f) {
    t.scatterNGP(f);
}

template <Interpl_t I, unsigned Dim, class M, class C,
          typename T, typename U, class... Args,
          std::enable_if_t<I == CIC_t, bool> = true>
void scatter(ParticleAttrib<T, U, Args...>& t, Field<Dim, M, C>& f) {
    t.scatterCIC(f);
}

template <unsigned Dim, class M, class C,
          typename T, typename U, class... Args>
void scatter(ParticleAttrib<T, U, Args...>& t, Field<Dim, M, C>& f) {
    scatter<CIC_t>(t, f);
}

int main() {

    ParticleAttrib<double, int> t;
    Field<3, double, double> f;

    scatter<NGP_t>(t, f);

    scatter<CIC_t>(t, f);

    scatter(t, f);

    return 0;
}
