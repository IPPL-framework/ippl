#include <Kokkos_Core.hpp>
#include <Ippl.h>

/////////////////////////////////////////////////////////

class Element {
    public:
        KOKKOS_FUNCTION void function() const {
            printf("Element::function()\n");
        }
};

class FEMSpace {
    public:
        Element element;

        FEMSpace(Element& x) : element(x) {}
 };

class LagrangeSpace : public FEMSpace {
    public:
        LagrangeSpace(Element& x) : FEMSpace(x) {}

        template <typename F>
        void evaluateAx(F& eval)
        {
            printf("inside evaluateAx \n");

            Kokkos::parallel_for("ClassA::execute", Kokkos::RangePolicy<>(0, 3), KOKKOS_CLASS_LAMBDA(int i) {
                printf("inside kokkos loop \n");

                int res = eval(i);

                printf("loop idx %d, result = %d", i, res);

            });
        }
};

struct EvalFunctor {
    int val;

    EvalFunctor(int v) : val(v) {}

    KOKKOS_FUNCTION int operator()(size_t i) const {
        return val * i;
    }
};


class FEMPoissonSolver {
    public:
        FEMPoissonSolver(LagrangeSpace& space) : lagrange_space(space) {}
        ~FEMPoissonSolver() = default;

        void solve() {
            
            int val = 2;

            EvalFunctor eval(val);

            lagrange_space.evaluateAx(eval);
        }

    private:
        LagrangeSpace lagrange_space;
};

/////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Element elem;
        
        LagrangeSpace lagrange_space(elem);

        FEMPoissonSolver fem(lagrange_space);

        fem.solve();
    }

    Kokkos::finalize();
    return 0;
}
