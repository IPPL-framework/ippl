#include <variant>
#include <Kokkos_Core.hpp>
#include <Ippl.h>

class Abstract {
    public:
        virtual ~Abstract() = default;

        // Pure virtual functions
        KOKKOS_FUNCTION virtual void function() const = 0;
};

class Concrete1 : public Abstract {
    public:
        KOKKOS_FUNCTION void function() const override {
            printf("Inside concrete 1 function\n");
        }
};

class Concrete2 : public Abstract {
    public:
        KOKKOS_FUNCTION void function() const override {
            printf("Inside concrete 2 function\n");
        }
};

class ClassB {
    public:
        KOKKOS_FUNCTION void function() const {
            printf("Inside B function\n");
        }
};

using ippl::detail::ConditionalType, ippl::detail::VariantFromConditionalTypes;

/*template <unsigned Dim>
using Concrete1_t = ConditionalType<Dim == 1, Concrete1>;

template<unsigned Dim>
using Concrete2_t = ConditionalType<Dim == 2, Concrete2>;

template <unsigned Dim>
using Concrete_t = VariantFromConditionalTypes<Concrete1_t<Dim>, Concrete2_t<Dim>>;
*/

using Concrete_t = std::variant<Concrete1, Concrete2>;

class ClassA {
    public:
        const Concrete_t& ptr_concrete;
        const ClassB& ptr_b;

        ClassA(Concrete1& x, ClassB& b) : ptr_concrete(x), ptr_b(b) {}
        ClassA(Concrete2& x, ClassB& b) : ptr_concrete(x), ptr_b(b) {}

        KOKKOS_FUNCTION
        void function() const {
            printf("Inside A function\n");
        }

        void execute(int N) {
            printf("Just before kokkos loop\n");

            Kokkos::parallel_for("ClassA::execute", Kokkos::RangePolicy<>(0, N), KOKKOS_CLASS_LAMBDA(int i) {
                function();
                printf("after A function\n");
                ptr_b.function();
                printf("after B function\n");
                if (std::holds_alternative<Concrete1>(ptr_concrete)) {
                    std::get<Concrete1>(ptr_concrete).function();
                } else if (std::holds_alternative<Concrete2>(ptr_concrete)) {
                    std::get<Concrete2>(ptr_concrete).function();
                }

                //std::visit([](auto& x){ x.function(); }, ptr_concrete);
                printf("after concrete function\n");
           });
        }
 };

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Concrete1 x1;
        Concrete2 x2;
        ClassB b;
        
        ClassA classA1(x1, b);
        ClassA classA2(x2, b);
        
        classA1.execute(1);
        classA2.execute(1);
    }

    Kokkos::finalize();
    return 0;
}

