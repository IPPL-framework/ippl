#include <variant>
#include <Kokkos_Core.hpp>
#include <Ippl.h>

/////////////////////////////////////////////////////////

template <class T>
class Abstract {
    public:
        virtual ~Abstract() = default;

        // Pure virtual functions
        KOKKOS_FUNCTION virtual void function() const = 0;
};

/////////////////////////////////////////////////////////

class Concrete1 : public Abstract<Concrete1> {
    public:
        KOKKOS_FUNCTION void function() const override {
            printf("Inside concrete 1 function\n");
        }
};

class Concrete2 : public Abstract<Concrete2> {
    public:
        KOKKOS_FUNCTION void function() const override {
            printf("Inside concrete 2 function\n");
        }
};

/////////////////////////////////////////////////////////

template <class T>
class ClassA {
    public:
        const Abstract<T>& ptr_concrete;

        ClassA(Abstract<T>& x) : ptr_concrete(x) {}

        void execute(int N) {
            printf("Test: CRTP \n");

            Kokkos::parallel_for("ClassA::execute", Kokkos::RangePolicy<>(0, N), KOKKOS_CLASS_LAMBDA(int i) {
                printf("before call to function\n");
                ptr_concrete.function();
                printf("after call to function\n");
           });
        }
 };

/////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Concrete1 x1;
        Concrete2 x2;
        
        ClassA<Concrete1> classA1(x1);
        ClassA<Concrete2> classA2(x2);
        
        classA1.execute(1);
        classA2.execute(1);
    }

    Kokkos::finalize();
    return 0;
}

