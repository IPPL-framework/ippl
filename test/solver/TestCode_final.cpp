#include <variant>
#include <Kokkos_Core.hpp>
#include <Ippl.h>

/////////////////////////////////////////////////////////

class Abstract {
    public:
        virtual ~Abstract() = default;

        // Pure virtual functions
        KOKKOS_FUNCTION virtual void function() const = 0;
};

/////////////////////////////////////////////////////////

class Concrete1 : public Abstract {
    public:
        KOKKOS_FUNCTION void function() const final {
            printf("Inside concrete 1 function\n");
        }
};

class Concrete2 : public Abstract {
    public:
        KOKKOS_FUNCTION void function() const final {
            printf("Inside concrete 2 function\n");
        }
};

/////////////////////////////////////////////////////////

class ClassA {
    public:
        const Abstract& ptr_concrete;

        ClassA(Abstract& x) : ptr_concrete(x) {}

        void execute(int N) {
            printf("Test: final keyword \n");

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
        
        ClassA classA1(x1);
        ClassA classA2(x2);
        
        classA1.execute(1);
        classA2.execute(1);
    }

    Kokkos::finalize();
    return 0;
}

