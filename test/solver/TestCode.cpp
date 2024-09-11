#include <Kokkos_Core.hpp>
#include <Ippl.h>

/////////////////////////////////////////////////////////

class Abstract {
    public:
        virtual ~Abstract() = default;

        /*
        KOKKOS_FUNCTION int call_function() const {
            printf("Abstract::call_function()\n");
            int result = this->function();
            return result;
        }

        KOKKOS_FUNCTION virtual int function() const = 0;
        */
};

/////////////////////////////////////////////////////////

class Concrete1 : public Abstract {
    public:
        KOKKOS_FUNCTION int call_function() const {
            printf("Concrete1::call_function()\n");
            int result = this->function();
            return result;
        }

        KOKKOS_FUNCTION int function() const {
            printf("Concrete1::function()\n");
            return 1;
        }
};

class Concrete2 : public Abstract {
    public:
        KOKKOS_FUNCTION int call_function() const {
            printf("Concrete2::call_function()\n");
            int result = this->function();
            return result;
        }

        KOKKOS_FUNCTION int function() const {
            printf("Concrete2::function()\n");
            return 2;
        }
};

/////////////////////////////////////////////////////////
template <typename T>
class ClassA {
    public:
        T concrete;

        ClassA(T& x) : concrete(x) {}

        virtual void execute(int N) = 0;
 };

template <typename T>
class ClassB : public ClassA<T> {
    public:
        ClassB(T& x) : ClassA<T>(x) {}

        void execute(int N) override
        {
            printf("Test: call function \n");

            Kokkos::parallel_for("ClassA::execute", Kokkos::RangePolicy<>(0, N), KOKKOS_CLASS_LAMBDA(int i) {
                printf("before call to function\n");
                int result = this->concrete.call_function();
                printf("after call to function, result = %d\n", result);
            });
        }
};

/////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Concrete1 x1;
        Concrete2 x2;
        
        ClassB<Concrete1> classB1(x1);
        ClassB<Concrete2> classB2(x2);
        
        classB1.execute(1);
        classB2.execute(1);
    }

    Kokkos::finalize();
    return 0;
}
