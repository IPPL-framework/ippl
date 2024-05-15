/**
@page hello_world HelloWorld

This is Hello World in IPPL.

@code

const char* TestName = "IPPL_Test";

#include "Ippl.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);

        msg << "Hello World" << endl;
    }
    ippl::finalize();

    return 0;
}
@endcode

The executable can be run with the following command-line arguments:
```
./HelloWorld --info 10
```
*/
