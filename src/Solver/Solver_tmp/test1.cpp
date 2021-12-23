#include <iostream>

int main() {

#if __cplusplus > 201703L
            std::cout << "C++20" << std::endl;
    #else
            std::cout << "C++17" << std::endl;
    #endif

    return 0;
}
