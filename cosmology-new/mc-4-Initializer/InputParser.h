#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include "TypesAndDefs.h"

#ifdef USENAMESPACE
namespace initializer {
#endif

class InputParser {

public:

    InputParser(std::string filename);
    bool getByName(std::string name, integer &param);
    bool getByName(std::string name, real &param);

private:

    std::map<std::string, real> RealDict_m;
    std::map<std::string, integer> IntDict_m;
    std::string filename_m;

    void parseFile();
    bool isInt(std::string str) {return str.find(".") == std::string::npos; }

};

#ifdef USENAMESPACE
}
#endif
 
#endif
