#include "InputParser.h"


#ifdef USENAMESPACE
namespace initializer {
#endif

InputParser::InputParser(std::string filename)
{
    filename_m = filename;
    parseFile();
}


bool InputParser::getByName(std::string name, int &param)
{
    std::map<std::string,int>::iterator it;
    it = IntDict_m.find(name);
    if(it == IntDict_m.end())
        return false;
    else {
        param = (*it).second;
        return true;
    }
}


bool InputParser::getByName(std::string name, real &param)
{
    std::map<std::string,real>::iterator it;
    it = RealDict_m.find(name);
    if(it == RealDict_m.end())
        return false;
    else {
        param = (*it).second;
        return true;
    }
}


void InputParser::parseFile()
{
    std::ifstream fin;
    std::string line = "";

    fin.open(filename_m.c_str());
    while(std::getline(fin, line)) {
        if(line.find("//") == 0 || line.empty()) 
            continue;
        else {
            int pos = line.find("=");
            std::string name = line.substr(0,pos);
            std::string svalue = line.substr(pos+1);
            if(isInt(svalue)) {
                std::istringstream instr(svalue);
                int value;
                instr >> value;
                IntDict_m.insert( std::pair<std::string, int>(name, value));
            } else {
                std::istringstream instr(svalue);
                real value;
                instr >> value;
                RealDict_m.insert( std::pair<std::string, real>(name, value));
            }
        }
    }
}

#ifdef USENAMESPACE
}
#endif

#ifdef TEST_PARSER
int main(int argc, char**argv)
{
    initializer::InputParser *par = new initializer::InputParser("ExampleMCInputFile.in");
    real zin = 0.0;
    bool found = par->getByName("zin", zin);
    std::cout << "found= " << found << " value= " << zin << std::endl;
    return 0;
}
#endif
