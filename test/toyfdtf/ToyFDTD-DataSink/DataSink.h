#ifndef DATASINK_H
#define DATASINK_H

#include "Ippl.h"

#define indent_l0 ""
#define indent_l1 "  "
#define indent_l2 "    "
#define indent_l3 "      "
#define indent_l4 "        "
#define indent_l5 "          "

class DataSink {

public:
    DataSink(UniformCartesian<3>& mesh, CenteredFieldLayout<3,UniformCartesian<3>,Cell>& field_layout, const int& guard_cell_size, 
             int &nx, int &ny, int &nz,
             double &dx, double& dy, double &dz,
             bool &binary, NDIndex<3> &lDom);
    ~DataSink();

    void dump(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD);
    void dumpASCII(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD);
    void dumpBinary(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD);

    void interpolateFieldValues1(const Field<Vektor<double,3>,3> &EFD);
    void interpolateFieldValues2(const Field<Vektor<double,3>,3> &HFD);

    void setIteration(const int &i);
    NDIndex<3> setNDIndex(const int& i, const int& j, const int& k);

private:
    int Nx_m;
    int Ny_m;
    int Nz_m;
    double dx_m;
    double dy_m;
    double dz_m;
    bool binary_m;
    int iteration_m;
    Field<Vektor<double,3>,3> NBV;
    std::vector<int> lDoms;
};

inline void DataSink::setIteration(const int &i) 
{
    iteration_m = i;
}

inline NDIndex<3> DataSink::setNDIndex(const int& i, const int& j, const int& k)
{
    return NDIndex<3>(Index(i,i), Index(j,j), Index(k,k));
}

#endif
