#include "Ippl.h"

#include <cmath>
#include "gtest/gtest.h"

class FieldTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::Field<double, dim> field_type;

    FieldTest()
    : nPoints(100)
    {
        setup();
    }

    void setup() {
        Index I(nPoints);
        NDIndex<dim> owned(I, I, I);

        e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = SERIAL;

        FieldLayout<dim> layout(owned,allParallel, 1);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        ippl::UniformCartesian<double, dim> mesh(owned, hx, origin);

        field = std::make_unique<field_type>(mesh, layout);
    }

    std::unique_ptr<field_type> field;
    size_t nPoints;
};



TEST_F(FieldTest, FieldSum) {
    double val = 1.0;

    *field = val;

    double sum = field->sum();

    ASSERT_DOUBLE_EQ(val * std::pow(nPoints, dim), sum);
}




int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}