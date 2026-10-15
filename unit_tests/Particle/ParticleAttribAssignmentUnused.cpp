#include "Ippl.h"

void assignParticleAttribFromUnusedTranslationUnit(
    ippl::ParticleAttrib<double>& scalar, ippl::ParticleAttrib<ippl::Vector<double, 3>>& vector) {
    scalar = 1.0;
    vector = ippl::Vector<double, 3>(1.0);
}
