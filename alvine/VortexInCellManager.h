#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "SimulationParameters.hpp"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class VortexInCellManagerBase
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalanceStrategy<T, Dim>,
                              FieldSolver<T, Dim, FieldContainer<T, Dim>>> {
public:
    SimulationParameters<T, Dim> params;

    VortexInCellManagerBase(SimulationParameters<T, Dim> params_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalanceStrategy<T, Dim>,
                           FieldSolver<T, Dim, FieldContainer<T, Dim>>>()
        , params(params_) {}

    ~VortexInCellManagerBase() {}

public:
    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {}
};

template <typename T, unsigned Dim>
class VortexInCellManager : public VortexInCellManagerBase<T, Dim> {
public:
    VortexInCellManager(SimulationParameters<T, Dim> params_)
        : VortexInCellManagerBase<T, Dim>(params_) {}

    ~VortexInCellManager() {}
};

template <typename T>
class VortexInCellManager<T, 2> : public VortexInCellManagerBase<T, 2> {
public:
    VortexInCellManager(SimulationParameters<T, 2> params_)
        : VortexInCellManagerBase<T, 2>(params_) {
        this->setFieldContainer(
            std::make_shared<FieldContainer<T, 2>>(this->params.mesh, this->params.fl));
    }

    ~VortexInCellManager() {}

    void par2grid() override { std::cout << "2dim par to grid" << std::endl; }

    void grid2par() override {}

    void advance() override {}
};

template <typename T>
class VortexInCellManager<T, 3> : public VortexInCellManagerBase<T, 3> {
public:
    VortexInCellManager(SimulationParameters<T, 3> params_)
        : VortexInCellManagerBase<T, 3>(params_) {
        this->setFieldContainer(
            std::make_shared<FieldContainer<T, 3>>(this->params.mesh, this->params.fl));
    }

    ~VortexInCellManager() {}

    void par2grid() override { std::cout << "3dim par to grid" << std::endl; }

    void grid2par() override {}

    void advance() override {}
};

#endif
