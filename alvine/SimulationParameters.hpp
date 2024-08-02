#ifndef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

template <typename T, unsigned Dim>
struct SimulationParameters {
    SimulationParameters(unsigned nt_, Vector_t<int, Dim>& nr_, std::string& solver_, double lbt_, double visc_,
                         Vector_t<T, Dim> rmin_ = 0.0, Vector_t<T, Dim> rmax_ = 10.0,
                         Vector_t<T, Dim> origin_ = 0.0)
        : nt(nt_)
        , nr(nr_)
        , solver(solver_)
        , lbt(lbt_)
        , visc(visc_)
        , rmin(rmin_)
        , rmax(rmax_)
        , origin(origin_)
        , np(0) {
        this->dr = this->rmax - this->rmin;

        this->hr = this->dr / this->nr;

        this->dt = std::min(0.05, 0.5 * (*std::min_element(this->hr.begin(), this->hr.end())));

        this->time = 0.0;

        this->it = 0;

        for (unsigned i = 0; i < Dim; i++) {
            this->domain[i] = ippl::Index(this->nr[i]);
        }

        this->decomp.fill(true);
    }

    int nt;
    Vector_t<int, Dim> nr;
    std::string solver;
    double lbt;
    double visc;

    double dt;
    double time;
    int it;
    Vector_t<T, Dim> hr;
    Vector_t<double, Dim> dr;
    ippl::NDIndex<Dim> domain;
    std::array<bool, Dim> decomp;

    Vector_t<T, Dim> rmin;
    Vector_t<T, Dim> rmax;
    Vector_t<T, Dim> origin;
    int np;
};

#endif
