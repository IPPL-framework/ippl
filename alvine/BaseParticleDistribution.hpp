template <typename T, unsigned Dim>
class IParticleDistribution {
public:
    virtual ~IParticleDistribution()                                                      = default;
    virtual void generateDistribution()                                                   = 0;
    virtual int getNumParticles() const                                                   = 0;
    virtual void applyFilter(std::function<bool(const ippl::Vector<T, Dim>&)> filterFunc) = 0;
    //virtual void merge(const IParticleDistribution<T, Dim>& other)                        = 0;

    virtual const typename ippl::detail::ViewType<ippl::Vector<T, Dim>, 1>::view_type& getParticles() const = 0;
};

template <typename T, unsigned Dim>
class PlacementStrategy {
public:
    virtual ~PlacementStrategy() = default;
    virtual void placeParticles(
        typename ippl::detail::ViewType<ippl::Vector<T, Dim>, 1>::view_type& container,
        ippl::Vector<T, Dim> rmin, ippl::Vector<T, Dim> rmax) const = 0;
};


template <typename T, unsigned Dim>
class ParticleDistributionBase : public IParticleDistribution<T, Dim> {
protected:
    ippl::Vector<T, Dim> rmin, rmax;
    typename ippl::detail::ViewType<ippl::Vector<T, Dim>, 1>::view_type particle_container;
    std::unique_ptr<PlacementStrategy<T, Dim>> placementStrategy;

public:
    ParticleDistributionBase(ippl::Vector<T, Dim> rmin_, ippl::Vector<T, Dim> rmax_, PlacementStrategy<T, Dim>* strategy)
        : rmin(rmin_)
        , rmax(rmax_)
        , placementStrategy(strategy){}

    virtual void generateDistribution() override {
        if (placementStrategy) {
            placementStrategy->placeParticles(particle_container, rmin, rmax);
        } else {
            throw std::logic_error("Placement strategy not initialized.");
        }
    }

    void applyFilter(std::function<bool(const ippl::Vector<T, Dim>&)> filterFunc) override {
        Kokkos::View<int*> indices("indices", particle_container.extent(0));
        int numInside = 0;

        Kokkos::parallel_scan(
            "FilterParticles", particle_container.extent(0),
            KOKKOS_LAMBDA(const int& i, int& update, const bool final) {
                if (filterFunc(particle_container(i))) {
                    if (final) {
                        indices(update) = i;
                    }
                    update += 1;
                }
            },
            numInside);

        typename ippl::detail::ViewType<ippl::Vector<T, Dim>, 1>::view_type new_particles(
            "FilteredParticles", numInside);

        Kokkos::parallel_for(
            "CopyFilteredParticles", numInside,
            KOKKOS_LAMBDA(const int& i) { new_particles(i) = particle_container(indices(i)); });

        particle_container = new_particles;
        Kokkos::fence();
    }

    void setPlacementStrategy(PlacementStrategy<T, Dim>* strategy) {
        placementStrategy.reset(strategy);
    }

    int getNumParticles() const override { return particle_container.extent(0); }

    const typename ippl::detail::ViewType<ippl::Vector<T, Dim>, 1>::view_type& getParticles() const override {
        return particle_container;
    }
};

template <typename T, unsigned Dim>
class GridPlacement : public PlacementStrategy<T, Dim> {
    ippl::Vector<int, Dim> num_points;
    ippl::Vector<T, Dim> rmin;
    ippl::Vector<T, Dim> rmax;

public:
    GridPlacement(ippl::Vector<int, Dim> num_points_)
        : num_points(num_points_) {}

    void placeParticles(
        typename ippl::detail::ViewType<ippl::Vector<T, Dim>, 1>::view_type& container,
        ippl::Vector<T, Dim> rmin, ippl::Vector<T, Dim> rmax) const override {
        ippl::Vector<T, Dim> dr = (rmax - rmin) / (num_points - 1);

         size_t total_num =
             std::reduce(num_points.begin(), num_points.end(), 1, std::multiplies<int>());
         Kokkos::resize(container, total_num);

        if constexpr (Dim == 2) {
            Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_points(0), num_points(1)});
            Kokkos::parallel_for(
                "2DGridInit", policy, KOKKOS_LAMBDA(const int i, const int j) {
                    ippl::Vector<T, 2> loc(i, j);
                    container(i * num_points(0) + j) = rmin + dr * loc;
                });
        }
        Kokkos::fence();
    }
};


template <typename T, unsigned Dim>
class GridDistribution : public ParticleDistributionBase<T, Dim> {
public:
    GridDistribution(ippl::Vector<int, Dim> num_points, ippl::Vector<T, Dim> rmin_, ippl::Vector<T, Dim> rmax_)
        : ParticleDistributionBase<T, Dim>(rmin_, rmax_, new GridPlacement<T, Dim>(num_points)) {
        this->generateDistribution();
    }
};

// template <typename T, unsigned Dim>
// class AbstractParticleDistribution {
//     using vector_type = ippl::Vector<T, Dim>;
//     using view_type   = typename ippl::detail::ViewType<vector_type, 1>::view_type;
//
// protected:
// public:
//     view_type particle_container;
//
//     AbstractParticleDistribution(vector_type rmin_, vector_type rmax_)
//         : rmin(rmin_)
//         , rmax(rmax_) {}
//
//     view_type getParticles() { return particle_container; }
//
//     int getNumParticles() { return particle_container.extent(0); }
//
//     void operator+=(const AbstractParticleDistribution<T, Dim>& other) {
//         size_t total_size = particle_container.extent(0) + other.particle_container.extent(0);
//
//         view_type tmp("combination", total_size);
//         auto& other_container = other.particle_container;
//
//         Kokkos::parallel_for(
//             "CopyOriginalData", particle_container.extent(0),
//             KOKKOS_LAMBDA(const int i) { tmp(i) = particle_container(i); });
//
//         Kokkos::parallel_for(
//             "CopyOtherData", other_container.extent(0), KOKKOS_LAMBDA(const int i) {
//                 tmp(particle_container.extent(0) + i) = other_container(i);
//             });
//         Kokkos::fence();
//
//         particle_container = tmp;
//     }
//
// private:
//     virtual void generateDistribution() = 0;
//
//     inline bool isInsideBoundary(const vector_type& point) {
//         for (unsigned d = 0; d < Dim; d++) {
//             if (point(d) > rmax(d) || point(d) < rmin(d)) {
//                 return false;
//             }
//         }
//
//         return true;
//     }
//
// protected:
//     vector_type rmin, rmax;
//
//     void removeParticlesOutsideBoundary() {
//         Kokkos::View<int*> indices("indices", particle_container.extent(0));
//         int numInside;
//
//         Kokkos::parallel_scan(
//             "FlagInsideParticles", particle_container.extent(0),
//             KOKKOS_LAMBDA(const int& i, int& update, const bool final) {
//                 if (isInsideBoundary(particle_container(i))) {
//                     if (final) {
//                         indices(update) = i;
//                     }
//                     update += 1;
//                 }
//             },
//             numInside);
//
//         view_type new_particles("FilteredParticles", numInside);
//
//         Kokkos::parallel_for(
//             "CopyInsideParticles", numInside,
//             KOKKOS_LAMBDA(const int& i) { new_particles(i) = particle_container(indices(i)); });
//
//         particle_container = new_particles;
//     }
// };
//
// template <typename T, unsigned Dim>
// class GridDistribution : public AbstractParticleDistribution<T, Dim> {
// public:
//     using vector_type = ippl::Vector<T, Dim>;
//     using view_type   = typename ippl::detail::ViewType<vector_type, 1>::view_type;
//     ippl::Vector<int, Dim> num_points;
//
//     GridDistribution(ippl::Vector<int, Dim> num_points_, vector_type rmin_, vector_type rmax_)
//         : AbstractParticleDistribution<T, Dim>(rmin_, rmax_)
//         , num_points(num_points_) {
//         generateDistribution();
//     }
//
// private:
//     void generateDistribution() override {
//         size_t total_num =
//             std::reduce(num_points.begin(), num_points.end(), 1, std::multiplies<int>());
//         Kokkos::resize(this->particle_container, total_num);
//
//         ippl::Vector<T, Dim> dr = (this->rmax - this->rmin) / (num_points - 1);
//
//         if constexpr (Dim == 2) {
//             Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_points(0),
//             num_points(1)});
//
//             Kokkos::parallel_for(
//                 "2DLoopInit", policy, KOKKOS_LAMBDA(const int i, const int j) {
//                     ippl::Vector<int, 2> loc(i, j);
//                     this->particle_container(i * num_points(0) + j) = dr * loc;
//                 });
//         } else if constexpr (Dim == 3) {
//             Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
//                 {0, 0, 0}, {num_points(0), num_points(1), num_points(2)});
//             Kokkos::parallel_for(
//                 "3DLoopInit", policy, KOKKOS_LAMBDA(const int i, const int j, const int k) {
//                     ippl::Vector<int, 3> loc(i, j, k);
//                     this->particle_container(i * num_points(0) + j * num_points(1) *
//                     num_points(0)
//                                              + k) = dr * loc;
//                 });
//         }
//         Kokkos::fence();
//     }
// };
