template <typename T, unsigned Dim>
class AbstractParticleDistribution {
  using vector_type = ippl::Vector<T, Dim>;
  using view_type = typename ippl::detail::ViewType<vector_type, 1>::view_type;

  protected:

  public:
    view_type particle_container;


    AbstractParticleDistribution(vector_type rmin_, vector_type rmax_) : rmin(rmin_), rmax(rmax_) {}

    view_type getParticles() {
        return particle_container;
    }

    int getNumParticles() {
      return particle_container.extent(0);
    }

    void operator+=(const AbstractParticleDistribution<T, Dim>& other) {
      size_t total_size = particle_container.extent(0) + other.particle_container.extent(0);

      view_type tmp("combination", total_size);
      auto& other_container = other.particle_container;

      Kokkos::parallel_for("CopyOriginalData", particle_container.extent(0), KOKKOS_LAMBDA(const int i) {
          tmp(i) = particle_container(i);
      });

      Kokkos::parallel_for("CopyOtherData", other_container.extent(0), KOKKOS_LAMBDA(const int i) {
          tmp(particle_container.extent(0) + i) = other_container(i); 
      });
      Kokkos::fence();

      particle_container = tmp;
      removeParticlesOutsideBoundary();
    }


  private:
    vector_type rmin, rmax;

    virtual void generateDistribution() = 0;

    inline bool isInsideBoundary(const vector_type& point) {
        for (unsigned d = 0; d < Dim; d++) {
            if (point(d) > rmax(d) || point(d) < rmin(d)) {
              return false;
            }
        }

        return true;
    }

  protected:

    void removeParticlesOutsideBoundary() {
        Kokkos::View<int*> indices("indices", particle_container.extent(0));
        int numInside;

        Kokkos::parallel_scan("FlagInsideParticles", particle_container.extent(0), KOKKOS_LAMBDA(const int& i, int& update, const bool final) {
            if (isInsideBoundary(particle_container(i))) {
                if (final) { 
                    indices(update) = i;
                }
                update += 1; 
            }
        }, numInside);

        view_type new_particles("FilteredParticles", numInside);

        Kokkos::parallel_for("CopyInsideParticles", numInside, KOKKOS_LAMBDA(const int& i) {
            new_particles(i) = particle_container(indices(i));
        });

        particle_container = new_particles;
    }

};

template <typename T, unsigned Dim>
class ConcreteParticleDistribution : public AbstractParticleDistribution<T, Dim> {
  public:
    using vector_type = ippl::Vector<T, Dim>;
    using view_type = typename ippl::detail::ViewType<vector_type, 1>::view_type;
    int val;

    ConcreteParticleDistribution(vector_type rmin_, vector_type rmax_, int val_) : AbstractParticleDistribution<T, Dim>(rmin_, rmax_) {
      this->val = val_;
      generateDistribution();

    }

    void print() {
      for (size_t i = 0; i < this->particle_container.extent(0); i++) {
        std::cout << this->particle_container(i) << std::endl;
      }

    }

  private:
    void generateDistribution() override {
        Kokkos::resize(this->particle_container, 10);
        for (size_t i = 0; i < 10; i++) {
          this->particle_container(i) = this->val;

        }

    }

};


