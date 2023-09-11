#ifndef IPPL_BASE_MANAGER_H
#define IPPL_BASE_MANAGER_H

#include "Ippl.h"



namespace ippl {

    class BaseManager {
    public:
        BaseManager()          = default;
        virtual ~BaseManager() = default;

        virtual void pre_run() { /* default does nothing */
        }

        virtual void post_run() { /* default does nothing */
        }

        virtual void pre_step(double /*t*/) { /* default does nothing */
        }

        virtual void post_step(double /*t*/) { /* default does nothing */
        }

        virtual void advance() { /* default does nothing */
        }
        /*
        void run(double tstart, double tstop) {
            for (double t = tstart; t <= tstop; t += dt) {
                this->pre_step(t);

                this->advance(t, dt);

                this->post_step(t);
            }
        }
        */
    };
}  // namespace ippl

#endif
