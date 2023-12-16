#ifndef IPPL_BASE_MANAGER_H
#define IPPL_BASE_MANAGER_H

#include "Ippl.h"


namespace ippl {

    /**
    * @class BaseManager
    * @brief A base class for managing simulations using IPPL.
    *
    * The BaseManager class provides a basic structure for running simulations with common steps.
    */
    class BaseManager {
    public:
        BaseManager()          = default;
        virtual ~BaseManager() = default;

       /**
        * @brief A method that should be used for setting up the simulation.
        *
        * Derived classes can override this method to allocate memory, initialize variables, etc.
        * The default implementation does nothing.
        */
        virtual void pre_run() { /* default does nothing */
        }

       /**
        * @brief A method that should be used for preparation before perfoming a step of simulation.
        *
        * The default implementation does nothing.
        */

        virtual void pre_step() { /* default does nothing */
        }

       /**
        * @brief A method that should be used after perfoming a step of simulation.
        *
        * Derived classes can override this method to dump data, increment time, etc.
        * The default implementation does nothing.
        */
        virtual void post_step() { /* default does nothing */
        }

       /**
        * @brief A method that should be used to execute a step of simulation.
        *
        * Derived classes can override this method to implement their own governing equation.
        * The default implementation does nothing.
        */
        virtual void advance() { /* default does nothing */
        }

       /**
        * @brief The main for loop fro running a simulation.
        *
        * This method performs a simulation run by calling pre_step, advance, and post_step
        * in a loop for a specified number of time steps.
        *
        * @param nt The number of time steps to run the simulation.
        */
        void run(int nt) {
            for (int it=0; it<nt; it++){
                this->pre_step();

                this->advance();

                this->post_step();
            }
        }

    };
}  // namespace ippl

#endif
