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

        virtual void pre_step() { /* default does nothing */
        }

        virtual void post_step() { /* default does nothing */
        }

        virtual void advance() { /* default does nothing */
        }
        
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
