#ifndef IPPL_STREAM_STEERING_BUTTON_H
#define IPPL_STREAM_STEERING_BUTTON_H

#include <ostream>

#include "Utility/IpplException.h"

namespace ippl {

    /**
     * @struct Button
     * @brief Momentary-action control for steering.
     *
     * A Button acts like an edge-trigger: it reports a single transition when
     * pressed and then resets. Useful for one-shot actions triggered from the
     * Catalyst GUI without latching state.
     */
    struct Button {
        Button() = default;

        // explicit not explicit allows: if(my_btn) to work
        operator bool() const { return value_m; }

        // explicit Button(bool v) : value_m(v) {}
        explicit Button(bool v) {
            if (v) {  // Button is initialized pressd
                value_m      = true;
                priorState_m = false;
            } else /* if(!v) */ {  // Button is initialized  unpressd
                value_m      = false;
                priorState_m = false;
            }
        }

        // Assignment operator: Button = bool
        Button& operator=(bool v) {
            if (v) {                              // Button is being pressed
                if (!value_m && !priorState_m) {  // True unpressed state
                    value_m      = true;
                    priorState_m = false;               // Fixed typo: was "priot_state"
                } else if (value_m && !priorState_m) {  // Button was pressed last iteration and is
                                                        // still "being pushed down"
                    // Internal button needs to snap back! And mark it as "freshly snapped"
                    value_m      = false;
                    priorState_m = true;
                } else if (!value_m
                           && priorState_m) {  // Button was pressed at some previous iteration and
                                               // is still "being pushed down"
                    // Button needs to stay in freshly snapped back state
                    value_m      = false;
                    priorState_m = true;
                } else if (value_m && priorState_m) {  // Impossible state
                    throw IpplException("CatalystAdaptor::Button Assignment",
                                        "Impossible State: Button is malfunctioning!!!");
                }
            } else {  // Button is unpressed
                value_m      = false;
                priorState_m = false;
            }
            return *this;  // CRITICAL: Return reference to this object for chaining
        }

        // Friend function to overload << operator for output streams
        friend std::ostream& operator<<(std::ostream& os, const Button& btn) {
            os << (btn.value_m ? "PUSHED" : "not PUSHED");
            return os;
        }

    private:
        bool value_m      = false;
        bool priorState_m = false;
    };  // Button

}  // namespace ippl

#endif
