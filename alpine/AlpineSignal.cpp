// ALPINE Signal Handler
//   Signal handling utilities for ALPINE
//
// Copyright (c) 2023 Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#include "AlpineSignal.h"

int interruptSignalReceived = 0;

void interruptHandler(int signal) {
    interruptSignalReceived = signal;
}

bool checkSignalHandler() {
    ippl::Comm->barrier();
    return interruptSignalReceived != 0;
}

void setSignalHandler() {
    struct sigaction sa {};
    sa.sa_handler = interruptHandler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGTERM, &sa, nullptr) == -1) {
        std::cerr << ippl::Comm->rank() << ": failed to set up signal handler for SIGTERM ("
                  << SIGTERM << ")" << std::endl;
    }
    if (sigaction(SIGINT, &sa, nullptr) == -1) {
        std::cerr << ippl::Comm->rank() << ": failed to set up signal handler for SIGINT ("
                  << SIGINT << ")" << std::endl;
    }
}
