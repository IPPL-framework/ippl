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

#include "Ippl.h"

#include <csignal>

extern int interruptSignalReceived;

/*!
 * Signal handler records the received signal
 * @param signal received signal
 */
void interruptHandler(int signal);

/*!
 * Checks whether a signal was received
 * @return Signal handler was called
 */
bool checkSignalHandler();

/*!
 * Sets up the signal handler
 */
void setSignalHandler();
