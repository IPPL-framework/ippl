// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

/***************************************************************************
 *
 * Simple routine to calculate 32-bit CRC, assuming 32-bit integers.
 *
 ***************************************************************************/

#ifndef IPPL_MESSAGE_CRC_H
#define IPPL_MESSAGE_CRC_H

#ifdef __cplusplus
extern "C"
{
#endif

    typedef unsigned int CRCTYPE;

    /* calculate the CRC for the given buffer of bytes, of length icnt */
    CRCTYPE crc(void *icp, int icnt);

#ifdef __cplusplus
}
#endif

#endif // IPPL_MESSAGE_CRC_H
