/*

This module implements a data acquisition interface

*/
#pragma once
#include <string>

#ifdef ARPAM_HAS_ALAZARTECH

namespace daq {
    std::string getDAQInfo();
}


#endif 
