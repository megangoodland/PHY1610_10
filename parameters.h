// 
// parameters.h
//
// Module for reading parameters for 1d random walk or diffusion on a ring
//

#ifndef PARAMETERSH
#define PARAMETERSH

#include <string>

// The function 'read_parameters' reads in the file
// inifilename and stores the parameters in that file in the variables
// L, D, T, dx, dt, Z, datafile and time_between_output.
void read_parameters(const std::string &inifilename, 
                     double&      L,        // ring length
                     double&      D,        // diffusion constant
                     double&      T,        // total time
                     double&      dx,       // spatial resolution
                     double&      dt,       // temporal resolution (time step)
                     int&         Z,        // number of walkers
                     std::string& datafile, // Name of the output file
                     double&      time_between_output); // controls how frequently the program writes out 

#endif
