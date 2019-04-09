// 
// walkring.cc
//
// 1d random walk on a ring
//
// Compile with make using provided Makefile 
//

#include <fstream>
#include <rarray>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include "walkring_output.h"
#include "walkring_timestep.h"
#include "parameters.h"

// the main function drives the simulation
int main(int argc, char *argv[]) 
{
  // Simulation parameters
  double      L;  // ring length
  double      D;  // diffusion constant
  double      T;  // time
  double      dx; // spatial resolution
  double      dt; // temporal resolution (time step)
  int         Z;  // number of walkers
  std::string datafile; // filename for output
  double      time_between_output;

  // Read parameters from a file given on the command line. 
  // If no file was given, use "params.ini".
  std::string paramFilename = argc>1?argv[1]:"params.ini";
  read_parameters(paramFilename, L, D, T, dx, dt, Z, datafile, time_between_output);

  // Compute derived parameters 
  const int numSteps = int(T/dt + 0.5);  // number of steps to take
  const int N = int(L/dx + 0.5);         // number of grid points
  const int outputEvery = int(time_between_output/dt + 0.5); // how many steps between output
  const double p = D*dt/pow(dx,2);       // probability to hop left or right
  const int outputcols = 48;             // number of columns for sparkline output
    
  // Allocate walker data
  int w[Z]; // array to hole walker data
  if (rank ==0){rarray<int,1> w_print(Z);} // rarray for copy of w to put in walkring_output
  
  // Setup initial conditions for w
  std::fill(w,w+Z,(N/2)); // Z is the length of w
 
   // Setup initial time
  double time = 0.0;
  
  // Copy reg type array to printout rarray
  for (int i = 0; i < Z; i++) {w_print[i] = w[i];}
  
  
  if (rank == 0){
    // Open a file for data output
    std::ofstream file;
    walkring_output_init(file, datafile);  
    // Initial output to screen
    walkring_output(file, 0, time, N, w_print, outputcols);}
  
  // Starting up with the MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  const int local_length = Z/size; // length of local arrays. Length of global array will be Z.
  int localdata[local_length]; // buffer of data to hold set of elements

  //int w_new[] = w; // Turning rarray into a regular C++ array for MPI
  
  // Time evolution
  for (int step = 1; step <= numSteps; step++) {
    
    MPI_Scatter(w, local_length, MPI_INT, localdata, local_length, MPI_INT, 0, MPI_COMM_WORLD); // Sharing the work
    
    // Want the input for walkring timestep to be the smaller arrays
    walkring_timestep(localdata, N, p, rank, size, local_length); // Compute next time point
    
    MPI_Gather(localdata, local_length, MPI_INT, w, local_length, MPI_INT, 0, MPI_COMM_WORLD); 
    
    if (rank == 0){
      // Copy reg type array to printout rarray
      for (int i = 0; i < Z; i++) w_print[i] = w[i];
      // Update time
      time += dt;
      
      // Periodically add data to the file
      if (step % outputEvery == 0 and step > 0)
      walkring_output(file, step, time, N, w_print, outputcols);
    }
  }
  
  MPI_Finalize();
  
  // Close file
  walkring_output_finish(file);

  // All done
  return 0;
}

