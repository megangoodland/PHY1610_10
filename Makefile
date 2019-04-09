# Makefile for diffring and walkring
#
# Ramses van Zon, March 2016
#

BLAS_INC?=.
BLAS_LIB?=.

CXX=mpicxx
CXXFLAGS=-O3 -g -march=native -std=c++11 -I${BLAS_INC}
LDFLAGS=-g -L${BLAS_LIB}
LDLIBS=-lopenblas
LD=${CXX}

.PHONY: all clean distclean run rundiffring runwalkring

all: diffring walkring

diffring: diffring.o diffring_timestep.o diffring_output.o parameters.o ticktock.o sparkline.o
	${LD} ${LDFLAGS} -o diffring diffring.o diffring_timestep.o diffring_output.o parameters.o ticktock.o sparkline.o ${LDLIBS}

walkring: walkring.o ticktock.o sparkline.o walkring_output.o walkring_timestep.o parameters.o
	${LD} ${LDFLAGS} -o walkring walkring.o ticktock.o sparkline.o walkring_output.o walkring_timestep.o parameters.o ${LDLIBS}

diffring.o: diffring.cc diffring_timestep.h
	${CXX} ${CXXFLAGS} -c -o diffring.o diffring.cc

walkring.o: walkring.cc walkring_output.h walkring_timestep.h parameters.h
	${CXX} ${CXXFLAGS} -c -o walkring.o walkring.cc

walkring_output.o: walkring_output.cc walkring_output.h
	${CXX} ${CXXFLAGS} -c -o walkring_output.o walkring_output.cc

walkring_timestep.o: walkring_timestep.cc walkring_timestep.h
	${CXX} ${CXXFLAGS} -c -o walkring_timestep.o walkring_timestep.cc

parameters.o: parameters.cc parameters.h
	${CXX} ${CXXFLAGS} -c -o parameters.o parameters.cc

diffring_output.o: diffring_output.cc diffring_output.h
	${CXX} ${CXXFLAGS} -c -o diffring_output.o diffring_output.cc

diffring_timestep.o: diffring_timestep.cc diffring_timestep.h
	${CXX} ${CXXFLAGS} -c -o diffring_timestep.o diffring_timestep.cc

ticktock.o: ticktock.cc ticktock.h sparkline.h
	${CXX} ${CXXFLAGS} -c -o ticktock.o ticktock.cc

sparkline.o: sparkline.cc sparkline.h sparkline.h
	${CXX} ${CXXFLAGS} -c -o sparkline.o sparkline.cc

clean:
	\rm -f ticktock.o diffring.o walkring.o walkring_output.o walkring_timestep.o parameters.o diffring.o diffring_output.o diffring_timestep.o parameters.o sparkline.o

distclean: clean
	\rm -f output.dat diffring walkring

run: rundiffring runwalkring

rundiffring: diffring params.ini
	./diffring params.ini

runwalkring: walkring params.ini
	./walkring params.ini

