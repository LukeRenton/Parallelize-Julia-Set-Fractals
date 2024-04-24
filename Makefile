P1 = fractal
#Choose either the C compiler or the C++ compiler from the following 2 lines
CC = gcc
CPP = g++
CFLAGS = -g -Wall
OMPFLAG = -fopenmp
INCFLAG = -I "../common/"
all: $(P1)

$(P1): $(P1).cpp
	$(CPP) $(INCFLAG) $(CFLAGS) $(OMPFLAG) $(P1).cpp -o $(P1) -lglut -lGL -w

clean:
	rm -vf $(P1)
