echo ------------Fractal is starting---------------
make clean
make
./fractal |& tee -a terminal.out
echo -------------Fractal is done------------------
echo
