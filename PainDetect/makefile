filename = faceRec.cpp

all: compile run

compile: $(filename) 
	g++ -ggdb `pkg-config --cflags opencv` -o `basename $(filename) .cpp` $(filename) `pkg-config --libs opencv` 
	
run:
	./`basename $(filename) .cpp`

clean:
	rm -rf ./`basename $(filename) .cpp` *.png

