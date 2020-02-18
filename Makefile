ifndef CXX
CXX=g++
endif

CFLAGS:=$(CFLAGS) -g -Wall -I./src

runtest:
	mkdir -p ./bin
	$(CXX) ./src/network.cpp ./test/test.cpp -o ./bin/test $(CFLAGS)
	./bin/test

examples_squares:
	mkdir -p ./bin
	$(CXX) ./src/network.cpp ./examples/test_squares.cpp -o ./bin/squares $(CFLAGS)
	./bin/squares

examples_draw:
	mkdir -p ./bin
	$(CXX) ./src/network.cpp ./examples/test_draw_image.cpp -o ./bin/draw $(CFLAGS)
	./bin/draw ./examples/cat.bmp > ./bin/cat1.bmp

examples_draw_mocked:
	mkdir -p ./bin
	$(CXX) ./examples/test_draw_image.cpp -o ./bin/draw $(CFLAGS) -I./examples
	./bin/draw ./examples/cat.bmp > ./bin/cat1.bmp

clean:
	rm -rf ./bin
	rm -f *.gcov *.gcda *.gcno
