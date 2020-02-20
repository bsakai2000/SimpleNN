ifndef CXX
CXX=g++
endif

CFLAGS:=$(CFLAGS) -g -Wall -I./src

check:
	mkdir -p ./bin
	$(CXX) ./src/network.cpp ./test/test.cpp -o ./bin/test $(CFLAGS)
	./bin/test

clean:
	rm -rf ./bin
	rm -f *.gcov *.gcda *.gcno
