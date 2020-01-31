runtest:
	mkdir -p ./bin
	g++ -g -Wall ./src/network.cpp ./test/test.cpp -o ./bin/test
	./bin/test

clean:
	rm -rf ./bin
