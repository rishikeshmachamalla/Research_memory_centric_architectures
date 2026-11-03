CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter -Isrc
all: test bench bfs pcdump
test: src/test_collectives.cpp src/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
bench: src/bench.cpp src/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
bfs: src/bfs.cpp src/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
pcdump: src/dump.cpp src/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
clean:
	rm -f test bench bfs pcdump
