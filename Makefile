BIN = bin
CPPSRC = src
WEBAPP = src/webapp # 수정필요
RSRC = resources
CFG = config

#-----------------------------------------------#

CXX = g++
CXXFLAGS = -std=c++11
OPENCV = `pkg-config --cflags --libs opencv4`
JSONCPP = `pkg-config --cflags --libs jsoncpp`
FFI = `pkg-config --cflags --libs libffi`

# Sources
RELEASE_SRC = $(CPPSRC)/yolo_cpu.cpp $(CPPSRC)/addon.cpp

#-----------------------------------------------#

all:
	$(CXX) $(CXXFLAGS) $(RELEASE_SRC) $(OPENCV) $(JSONCPP) $(FFI) -c
	@echo "Compile is done!"


clean:


#-----------------------------------------------#

