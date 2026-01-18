CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell python -m pybind11 --includes)

LIBNAME = helpers_cpp
LIBEXT = $(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@