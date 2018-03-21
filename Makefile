GCC = g++
VERS ?= 2
OMP := $(if $(strip $(OMP)), $(strip $(OMP)), $(or ${OMP_NUM_THREADS}, 0))
ifneq ($(OMP), 0)
	CFLAGS += -Domp_num_threads=$(OMP) -fopenmp
LIBS += -lgomp
endif
INC_DIR = include/v$(VERS)/
SRC_DIR = src/v$(VERS)/
OBJ_DIR = obj/
INCL = -I$(INC_DIR) -I$(INC_DIR)CCfits -I$(INC_DIR)cfitsio
LIBS += -L${MY_LIB} -L/usr/lib64 -lm -l sqlite3
OBJS = $(OBJ_DIR)calc_distances.o $(OBJ_DIR)io.o $(OBJ_DIR)run_calc.o
EXEC = run
CFLAGS += -Wall -g -std=c++11
OFLAGS = -g

all : $(EXEC)

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(GCC) $(INCL) $(LIBS) $(CFLAGS) -c -lstdc++fs $^ -o $@

$(EXEC) : $(OBJS)
	$(GCC) $(OFLAGS) $^ $(LIBS) -o $(EXEC) -lstdc++fs

clean:
	@rm -f $(OBJS)

uninstall:
	@rm -f $(OBJS) $(EXEC)

print-%:
	@echo '$* = $($*)'
	@echo '    origin = $(origin $*)'
	@echo '    flavor = $(flavor $*)'
	@echo '    value  = $(value $*)'
