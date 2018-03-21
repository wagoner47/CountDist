#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include "calc_distances.h"

std::vector<Pos> readCatalog(std::string);

void writeFnames(std::string, std::vector<std::string>);

namespace ioconstants {
	const double DELTAP = 50.0;
	const double DELTAL = 60.0;
}

#endif
