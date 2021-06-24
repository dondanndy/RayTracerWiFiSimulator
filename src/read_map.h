#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include "wall.h"
#include "vec3.h"

std::vector<Wall> read_map(const std::string& filename);
point3 cleanup_point_info(const std::string& raw_point_info);