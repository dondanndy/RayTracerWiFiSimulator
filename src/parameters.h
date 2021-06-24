//Constant parameters for simulation.
#pragma once

#include <cmath>

//Dirty way to make a constexpr pow for below.
constexpr size_t pow_const(size_t base, size_t exp) {
	size_t res = base;
	
	for (size_t i = 1; i < exp; i++) {
		res *= base;
	}

	return res;
}

//No constexpr pow :(
constexpr size_t MAX_REBOUND = 5;
constexpr size_t MAX_STACK_VALUES = pow_const(2, MAX_REBOUND+1) - MAX_REBOUND + 2;

constexpr float CUTOFF_POWER = 1e-12;
constexpr float EMITTING_POWER = 0.1; //Watts

constexpr float PI = 3.1415926535;

constexpr float RECEPTOR_RADIUS = 0.025; //Meters
constexpr float ANGLE_STEP = 0.125; //Degrees