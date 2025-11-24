#pragma once

#define OPTARG_CNT(_0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, ...) a11
#define OPTARG_ARGC(...)                                                       \
	OPTARG_CNT(, ##__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define OPTARG_CAT0(x, y) x##y
#define OPTARG_CAT(x, y) OPTARG_CAT0(x, y)

#define OPTARG_FUNC(fname, ...)                                                \
	OPTARG_CAT(fname##_, OPTARG_ARGC(__VA_ARGS__))(__VA_ARGS__)
