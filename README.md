# kokkos-cobyla
Port of COBYLA algorithm to Kokkos

The purpose of this library is to allow the use of the COBYLA algorithm
*from inside Kokkos parallel kernels*.
A common use case is that, in a FEM context,
one may need to solve a small constrained optimization problem in each element.
That problem is exactly what kokkos-cobyla was designed to solve.

For better or worse, this is essentially a direct translation of the original
F77 code to C++, with all the `goto`s and indexing from 1 that
come with it.

# Installation
kokkos-cobyla is a header-only library.
In fact, there's only one header.
If you want, you can just copy `kokkos_cobyla.hpp` from here to
some place accessible and included by your project.

Alternatively, I wrote a bunch of overkill CMake,
so you can use that.
Follow the typical cmake invocation:
```sh
cmake -DWITH_TESTS=ON/OFF -DCMAKE_INSTALL_PREFIX=/path/to/install/dir /path/to/source/dir
make
ctest # optional, if you set WITH_TESTS=ON
make install
```
This really just copies `kokkos_cobyla.hpp` into
`/path/to/install/dir/include`.
If you did not provide an install directory path,
it should install it to the default installation directory
(I believe on Linux this is typically `/usr/`).
Helpfully, it also generates CMake config files so that your
CMake-driven project can simply add
```cmake
find_package(kokkos-cobyla REQUIRED)
```
and then, for each executable or library,
```cmake
target_link_libraries(myTarget kokkos-cobyla::kokkos-cobyla)
```

If you elected to turn `WITH_TESTS=ON` in the above CMake invocation,
running `make` will compile the unit tests.
These are derived from the original COBYLA source code.
They also serve as helpful examples for using kokkos-cobyla.

Note that you obviously have to have Kokkos for kokkos-cobyla to work.
The unit tests will not be built if CMake can't find Kokkos.
If you aren't building with tests, CMake will warn you if it can't find Kokkos,
but it'll still install kokkos-cobyla.
Just ensure that your project has access to both, and it should be fine.

# Use
In your C++ project,
```c++
#include <kokkos_cobyla.hpp>
```
All of the functions are hidden behind the `kokkos_cobyla` namespace.
The COBYLA functions retain their original call signatures
with the exceptions that `iprint`, which controls the print level,
is now a template argument,
and the `CALCFC` subroutine is now provided as a function argument.
If compiling with a C++17 (or later)-compliant compiler,
by default,
the printing code won't even be compiled,
so it should be nice and fast.
This is because kokkos-cobyla functions are intended to be called from
within CUDA kernels, from which it is slow
(and, probably, more verbose than you'd ever want)
to print.

Just as with the original COBYLA, some scratch space is needed.
To comply with Kokkos semantics,
these should be allocated from the host before entering a parallel region.
The scratch space should be provided by two Views,
one containing an integral type
and the other containing a floating point type.
For convenience,
the `requiredIntegralWorkViewSize` and `requiredScalarWorkViewSize`
functions are provided to compute the amount of space required for
these two Views, respectively.

Typically, you will only ever need to interact with the `cobyla` function.
It is heavily templated such that it'll accept pretty much anything you
throw at it.
In particular, the `CALCFC` subroutine required by COBYLA can be provided
as a function pointer, `std::function`, lambda, functor,
or pretty much anything else with a compatible call signature.

See `unit_tests.cpp` for examples of use.
