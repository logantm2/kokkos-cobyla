/*
COBYLA---Constrained Optimization BY Linear Approximation.
Copyright (C) 1992 M. J. D. Powell (University of Cambridge)
Copyright (C) 2022 L. T. Meredith

This package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License
 https://www.gnu.org/copyleft/lesser.html
for more details.
*/

#ifndef KOKKOS_COBYLA_HPP
#define KOKKOS_COBYLA_HPP

#include <Kokkos_Core.hpp>

/*
This subroutine minimizes an objective function F(X) subject to M
inequality constraints on X, where X is a vector of variables that has
N components. The algorithm employs linear approximations to the
objective and constraint functions, the approximations being formed by
linear interpolation at N+1 points in the space of the variables.
We regard these interpolation points as vertices of a simplex. The
parameter RHO controls the size of the simplex and it is reduced
automatically from RHOBEG to RHOEND. For each RHO the subroutine tries
to achieve a good vector of variables for the current size, and then
RHO is reduced until the value RHOEND is reached. Therefore RHOBEG and
RHOEND should be set to reasonable initial changes to and the required
accuracy in the variables respectively, but this accuracy should be
viewed as a subject for experimentation because it is not guaranteed.
The subroutine has an advantage over many of its competitors, however,
which is that it treats each constraint individually when calculating
a change to the variables, instead of lumping the constraints together
into a single penalty function. The name of the subroutine is derived
from the phrase Constrained Optimization BY Linear Approximations.

The user must set the values of N, M, RHOBEG and RHOEND, and must
provide an initial vector of variables in X. Further, the value of
IPRINT should be set to 0, 1, 2 or 3, which controls the amount of
printing during the calculation. Specifically, there is no output if
IPRINT=0 and there is output only at the end of the calculation if
IPRINT=1. Otherwise each new value of RHO and SIGMA is printed.
Further, the vector of variables and some function information are
given either when RHO is reduced or when each new value of F(X) is
computed in the cases IPRINT=2 or IPRINT=3 respectively. Here SIGMA
is a penalty parameter, it being assumed that a change to X is an
improvement if it reduces the merit function
           F(X)+SIGMA*MAX(0.0,-C1(X),-C2(X),...,-CM(X)),
where C1,C2,...,CM denote the constraint functions that should become
nonnegative eventually, at least to the precision of RHOEND. In the
printed output the displayed term that is multiplied by SIGMA is
called MAXCV, which stands for 'MAXimum Constraint Violation'. The
argument MAXFUN is an integer variable that must be set by the user to a
limit on the number of calls of CALCFC, the purpose of this routine being
given below. The value of MAXFUN will be altered to the number of calls
of CALCFC that are made. The arguments W and IACT provide real and
integer arrays that are used as working space. Their lengths must be at
least N*(3*N+2*M+11)+4*M+6 and M+1 respectively.

In order to define the objective and constraint functions, we require
a subroutine that has the name and arguments
           SUBROUTINE CALCFC (N,M,X,F,CON)
           DIMENSION X(*),CON(*)  .
The values of N and M are fixed and have been defined already, while
X is now the current vector of variables. The subroutine should return
the objective and constraint functions at X in F and CON(1),CON(2),
...,CON(M). Note that we are trying to adjust X so that F(X) is as
small as possible subject to the constraint functions being nonnegative.

Partition the working space array W to provide the storage that is needed
for the main calculation.
*/

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_FUNCTION
void cobyla(
    IntegralType n,
    IntegralType m,
    SolutionViewType x,
    ScalarType rhobeg,
    ScalarType rhoend,
    // IntegralType iprint,
    IntegralType maxfun,
    ScalarWorkViewType w,
    IntegralWorkViewType iact
);

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_FUNCTION
void cobylb(
    IntegralType n,
    IntegralType m,
    IntegralType mpp,
    SolutionViewType x,
    ScalarType rhobeg,
    ScalarType rhoend,
    // IntegralType iprint,
    IntegralType maxfun,
    ScalarWorkViewType con,
    ScalarWorkViewType sim,
    ScalarWorkViewType simi,
    ScalarWorkViewType datmat,
    ScalarWorkViewType a,
    ScalarWorkViewType vsig,
    ScalarWorkViewType veta,
    ScalarWorkViewType sigbar,
    ScalarWorkViewType dx,
    ScalarWorkViewType w,
    IntegralWorkViewType iact
);

#endif // KOKKOS_COBYLA_HPP