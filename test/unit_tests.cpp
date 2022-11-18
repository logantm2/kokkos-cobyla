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

#include "kokkos_cobyla.hpp"
#include <gtest/gtest.h>

using namespace kokkos_cobyla;

namespace math =
#if KOKKOS_VERSION < 30700
    Kokkos::Experimental;
#else
    Kokkos;
#endif

static constexpr double rhobeg = 0.5;
static constexpr double rhoend = 0.0001;
static constexpr int maxfun = 2000;

// Minimization of a simple quadratic function of two variables.
KOKKOS_INLINE_FUNCTION
void SimpleQuadratic(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1)))
) {
    f = 10.0 * math::pow(x(0) + 1.0, 2.0) + math::pow(x(1), 2.0);
}

TEST(unit_tests, SimpleQuadratic) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=0;
    Kokkos::View<double*> x("SimpleQuadratic::x", n);
    Kokkos::View<double*> w("SimpleQuadratic::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("SimpleQuadratic::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "SimpleQuadratic::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            SimpleQuadratic
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - (-1.0), 2.0);
    l2_error += std::pow(h_x(1) -   0.0 , 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// Easy two dimensional minimization in unit circle.
KOKKOS_INLINE_FUNCTION
void TwoDUnitCircleCalculation(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = x(0) * x(1);
    con(0) = 1.0 - math::pow(x(0), 2.0) - math::pow(x(1), 2.0);
}

TEST(unit_tests, TwoDUnitCircleCalculation) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=1;
    Kokkos::View<double*> x("TwoDUnitCircleCalculation::x", n);
    Kokkos::View<double*> w("TwoDUnitCircleCalculation::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("TwoDUnitCircleCalculation::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "TwoDUnitCircleCalculation::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            &TwoDUnitCircleCalculation
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - std::sqrt(0.5), 2.0);
    l2_error += std::pow(h_x(1) + std::sqrt(0.5), 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// Easy three dimensional minimization in ellipsoid.
KOKKOS_INLINE_FUNCTION
void ThreeDEllipsoidCalculation(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = x(0) * x(1) * x(2);
    con(0) = 1.0 - math::pow(x(0), 2.0) - 2.0*math::pow(x(1), 2.0) - 3.0*math::pow(x(2), 2.0);
}

TEST(unit_tests, ThreeDEllipsoidCalculation) {
    Kokkos::ScopeGuard kokkos;

    int n=3;
    int m=1;
    Kokkos::View<double*> x("ThreeDEllipsoidCalculation::x", n);
    Kokkos::View<double*> w("ThreeDEllipsoidCalculation::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("ThreeDEllipsoidCalculation::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "ThreeDEllipsoidCalculation::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            &ThreeDEllipsoidCalculation
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - std::sqrt(1.0/3.0), 2.0);
    l2_error += std::pow(h_x(1) - std::sqrt(1.0/6.0), 2.0);
    l2_error += std::pow(h_x(2) +           1.0/3.0 , 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// Weak version of Rosenbrock's problem.
KOKKOS_INLINE_FUNCTION
void WeakRosenbrock(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1)))
) {
    f = math::pow(math::pow(x(0), 2.0) - x(1), 2.0) + math::pow(1.0 + x(0), 2.0);
}

TEST(unit_tests, WeakRosenbrock) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=0;
    Kokkos::View<double*> x("WeakRosenbrock::x", n);
    Kokkos::View<double*> w("WeakRosenbrock::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("WeakRosenbrock::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "WeakRosenbrock::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            WeakRosenbrock
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) + 1.0, 2.0);
    l2_error += std::pow(h_x(1) - 1.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// Intermediate version of Rosenbrock's problem.
KOKKOS_INLINE_FUNCTION
void IntermediateRosenbrock(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1)))
) {
    f = 10.0 * math::pow(math::pow(x(0), 2.0) - x(1), 2.0) + math::pow(1.0 + x(0), 2.0);
}

TEST(unit_tests, IntermediateRosenbrock) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=0;
    Kokkos::View<double*> x("IntermediateRosenbrock::x", n);
    Kokkos::View<double*> w("IntermediateRosenbrock::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("IntermediateRosenbrock::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "IntermediateRosenbrock::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            IntermediateRosenbrock
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) + 1.0, 2.0);
    l2_error += std::pow(h_x(1) - 1.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// This problem is taken from Fletcher's book Practical Methods of
// Optimization and has the equation number (9.1.15).
KOKKOS_INLINE_FUNCTION
void Fletcher9(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = - x(0) - x(1);
    con(0) = x(1) - math::pow(x(0), 2.0);
    con(1) = 1.0 - math::pow(x(0), 2.0) - math::pow(x(1), 2.0);
}

TEST(unit_tests, Fletcher9) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=2;
    Kokkos::View<double*> x("Fletcher9::x", n);
    Kokkos::View<double*> w("Fletcher9::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("Fletcher9::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "Fletcher9::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            Fletcher9
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - std::sqrt(0.5), 2.0);
    l2_error += std::pow(h_x(1) - std::sqrt(0.5), 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// This problem is taken from Fletcher's book Practical Methods of
// Optimization and has the equation number (14.4.2).
KOKKOS_INLINE_FUNCTION
void Fletcher14(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = x(2);
    con(0) = 5.0 * x(0) - x(1) + x(2);
    con(1) = x(2) - math::pow(x(0), 2.0) - math::pow(x(1), 2.0) - 4.0 * x(1);
    con(2) = x(2) - 5.0 * x(0) - x(1);
}

TEST(unit_tests, Fletcher14) {
    Kokkos::ScopeGuard kokkos;

    int n=3;
    int m=3;
    Kokkos::View<double*> x("Fletcher14::x", n);
    Kokkos::View<double*> w("Fletcher14::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("Fletcher14::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "Fletcher14::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            Fletcher14
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0)      , 2.0);
    l2_error += std::pow(h_x(1) + 3.0, 2.0);
    l2_error += std::pow(h_x(2) + 3.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// This problem is taken from page 66 of Hock and Schittkowski's book Test
// Examples for Nonlinear Programming Codes. It is their test problem Number
// 43, and has the name Rosen-Suzuki.
KOKKOS_INLINE_FUNCTION
void RosenSuzuki(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f =
        math::pow(x(0), 2.0) +
        math::pow(x(1), 2.0) +
        2.0 * math::pow(x(2), 2.0) +
        math::pow(x(3), 2.0) -
        5.0 * x(0) - 5.0 * x(1) - 21.0 * x(2) + 7.0 * x(3);
    con(0) = 8.0 - math::pow(x(0), 2.0) - math::pow(x(1), 2.0) -
        math::pow(x(2), 2.0) - math::pow(x(3), 2.0) -
        x(0) + x(1) - x(2) + x(3);
    con(1) = 10.0 - math::pow(x(0), 2.0) - 2.0 * math::pow(x(1), 2.0)
        - math::pow(x(2), 2.0) - 2.0 * math::pow(x(3), 2.0) + x(0) + x(3);
    con(2) = 5.0 - 2.0 * math::pow(x(0), 2.0) - math::pow(x(1), 2.0) -
        math::pow(x(2), 2.0) - 2.0 * x(0) + x(1) + x(3);
}

TEST(unit_tests, RosenSuzuki) {
    Kokkos::ScopeGuard kokkos;

    int n=4;
    int m=3;
    Kokkos::View<double*> x("RosenSuzuki::x", n);
    Kokkos::View<double*> w("RosenSuzuki::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("RosenSuzuki::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "RosenSuzuki::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            RosenSuzuki
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0)      , 2.0);
    l2_error += std::pow(h_x(1) - 1.0, 2.0);
    l2_error += std::pow(h_x(2) - 2.0, 2.0);
    l2_error += std::pow(h_x(3) + 1.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// This problem is taken from page 111 of Hock and Schittkowski's
// book Test Examples for Nonlinear Programming Codes. It is their
// test problem Number 100.
KOKKOS_INLINE_FUNCTION
void HockSchittkowski100(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f =
        math::pow(x(0) - 10.0, 2.0) +
        5.0 * math::pow(x(1) - 12.0, 2.0) +
        math::pow(x(2), 4.0) +
        3.0 * math::pow(x(3) - 11.0, 2.0) +
        10.0 * math::pow(x(4), 6.0) +
        7.0 * math::pow(x(5), 2.0) +
        math::pow(x(6), 4.0) -
        4.0 * x(5) * x(6) - 10.0 * x(5) - 8.0 * x(6);
    con(0) = 127.0 -
        2.0 * math::pow(x(0), 2.0) -
        3.0 * math::pow(x(1), 4.0) -
        x(2) -
        4.0 * math::pow(x(3), 2.0) -
        5.0 * x(4);
    con(1) = 282.0 -
        7.0 * x(0) -
        3.0 * x(1) -
        10.0 * math::pow(x(2), 2.0) -
        x(3) +
        x(4);
    con(2) = 196.0 -
        23.0 * x(0) -
        math::pow(x(1), 2.0) -
        6.0 * math::pow(x(5), 2.0) +
        8.0 * x(6);
    con(3) =
        -4.0 * math::pow(x(0), 2.0) -
        math::pow(x(1), 2.0) +
        3.0 * x(0) * x(1) -
        2.0 * math::pow(x(2), 2.0) -
        5.0 * x(5) +
        11.0 * x(6);
}

TEST(unit_tests, HockSchittkowski100) {
    Kokkos::ScopeGuard kokkos;

    int n=7;
    int m=4;
    Kokkos::View<double*> x("HockSchittkowski100::x", n);
    Kokkos::View<double*> w("HockSchittkowski100::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("HockSchittkowski100::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "HockSchittkowski100::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            HockSchittkowski100
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - 2.330499 , 2.0);
    l2_error += std::pow(h_x(1) - 1.951372 , 2.0);
    l2_error += std::pow(h_x(2) + 0.4775414, 2.0);
    l2_error += std::pow(h_x(3) - 4.365726 , 2.0);
    l2_error += std::pow(h_x(4) + 0.624487 , 2.0);
    l2_error += std::pow(h_x(5) - 1.038131 , 2.0);
    l2_error += std::pow(h_x(6) - 1.594227 , 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

// This problem is taken from page 415 of Luenberger's book Applied
// Nonlinear Programming. It is to maximize the area of a hexagon of
// unit diameter.
KOKKOS_INLINE_FUNCTION
void HexagonArea(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = -0.5 * (
        x(0) * x(3) - x(1) * x(2) + x(2) * x(8) - x(4) * x(8) + x(4) * x(7) - x(5) * x(6)
    );
    con(0) = 1.0 - math::pow(x(2), 2.0) - math::pow(x(3), 2.0);
    con(1) = 1.0 - math::pow(x(8), 2.0);
    con(2) = 1.0 - math::pow(x(4), 2.0) - math::pow(x(5), 2.0);
    con(3) = 1.0 - math::pow(x(0), 2.0) - math::pow(x(1) - x(8), 2.0);
    con(4) = 1.0 - math::pow(x(0) - x(4), 2.0) - math::pow(x(1) - x(5), 2.0);
    con(5) = 1.0 - math::pow(x(0) - x(6), 2.0) - math::pow(x(1) - x(7), 2.0);
    con(6) = 1.0 - math::pow(x(2) - x(4), 2.0) - math::pow(x(3) - x(5), 2.0);
    con(7) = 1.0 - math::pow(x(2) - x(6), 2.0) - math::pow(x(3) - x(7), 2.0);
    con(8) = 1.0 - math::pow(x(6), 2.0) - math::pow(x(7) - x(8), 2.0);
    con(9) = x(0) * x(3) - x(1) * x(2);
    con(10) = x(2) * x(8);
    con(11) = - x(4) * x(8);
    con(12) = x(4) * x(7) - x(5) * x(6);
    con(13) = x(8);
}

TEST(unit_tests, HexagonArea) {
    Kokkos::ScopeGuard kokkos;

    int n=9;
    int m=14;
    Kokkos::View<double*> x("HexagonArea::x", n);
    Kokkos::View<double*> w("HexagonArea::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("HexagonArea::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "HexagonArea::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            HexagonArea
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double tempa = h_x(0) + h_x(2) + h_x(4) + h_x(6);
    double tempb = h_x(1) + h_x(3) + h_x(5) + h_x(7);
    double tempc = 0.5 / std::sqrt(tempa*tempa + tempb*tempb);
    double tempd = tempc * std::sqrt(3.0);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - (tempd*tempa + tempc*tempb), 2.0);
    l2_error += std::pow(h_x(1) - (tempd*tempb - tempc*tempa), 2.0);
    l2_error += std::pow(h_x(2) - (tempd*tempa - tempc*tempb), 2.0);
    l2_error += std::pow(h_x(3) - (tempd*tempb + tempc*tempa), 2.0);
    l2_error += std::pow(h_x(4) - (tempd*tempa + tempc*tempb), 2.0);
    l2_error += std::pow(h_x(5) - (tempd*tempb - tempc*tempa), 2.0);
    l2_error += std::pow(h_x(6) - (tempd*tempa - tempc*tempb), 2.0);
    l2_error += std::pow(h_x(7) - (tempd*tempb + tempc*tempa), 2.0);
    l2_error += std::pow(h_x(8)                              , 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}
