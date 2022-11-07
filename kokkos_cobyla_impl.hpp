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

#ifndef KOKKOS_COBYLA_IMPL_HPP
#define KOKKOS_COBYLA_IMPL_HPP

#include "kokkos_cobyla.hpp"

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_INLINE_FUNCTION
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
) {
    using SizeType = typename ScalarWorkViewType::size_type;
    SizeType mpp = m+2;
    SizeType isim = mpp;
    SizeType isimi = isim + n*n + n;
    SizeType idatm = isimi + n*n;
    SizeType ia = idatm + n*mpp + mpp;
    SizeType ivsig = ia + m*n + n;
    SizeType iveta = ivsig + n;
    SizeType isigb = iveta + n;
    SizeType idx = isigb + n;
    SizeType iwork = idx + n;
    SizeType total_size = n*(3*n + 2*m + 11) + 4*m + 6;

    cobylb(
        n,
        m,
        mpp,
        x,
        rhobeg,
        rhoend,
        // iprint,
        maxfun,
        Kokkos::subview(w, Kokkos::make_pair(0, isim)),
        Kokkos::subview(w, Kokkos::make_pair(isim, isimi)),
        Kokkos::subview(w, Kokkos::make_pair(isimi, idatm)),
        Kokkos::subview(w, Kokkos::make_pair(idatm, ia)),
        Kokkos::subview(w, Kokkos::make_pair(ia, ivsig)),
        Kokkos::subview(w, Kokkos::make_pair(ivsig, iveta)),
        Kokkos::subview(w, Kokkos::make_pair(iveta, isigb)),
        Kokkos::subview(w, Kokkos::make_pair(isigb, idx)),
        Kokkos::subview(w, Kokkos::make_pair(idx, iwork)),
        Kokkos::subview(w, Kokkos::make_pair(iwork, total_size)),
        iact
    );
}

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_INLINE_FUNCTION
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
    ScalarWorkViewType sim_flat,
    ScalarWorkViewType simi_flat,
    ScalarWorkViewType datmat_flat,
    ScalarWorkViewType a_flat,
    ScalarWorkViewType vsig,
    ScalarWorkViewType veta,
    ScalarWorkViewType sigbar,
    ScalarWorkViewType dx,
    ScalarWorkViewType w,
    IntegralWorkViewType iact
) {
    // Wrap unmanaged Views around the flattened Views
    // so that we can do 2D indexing.
    Kokkos::View<
        ScalarWorkViewType::value_type**,
        ScalarWorkViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    >
    sim(sim_flat.data(), n, n+1),
    simi(simi_flat.data(), n, n),
    datmat(datmat_flat.data(), mpp, n+1),
    a(a_flat.data(), n, m+1);

    namespace math =
#if KOKKOS_VERSION < 30700
        Kokkos;
#else
        Kokkos::Experimental;
#endif

    // Set the initial values of some parameters. The last column of SIM holds
    // the optimal vertex of the current simplex, and the preceding N columns
    // hold the displacements from the optimal vertex to the other vertices.
    // Further, SIMI holds the inverse of the matrix that is contained in the
    // first N columns of SIM.
    IntegralType iptem = n < 5 ? n : 5;
    IntegralType iptemp = iptem + 1;
    IntegralType np = n + 1;
    IntegralType mp = m + 1;
    ScalarType alpha = 0.25;
    ScalarType beta  = 2.1;
    ScalarType gamma = 0.5;
    ScalarType delta = 1.1;
    ScalarType rho = rhobeg;
    ScalarType parmu = 0.0;

    IntegralType nfvals = 0;
    ScalarType temp = 1.0/rho;
    for (IntegralType i=0; i<n; i++) {
        sim(i, n) = x(i);
        for (IntegralType j=0; j<n; j++) {
            simi(i,j) = 0.0;
        }
        sim(i,i) = rho;
        simi(i,i) = temp;
    }
    IntegralType jdrop = np;
    IntegralType ibrnch = 0;

    // Make the next call of the user-supplied subroutine CALCFC. These
    // instructions are also used for calling CALCFC during the iterations of
    // the algorithm.
line_40:
    if (nfvals >= maxfun and nfvals > 0) {
        goto line_600;
    }
    nfvals = nfvals + 1;
    // LTM do CALCFC call here
    ScalarType resmax = 0.0;
    if (m > 0) {
        for (IntegralType k=0; k<m; k++) {
            resmax = math::fmax(resmax, -con(k));
        }
    }
    con(m) = f;
    con(m+1) = resmax;
    if (ibrnch == 1) {
        goto line_440;
    }

    // Set the recently calculated function values in a column of DATMAT. This
    // array has a column for each vertex of the current simplex, the entries of
    // each column being the values of the constraint functions (if any)
    // followed by the objective function and the greatest constraint violation
    // at the vertex.
    for (IntegralType k=0; k<mpp; k++) {
        datmat(k, jdrop-1) = con(k);
    }
    if (nfvals > np) {
        goto line_130;
    }

    // Exchange the new vertex of the initial simplex with the optimal vertex if
    // necessary. Then, if the initial simplex is not complete, pick its next
    // vertex and calculate the function values there.
    if (jdrop <= n) {
        if (datmat(m,n) <= f) {
            x(jdrop-1) = sim(jdrop-1, n);
        }
        else {
            sim(jdrop-1, n) = x(jdrop-1);
            for (IntegralType k=0; k<mpp; k++) {
                datmat(k, jdrop-1) = datmat(k, n);
                datmat(k, n) = con(k);
            }
            for (IntegralType k=0; k<jdrop; k++) {
                sim(jdrop-1, k) = -rho;
                temp = 0.0;
                for (IntegralType i=k; i<jdrop; i++) {
                    temp = temp - simi(i,k);
                }
                simi(jdrop-1, k) = temp;
            }
        }
    }
    if (nfvals <= n) {
        jdrop = nfvals;
        x(jdrop-1) = x(jdrop-1) + rho;
        goto line_40;
    }
line_130:
    ibrnch=1;

    // Identify the optimal vertex of the current simplex.
line_140:
    ScalarType phimin = datmat(m, n) + parmu * datmat(m+1,n);
    IntegralType nbest = n;
    for (IntegralType j=0; j<n; j++) {
        temp = datmat(m,j) + parmu * datmat(m+1, j);
        if (temp < phimin) {
            nbest = j;
            phimin = temp;
        }
        else if (temp == phimin and parmu == 0.0) {
            if (datmat(m+1, j) < datmat(m+1, nbest)) {
                nbest = j;
            }
        }
    }

    // Switch the best vertex into pole position if it is not there already,
    // and also update SIM, SIMI and DATMAT.
    if (nbest < n) {
        for (IntegralType i=0; i<mpp; i++) {
            temp = datmat(i, n);
            datmat(i, n) = datmat(i, nbest);
            datmat(i, nbest) = temp;
        }
        for (IntegralType i=0; i<n; i++) {
            temp = sim(i, nbest);
            sim(i, nbest) = 0.0;
            sim(i, n) = sim(i, n) + temp;
            ScalarType tempa = 0.0;
            for (IntegralType k=0; k<n; k++) {
                sim(i, k) = sim(i, k) - temp;
                tempa = tempa - simi(k, i);
                simi(nbest, i) = tempa;
            }
        }
    }

    // Make an error return if SIGI is a poor approximation to the inverse of
    // the leading N by N submatrix of SIG.
    ScalarType error = 0.0;
    for (IntegralType i=0; i<n; i++) {
        for (IntegralType j=0; j<n; j++) {
            temp = 0.0;
            if (i == j) {
                temp = temp - 1.0;
            }
            for (IntegralType k=0; k<n; k++) {
                temp = temp + simi(i,k)*sim(k,j);
            }
            error = math::fmax(error, math::fabs(temp));
        }
    }
    if (error > 0.1) {
        goto line_600;
    }

    // Calculate the coefficients of the linear approximations to the objective
    // and constraint functions, placing minus the objective function gradient
    // after the constraint gradients in the array A. The vector W is used for
    // working space.
    for (IntegralType k=0; k<mp; k++) {
        con(k) = -datmat(k, n);
        for (IntegralType j=0; j<n; j++) {
            w(j) = datmat(k,j) + con(k);
        }
        for (IntegralType i=0; i<n; i++) {
            temp = 0.0;
            for (IntegralType j=0; j<n; j++) {
                temp = temp + w(j)*simi(j, i);
            }
            if (k == mp) {
                temp = -temp;
            }
            a(i,k) = temp;
        }
    }

    // Calculate the values of sigma and eta, and set IFLAG=0 if the current
    // simplex is not acceptable.
    IntegralType iflag = 1;
    ScalarType parsig = alpha*rho;
    ScalarType pareta = beta*rho;
    for (IntegralType j=0; j<n; j++) {
        ScalarType wsig = 0.0;
        ScalarType weta = 0.0;
        for (IntegralType i=0; i<n; i++) {
            wsig = wsig + simi(j,i)*simi(j,i);
            weta = weta + sim(i,j) *sim(i,j);
        }
        vsig(j) = 1.0 / math::sqrt(wsig);
        veta(j) = math::sqrt(weta);
        if (vsig(j) < parsig or veta(j) > pareta) {
            iflag = 0;
        }
    }

    // If a new vertex is needed to improve acceptability, then decide which
    // vertex to drop from the simplex.
    if (ibrnch == 1 or iflag == 1) {
        goto line_370;
    }
    jdrop = -1;
    temp = pareta;
    for (IntegralType j=0; j<n; j++) {
        if (veta(j) > temp) {
            jdrop = j;
            temp = veta(j);
        }
    }
    if (jdrop == -1) {
        for (IntegralType j=0; j<n; j++) {
            if (vsig(j) < temp) {
                jdrop = j;
                temp = vsig(j);
            }
        }
    }

    // Calculate the step to the new vertex and its sign.
    temp = gamma*rho*vsig(jdrop);
    for (IntegralType i=0; i<n; i++) {
        dx(i) = temp*simi(jdrop, i);
    }
    ScalarType cvmaxp = 0.0;
    ScalarType cvmaxm = 0.0;
    ScalarType sum = 0.0;
    for (IntegralType k=0; k<mp; k++) {
        sum = 0.0;
        for (IntegralType i=0; i<n; i++) {
            sum = sum + a(i,k)*dx(i);
        }
        if (k < m) {
            temp = datmat(k, n);
            cvmaxp = math::fmax(cvmaxp, -sum-temp);
            cvmaxm = math::fmax(cvmaxm,  sum-temp);
        }
    }
    ScalarType dxsign = 1.0;
    if (parmu*(cvmaxp-cvmaxm) > sum+sum) {
        dxsign = -1.0;
    }

    // Update the elements of SIM and SIMI, and set the next X.
    temp = 0.0;
    for (IntegralType i=0; i<n; i++) {
        dx(i) = dxsign*dx(i);
        sim(i,jdrop) = dx(i);
        temp = temp + simi(jdrop,i)*dx(i);
    }
    for (IntegralType i=0; i<n; i++) {
        simi(jdrop,i) = simi(jdrop,i)/temp;
    }
    for (IntegralType j=0; j<n; j++) {
        if (j != jdrop) {
            temp = 0.0;
            for (IntegralType i=0; i<n; i++) {
                temp = temp + simi(j,i)*dx(i);
            }
            for (IntegralType i=0; i<n; i++) {
                simi(j,i) = simi(j,i) - temp*simi(jdrop,i);
            }
        }
        x(j) = sim(j,n) + dx(j);
    }
    goto line_40;

    // Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO.
line_370:
    IntegralType izdota = n*n;
    IntegralType ivmc = izdota + n;
    IntegralType isdirn = ivmc + mp;
    IntegralType idxnew = isdirn + n;
    IntegralType ivmd = idxnew + n;
    IntegralType ifull = 0;
    trstlp(
        n,
        m,
        a_flat,
        con,
        rho,
        dx,
        ifull,
        iact,
        Kokkos::subview(w, Kokkos::make_pair(0, izdota)),
        Kokkos::subview(w, Kokkos::make_pair(izdota, ivmc)),
        Kokkos::subview(w, Kokkos::make_pair(ivmc, isdirn)),
        Kokkos::subview(w, Kokkos::make_pair(isdirn, idxnew)),
        Kokkos::subview(w, Kokkos::make_pair(idxnew, ivmd)),
        Kokkos::subview(w, Kokkos::make_pair(ivmd, ivmd+m))
    );
    if (ifull == 0) {
        temp = 0.0;
        for (IntegralType i=0; i<n; i++) {
            temp = temp + dx(i)*dx(i);
        }
        if (temp < 0.25 * rho * rho) {
            ibrnch=1;
            goto line_550;
        }
    }

    // Predict the change to F and the new maximum constraint violation if the
    // variables are altered from x(0) to x(0)+DX.
    ScalarType resnew = 0.0;
    con(m) = 0.0;
    for (IntegralType k=0; k<mp; k++) {
        sum = con(k);
        for (IntegralType i=0; i<n; i++) {
            sum = sum - a(i,k)*dx(i);
        }
        if (k < m) {
            resnew = math::fmax(resnew, sum);
        }
    }

    // Increase PARMU if necessary and branch back if this change alters the
    // optimal vertex. Otherwise PREREM and PREREC will be set to the predicted
    // reductions in the merit function and the maximum constraint violation
    // respectively.
    ScalarType barmu = 0.0;
    ScalarType prerec = datmat(mp, n) - resnew;
    if (prerec > 0.0) {
        barmu = sum/prerec;
    }
    if (parmu < 1.5 * barmu) {
        parmu = 2.0*barmu;
        ScalarType phi = datmat(m,n) + parmu*datmat(m+1,n);
        for (IntegralType j=0; j<n; j++) {
            temp = datmat(m,j) + parmu*datmat(m+1,j);
            if (temp < phi) {
                goto line_140;
            }
            if (temp == phi and parmu == 0.0) {
                if (datmat(m+1,j) < datmat(m+1,n)) {
                    goto line_140;
                }
            }
        }
    }
    ScalarType prerem = parmu*prerec - sum;

    // Calculate the constraint and objective functions at x(*). Then find the
    // actual reduction in the merit function.
    for (IntegralType i=0; i<n; i++) {
        x(i) = sim(i,n) + dx(i);
    }
    ibrnch=1;
    goto line_40;
line_440:
    ScalarType vmold = datmat(m,n) + parmu*datmat(m+1,n);
    ScalarType mvnew = f + parmu*resmax;
    ScalarType trured = vmold - vmnew;
    if (parmu == 0.0 and f == datmat(m,n)) {
        prerem = prerec;
        trured = datmat(m+1,n) - resmax;
    }

    // Begin the operations that decide whether x(*) should replace one of the
    // vertices of the current simplex, the change being mandatory if TRURED is
    // positive. Firstly, JDROP is set to the index of the vertex that is to be
    // replaced.
    ScalarType ratio = 0.0;
    if (trured <= 0.0) {
        ratio = 1.0;
    }
    jdrop = -1;
    for (IntegralType j=0; j<n; j++) {
        temp = 0.0;
        for (IntegralType i=0; i<n; i++) {
            temp = temp + simi(j,i)*dx(i);
        }
        temp = math::fabs(temp);
        if (temp > ratio) {
            jdrop = j;
            ratio = temp;
        }
        sigbar(j) = temp*vsig(j);
    }

    // Calculate the value of ell.
    ScalarType edgmax = delta * rho;
    IntegralType l=-1;
    for (IntegralType j=0; j<n; j++) {
        if (sigbar(j) >= parsig or sigbar(j) >= vsig(j)) {
            temp = veta(j);
            if (trured > 0.0) {
                temp = 0.0;
                for (IntegralType i=0; i<n; i++) {
                    temp = temp + math::pow(dx(i) - sim(i,j), 2.0);
                }
                temp = math::sqrt(temp);
            }
            if (temp > edgmax) {
                l = j;
                edgmax = temp;
            }
        }
    }
    if (l >= 0) {
        jdrop = l;
    }
    if (jdrop == -1) {
        goto line_550;
    }

    // Revise the simplex by updating the elements of SIM, SIMI and DATMAT.
    temp = 0.0;
    for (IntegralType i=0; i<n; i++) {
        sim(i, jdrop) = dx(i);
        temp = temp + simi(jdrop,i) * dx(i);
    }
    for (IntegralType i=0; i<n; i++) {
        simi(jdrop,i) = simi(jdrop,i) / temp;
    }
    for (IntegralType j=0; j<n; j++) {
        if (j != jdrop) {
            temp = 0.0;
            for (IntegralType i=0; i<n; i++) {
                temp = temp + simi(j,i) * dx(i);
            }
            for (IntegralType i=0; i<n; i++) {
                simi(j,i) = simi(j,i) - temp * simi(jdrop,i);
            }
        }
    }
    for (IntegralType k=0; k<mpp; k++) {
        datmat(k,jdrop) = con(k);
    }

    // Branch back for further iterations with the current RHO.
    if (trured > 0.0 and trured >= 0.1*prerem) {
        goto line_140;
    }
line_550:
    if (iflag == 0) {
        ibrnch = 0;
        goto line_140;
    }

    // Otherwise reduce RHO if it is not at its least value and reset PARMU.
    if (rho > rhoend) {
        rho = 0.5*rho;
        if (rho <= 1.5*rhoend) {
            rho = rhoend;
        }
        if (parmu > 0.0) {
            ScalarType denom = 0.0;
            ScalarType cmin, cmax;
            for (IntegralType k=0; k<mp; k++) {
                cmin = datmat(k,n);
                cmax = cmin;
                for (IntegralType i=0; i<n; i++) {
                    cmin = math::fmin(cmin, datmat(k,i));
                    cmax = math::fmax(cmax, datmat(k,i));
                }
                if (k < m and cmin < 0.5*cmax) {
                    temp = math::fmax(cmax, 0.0) - cmin;
                    if (denom <= 0.0) {
                        denom = temp;
                    }
                    else {
                        denom = math::fmin(denom, temp);
                    }
                }
            }
            if (denom == 0.0) {
                parmu = 0.0
            }
            else if (cmax-cmin < parmu*denom) {
                parmu = (cmax-cmin)/denom;
            }
        }
        goto line_140;
    }

    // Return the best calculated values of the variables.
    if (ifull == 1) goto line_620;
line_600:
    for (IntegralType i=0; i<n; i++) {
        x(i) = sim(i, n);
    }
    f = datmat(m,n);
    resmax = datmat(mp,n);
line_620:
    maxfun = nfvals;
    return;
}

template <
    typename IntegralType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_FUNCTION
void trstlp(
    IntegralType n,
    IntegralType m,
    ScalarWorkViewType a_flat,
    ScalarWorkViewType b,
    ScalarType rho,
    ScalarWorkViewType dx,
    IntegralType &ifull,
    IntegralWorkViewType iact,
    ScalarWorkViewType z_flat,
    ScalarWorkViewType zdota,
    ScalarWorkViewType vmultc,
    ScalarWorkViewType sdirn,
    ScalarWorkViewType dxnew,
    ScalarWorkViewType vmultd
) {
    // This subroutine calculates an N-component vector DX by applying the
    // following two stages. In the first stage, DX is set to the shortest
    // vector that minimizes the greatest violation of the constraints
    // A(1,K)*DX(1)+A(2,K)*DX(2)+...+A(N,K)*DX(N) .GE. B(K), K=2,3,...,M,
    // subject to the Euclidean length of DX being at most RHO. If its length is
    // strictly less than RHO, then we use the resultant freedom in DX to
    // minimize the objective function
    //         -A(1,M+1)*DX(1)-A(2,M+1)*DX(2)-...-A(N,M+1)*DX(N)
    // subject to no increase in any greatest constraint violation. This
    // notation allows the gradient of the objective function to be regarded as
    // the gradient of a constraint. Therefore the two stages are distinguished
    // by MCON .EQ. M and MCON .GT. M respectively. It is possible that a
    // degeneracy may prevent DX from attaining the target length RHO. Then the
    // value IFULL=0 would be set, but usually IFULL=1 on return.

    // In general NACT is the number of constraints in the active set and
    // IACT(1),...,IACT(NACT) are their indices, while the remainder of IACT
    // contains a permutation of the remaining constraint indices. Further, Z is
    // an orthogonal matrix whose first NACT columns can be regarded as the
    // result of Gram-Schmidt applied to the active constraint gradients. For
    // J=1,2,...,NACT, the number ZDOTA(J) is the scalar product of the J-th
    // column of Z with the gradient of the J-th active constraint. DX is the
    // current vector of variables and here the residuals of the active
    // constraints should be zero. Further, the active constraints have
    // nonnegative Lagrange multipliers that are held at the beginning of
    // VMULTC. The remainder of this vector holds the residuals of the inactive
    // constraints at DX, the ordering of the components of VMULTC being in
    // agreement with the permutation of the indices of the constraints that is
    // in IACT. All these residuals are nonnegative, which is achieved by the
    // shift RESMAX that makes the least residual zero.all the active
    // constraint violations by one simultaneously.

    namespace math =
#if KOKKOS_VERSION < 30700
        Kokkos;
#else
        Kokkos::Experimental;
#endif

    // LTM Wrap unmanaged Views around the flattened Views
    // so that we can do 2D indexing.
    Kokkos::View<
        ScalarWorkViewType::value_type**,
        ScalarWorkViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    >
    a(a_flat.data(), n, m+1),
    z(z_flat.data(), n, n);

    // Initialize Z and some other variables. The value of RESMAX will be
    // appropriate to DX=0, while ICON will be the index of a most violated
    // constraint if RESMAX is positive. Usually during the first stage the
    // vector SDIRN gives a search direction that reduces
    ifull = 1;
    IntegralType mcon = m-1;
    IntegralType nact = -1;
    ScalarType resmax = 0.0;
    IntegralType icon = -1;
    for (IntegralType i=0; i<n; i++) {
        for (IntegralType j=0; j<n; j++) {
            z(i,j) = 0.0;
        }
        z(i,i) = 1.0;
        dx(i) = 0.0;
    }
    if (m >= 1) {
        for (IntegralType k=0; k<m; k++) {
            if (b(k) > resmax) {
                resmax = b(k)
                icon = k;
            }
        }
        for (IntegralType k=0; k<m; k++) {
            iact(k) = k;
            vmultc(k) = resmax - b(k);
        }
    }
    if (resmax == 0.0) goto line_480;
    for (IntegralType i=0; i<n; i++) {
        sdirn(i) = 0.0;
    }

    // End the current stage of the calculation if 3 consecutive iterations
    // have either failed to reduce the best calculated value of the objective
    // function or to increase the number of active constraints since the best
    // value was calculated. This strategy prevents cycling, but there is a
    // remote possibility that it will cause premature termination.
line_60:
    ScalarType optold = 0.0;
    ScalarType optnew;
    IntegralType icount = 0;
    if (mcon == m-1) {
        optnew = resmax;
    }
    else {
        optnew = 0.0;
        for (IntegralType i=0; i<n; i++) {
            optnew = optnew - dx(i)*a(i,mcon);
        }
    }
    IntegralType nactx = 0;
    if (icount == 0 or optnew < optold) {
        optold = optnew;
        nactx = nact;
        icount = 3;
    }
    else if (nact > nactx) {
        nactx = nact;
        icount = 3;
    }
    else {
        icount = icount-1;
        if (icount == 0) goto line_490;
    }

    // If ICON exceeds NACT, then we add the constraint with index IACT(ICON) to
    // the active set. Apply Givens rotations so that the last N-NACT-1 columns
    // of Z are orthogonal to the gradient of the new constraint, a scalar
    // product being set to zero if its nonzero value could be due to computer
    // rounding errors. The array DXNEW is used for working space.
    if (icon <= nact) goto line_260;
    IntegralType kk = iact(icon);
    for (IntegralType i=0; i<n; i++) {
        dxnew(i) = a(i,kk);
    }
    ScalarType tot = 0.0;
    IntegralType k = n-1;
    IntegralType kp;
    ScalarType temp;
    ScalarType alpha;
    ScalarType beta;
line_100:
    if (k > nact) {
        ScalarType sp = 0.0;
        ScalarType spabs = 0.0;
        for (IntegralType i=0; i<n; i++) {
            temp = z(i,k)*dxnew(i);
            sp = sp + temp;
            spabs = spabs + math::fabs(temp);
        }
        ScalarType acca = spabs + 0.1 * math::fabs(sp);
        ScalarType accb = spabs + 0.2 * math::fabs(sp);
        if (spabs >= acca or acca >= accb) sp = 0.0;
        if (tot == 0.0) {
            tot = sp;
        }
        else {
            kp = k+1;
            temp = math::sqrt(sp*sp + tot*tot);
            alpha = sp/temp;
            beta = tot/temp;
            tot = temp;
            for (IntegralType i=0; i<n; i++) {
                temp = alpha * z(i,k) + beta * z(i,kp);
                z(i,kp) = alpha * z(i,kp) - beta * z(i,k);
                z(i,k) = temp;
            }
        }
        k = k-1;
        goto line_100;
    }

    // Add the new constraint if this can be done without a deletion from the
    // active set.
    if (tot != 0.0) {
        nact = nact+1;
        zdota(nact) = tot;
        vmultc(icon) = vmultc(nact);
        vmultc(nact) = 0.0;
        goto line_210;
    }

    // The next instruction is reached if a deletion has to be made from the
    // active set in order to make room for the new active constraint, because
    // the new constraint gradient is a linear combination of the gradients of
    // the old active constraints. Set the elements of VMULTD to the multipliers
    // of the linear combination. Further, set IOUT to the index of the
    // constraint to be deleted, but branch if no suitable index can be found.
    ScalarType ratio = -1.0;
    k = nact;
line_130:
    ScalarType zdotv = 0.0;
    ScalarType zdvabs = 0.0;
    IntegralType kw;
    for (IntegralType i=0; i<n; i++) {
        temp = z(i,k)*dxnew(i);
        zdotv = zdotv + temp;
        zdvabs = zdvabs + math::fabs(temp);
    }
    acca = zdvabs + 0.1*math::fabs(zdotv);
    accb = zdvabs + 0.2*math::fabs(zdotv);
    if (zdvabs < acca and acca < accb) {
        temp = zdotv/zdota(k);
        if (temp > 0.0 and iact(k) < m) {
            tempa = vmultc(k)/temp;
            if (ratio < 0.0 or tempa < ratio) {
                ratio = tempa;
                iout = k;
            }
        }
        if (k >= 2) {
            kw = iact(k);
            for (IntegralType i=0; i<n; i++) {
                dxnew(i) = dxnew(i) - temp*a(i,kw);
            }
        }
        vmultd(k) = temp;
    }
    else {
        vmultd(k) = 0.0
    }
    k = k-1;
    if (k > 0) goto line_130;
    if (ratio < 0.0) goto line_490;

    // Revise the Lagrange multipliers and reorder the active constraints so
    // that the one to be replaced is at the end of the list. Also calculate the
    // new value of ZDOTA(NACT) and branch if it is not acceptable.
    for (IntegralType k=0; k<nact; k++) {
        vmultc(k) = math::fmax(0.0, vmultc(k) - ratio*vmultd(k));
    }
    if (icon < nact) {
        IntegralType isave = iact(icon);
        ScalarType vsave = vmultc(icon);
        k = icon;
line_170:
        IntegralType kp = k + 1;
        kw = iact(kp);
        ScalarType sp = 0.0;
        for (IntegralType i=0; i<n; i++) {
            sp = sp + z(i,k) * a(i,kw);
        }
        temp = math::sqrt(sp*sp + zdota(kp)*zdota(kp));
        ScalarType alpha = zdota(kp)/temp;
        ScalarType beta = sp/temp;
        zdota(kp) = alpha*zdota(k);
        zdota(k) = temp;
        for (IntegralType i=0; i<n; i++) {
            temp = alpha*z(i,kp) + beta*z(i,k);
            z(i,kp) = alpha*z(i,k) - beta*z(i,kp);
            z(i,k) = temp;
        }
        iact(k) = kw;
        vmultc(k) = vmultc(kp);
        k = kp;
        if (k < nact-1) goto line_170;
        iact(k) = isave;
        vmultc(k) = vsave;
    }
    temp = 0.0;
    for (IntegralType i=0; i<n; i++) {
        temp = temp + z(i,nact)*a(i,kk);
    }
    if (temp == 0.0) goto line_490;
    zdota(nact) = temp;
    vmultc(icon) = 0.0;
    vmultc(nact) = ratio;

    // Update IACT and ensure that the objective function continues to be
    // treated as the last active constraint when MCON>M.
line_210:
    iact(icon) = iact(nact);
    iact(nact) = kk;
    if (mcon > m and kk != mcon) {
        k = nact-1;
        sp = 0.0;
        for (IntegralType i=0; i<n; i++) {
            sp = sp + z(i,k) * a(i,kk);
        }
        temp = math::sqrt(sp*sp + zdota(nact)*zdota(nact));
        ScalarType alpha = zdota(nact) / temp;
        ScalarType beta = sp / temp;
        zdota(nact) = alpha*zdota(k);
        zdota(k) = temp;
        for (IntegralType i=0; i<n; i++) {
            temp = alpha * z(i,nact) + beta*z(i,k);
            z(i,nact) = alpha*z(i,k) - beta*z(i,nact);
            z(i,k) = temp;
        }
        iact(nact) = iact(k);
        iact(k) = kk;
        temp = vmultc(k);
        vmultc(k) = vmultc(nact);
        vmultc(nact) = temp;
    }

    // If stage one is in progress, then set SDIRN to the direction of the next
    // change to the current vector of variables.
    if (mcon > m) goto line_320;
    kk = iact(nact);
    temp = 0.0;
    for (IntegralType i=0; i<n; i++) {
        temp = temp + sdirn(i)*a(i,kk);
    }
    temp = temp - 1.0;
    temp = temp / zdota(nact);
    for (IntegralType i=0; i<n; i++) {
        sdirn(i) = sdirn(i) - temp*z(i,nact);
    }
    goto line_340;

    // Delete the constraint that has the index IACT(ICON) from the active set.
line_260:
    if (icon < nact) {
        isave = iact(icon);
        vsave = vmultc(icon);
        k = icon;
line_270:
        kp = k + 1;
        kk = iact(kp);
        sp = 0.0;
        for (IntegralType i=0; i<n; i++) {
            sp = sp + z(i,k)*a(i,kk);
        }
        temp = math::sqrt(sp*sp + zdota(kp)*zdota(kp));
        alpha = zdota(kp) / temp;
        beta = sp / temp;
        zdota(kp) = alpha*zdota(k);
        zdota(k) = temp;
        for (IntegralType i=0; i<n; i++) {
            temp = alpha*z(i,kp) + beta*z(i,k);
            z(i,kp) = alpha*z(i,k) - beta*z(i,kp);
            z(i,k) = temp;
        }
        iact(k) = kk;
        vmultc(k) = vmultc(kp);
        k = kp;
        if (k < nact) goto line_270;
        iact(k) = isave;
        vmultc(k) = vsave;
    }
    nact = nact - 1;

    // If stage one is in progress, then set SDIRN to the direction of the next
    // change to the current vector of variables.
    if (mcon > m) goto line_320;
    temp = 0.0;
    for (IntegralType i=0; i<n; i++) {
        temp = temp + sdirn(i)*z(i, nact+1);
    }
    for (IntegralType i=0; i<n; i++) {
        sdirn(i) = sdirn(i) - temp*z(i, nact+1);
    }
    goto line_340;

    // Pick the next search direction of stage two.
line_320:
    temp = 1.0 / zdota(nact);
    for (IntegralType i=0; i<n; i++) {
        sdirn(i) = temp*z(i, nact);
    }

    // Calculate the step to the boundary of the trust region or take the step
    // that reduces RESMAX to zero. The two statements below that include the
    // factor 1.0E-6 prevent some harmless underflows that occurred in a test
    // calculation. Further, we skip the step if it could be zero within a
    // reasonable tolerance for computer rounding errors.
line_340:
    ScalarType dd = rho*rho;
    ScalarType sd = 0.0;
    ScalarType ss = 0.0;
    for (IntegralType i=0; i<n; i++) {
        if (math::fabs(dx(i)) >= 1.0e-6*rho) dd = dd - math::pow(dx(i), 2.0);
        sd = sd + dx(i)*sdirn(i);
        ss = ss + math::pow(sdirn(i), 2.0);
    }
    if (dd <= 0.0) goto line_490;
    temp = math::sqrt(ss*dd);
    if (math::fabs(sd) >= 1.0e-6*temp) temp = math::sqrt(ss*dd + sd*sd);
    ScalarType stpful = dd / (temp + sd);
    ScalarType step = stpful;
    if (mcon == m) {
        acca = step + 0.1*resmax;
        accb = step + 0.2*resmax;
        if (step >= acca or acca >= accb) goto line_480;
        step = math::fmin(step, resmax);
    }

    // Set DXNEW to the new variables if STEP is the steplength, and reduce
    // RESMAX to the corresponding maximum residual if stage one is being done.
    // Because DXNEW will be changed during the calculation of some Lagrange
    // multipliers, it will be restored to the following value later.
    for (IntegralType i=0; i<n; i++) {
        dxnew(i) = dx(i) + step*sdirn(i);
    }
    if (mcon == m) {
        resold = resmax;
        resmax = 0.0;
        for (IntegralType k=0; k<nact; k++) {
            kk = iact(k);
            temp = b(kk);
            for (IntegralType i=0; i<n; i++) {
                temp = temp - a(i,kk)*dxnew(i);
            }
            resmax = math::fmax(resmax, temp);
        }
    }

    // Set VMULTD to the VMULTC vector that would occur if DX became DXNEW. A
    // device is included to force VMULTD(K)=0.0 if deviations from this value
    // can be attributed to computer rounding errors. First calculate the new
    // Lagrange multipliers.
    k = nact;
line_390:
    ScalarType zdotw = 0.0;
    ScalarType zdwabs = 0.0;
    for (IntegralType i=0; i<n; i++) {
        temp = z(i,k)*dxnew(i);
        zdotw = zdotw + temp;
        zdwabs = zdwabs + math::fabs(temp);
    }
    acca = zdwabs + 0.1*math::fabs(zdotw);
    accb = zdwabs + 0.2*math::fabs(zdotw);
    if (zdwabs >= acca or acca >= accb) zdotw = 0.0;
    vmultd(k) = zdotw/zdota(k);
    if (k >= 2) {
        kk = iact(k);
        for (IntegralType i=0; i<n; i++) {
            dxnew(i) = dxnew(i) - vmultd(k)*a(i,kk);
        }
        k = k-1;
        goto line_390;
    }
    if (mcon > m) vmultd(nact) = math::fmax(0.0, vmultd(nact));

    // Complete VMULTC by finding the new constraint residuals.
    for (IntegralType i=0; i<n; i++) {
        dxnew(i) = dx(i) + step*sdirn(i);
    }
    if (mcon > nact) {
        kl = nact + 1;
        for (IntegralType k=kl; k<=mcon; kl++) {
            kk = iact(k);
            sum = resmax-b(kk);
            sumabs = resmax + math::fabs(b(kk));
            for (IntegralType i=0; i<n; i++) {
                temp = a(i,kk)*dxnew(i);
                sum = sum + temp;
                sumabs = sumabs + math::fabs(temp);
            }
            acca = sumabs + 0.1*math::fabs(sum);
            accb = sumabs + 0.2*math::fabs(sum);
            if (sumabs >= acca or acca >= accb) sum = 0.0;
            vmultd(k) = sum;
        }
    }

    // Calculate the fraction of the step from DX to DXNEW that will be taken.
    ratio = 1.0;
    icon = 0;
    for (IntegralType k=0; k<mcon; k++) {
        if (vmultd(k) < 0.0) {
            temp = vmultc(k)/(vmultc(k) - vmultd(k));
            if (temp < ratio) {
                ratio = temp;
                icon = k;
            }
        }
    }

    // Update DX, VMULTC and RESMAX.
    temp = 1.0 - ratio;
    for (IntegralType i=0; i<n; i++) {
        dx(i) = temp*dx(i) + ratio*dxnew(i);
    }
    for (IntegralType k=0; k<mcon; k++) {
        vmultc(k) = math::fmax(0.0, temp*vmultc(k) + ratio*vmultd(k));
    }
    if (mcon == m) resmax = resold + ratio*(resmax - resold);

    // If the full step is not acceptable then begin another iteration.
    // Otherwise switch to stage two or end the calculation.
    if (icon > 0) goto line_70;
    if (step == stpful) goto line_500;
line_480:
    mcon = m + 1;
    icon = mcon;
    iact(mcon) = mcon;
    vmultc(mcon) = 0.0;
    goto line_60;

    // We employ any freedom that may be available to reduce the objective
    // function before returning a DX whose length is less than RHO.
line_490:
    if (mcon == m) goto line_480;
    ifull = 0;
line_500:
    return;
}

#endif // KOKKOS_COBYLA_IMPL_HPP
