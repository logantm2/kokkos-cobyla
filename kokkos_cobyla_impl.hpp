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
    trstpl(
        n,
        m,
        a,
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

#endif // KOKKOS_COBYLA_IMPL_HPP
