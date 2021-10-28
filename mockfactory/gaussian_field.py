import logging

import numpy as np

from pmesh.pm import RealField, ComplexField, ParticleMesh
from nbodykit.mpirng import MPIRandomState
from nbodykit import CurrentMPIComm
from nbodykit.source.mesh.field import FieldMesh
import mpsort


def _get_los(los):
    if isinstance(los,str):
        los = 'xyz'.index(los)
    if np.ndim(los) == 0:
        ilos = los
        los = np.zeros(3,dtype=self.dtype)
        los[ilos] = 1.
    los = np.asarray(los)
    return los


def cartesian_to_sky(position, wrap=True, degree=True):
    """Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (3,N)
        position in cartesian coordinates.
    wrap : bool, optional
        whether to wrap ra into [0,2*pi]
    degree : bool, optional
        whether RA, Dec are in degree (True) or radian (False).

    Returns
    -------
    dist : array
        distance.
    ra : array
        RA.
    dec : array
        Dec.
    """
    dist = sum(pos**2 for pos in position)**0.5
    ra = np.arctan2(position[1],position[0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(position[2]/dist)
    conversion = np.pi/180. if degree else 1.
    return dist, ra/conversion, dec/conversion


class GaussianFieldMesh(object):

    logger = logging.getLogger("GaussianFieldMesh")
    """
    A MeshSource object that generates a :class:`~pmesh.pm.RealField` density
    mesh from a linear power spectrum function :math:`P(k)`.

    Parameters
    ----------
    Plin: callable
        the callable linear power spectrum function, which takes the
        wavenumber as its single argument
    BoxSize : float, 3-vector of floats
        the size of the box to generate the grid on
    Nmesh : int, 3-vector of int
        the number of the mesh cells per side
    seed : int, optional
        the global random seed, used to set the seeds across all ranks
    unitary_amplitude: bool, optional
        ``True`` to remove variance from the complex field by fixing the
        amplitude to :math:`P(k)` and only the phase is random.
    inverted_phase : bool, optional
        ``True`` to invert phase of the complex field by fixing the
        amplitude to :math:`P(k)` and only the phase is random.
    los : 'x', 'y', 'z'; int, 3-vector of int, optional
        los :math:`\hat{\eta}` used to paint anisotropic power spectrum
        if provided, ``Plin`` should depend :math:`(k,\hat{k} \cdot \hat{\eta})`.
    comm : MPI communicator
        the MPI communicator
    """
    @CurrentMPIComm.enable
    def __init__(self, Plin, BoxSize, Nmesh, seed=None,
                unitary_amplitude=False,
                inverted_phase=False,
                los=None,
                dtype='f4',
                comm=None):

        self.Plin = Plin
        # cosmology and communicator
        self.comm    = comm
        # set the seed randomly if it is None
        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs = {}
        self.attrs['seed'] = seed
        self.attrs['unitary_amplitude'] = unitary_amplitude
        self.attrs['inverted_phase'] = inverted_phase
        self.dtype = dtype

        if Nmesh is None or BoxSize is None:
            raise ValueError('both Nmesh and BoxSize must not be None to initialize ParticleMesh')

        Nmesh = np.array(Nmesh)
        if Nmesh.ndim == 0:
            ndim = 3
        else:
            ndim = len(Nmesh)
        _Nmesh = np.empty(ndim, dtype='i8')
        _Nmesh[:] = Nmesh
        self.pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh,
                                dtype=self.dtype, comm=self.comm)

        self.attrs['BoxSize'] = self.pm.BoxSize.copy()
        self.attrs['Nmesh'] = self.pm.Nmesh.copy()

        if los is not None:
            los = self._get_los(los)
        self.attrs['los'] = los
        self.set_complex_delta_field()

    def set_complex_delta_field(self):
        delta_k = self.pm.generate_whitenoise(self.attrs['seed'],type='untransposedcomplex',unitary=self.attrs['unitary_amplitude'])
        if self.comm.rank == 0:
            self.logger.info("White noise generated")

        if self.attrs['inverted_phase']: delta_k[...] *= -1
        # volume factor needed for normalization
        norm = 1.0 / self.pm.BoxSize.prod()
        # iterate in slabs over fields
        # loop over the mesh, slab by slab
        los = self.attrs['los']
        for kslab,delta_slab in zip(delta_k.slabs.x,delta_k.slabs):
            # the square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            zero_idx = k2 == 0.
            k2[zero_idx] = 1. # avoid dividing by zero
            # the linear power (function of k)
            k = (k2**0.5).flatten()
            if los is not None:
                mu = sum(kk*ll for kk,ll in zip(kslab,los)).flatten()/k
                power = self.Plin(k,mu)
            else:
                power = self.Plin(k)

            # multiply complex field by sqrt of power
            delta_slab[...].flat *= (power*norm)**0.5

            # set k == 0 to zero (zero config-space mean)
            delta_slab[zero_idx] = 0.
        self.delta_k = delta_k

    def set_real_delta_field(self, bias=None, lognormal_transform=False, BoxCenter=0.):
        delta_r = self.delta_k.c2r()
        self.attrs['BoxCenter'] = np.empty(3,dtype=self.dtype)
        self.attrs['BoxCenter'][:] = BoxCenter
        offset = self.attrs['BoxCenter'] + 0.5*self.pm.BoxSize / self.pm.Nmesh
        if bias is not None:
            if callable(bias):
                for islabs in zip(delta_r.slabs.x, delta_r.slabs):
                    rslab, delta_slab = islabs[:2]
                    rnorm = np.sum((r + o)**2 for r,o in zip(rslab,offset))**0.5
                    delta_slab[...].flat *= bias(rnorm.flatten())
            else:
                delta_r *= bias
        if lognormal_transform:
            delta_r[:] = np.exp(delta_r.value)
            delta_r[:] /= delta_r.cmean(dtype='f8')
            delta_r[:] -= 1.
        self.delta_r = delta_r
        #self.delta_r[:] += 1.

    def set_linear_rsd_local_los(self, growth_rate=None):
        # cartesian product faster than harmonic (I guess due to non-trivial Ymls)
        offset = self.attrs['BoxCenter'] + 0.5*self.pm.BoxSize / self.pm.Nmesh
        disp_deriv_k = self.delta_k.copy()
        delta_rsd = self.delta_r.copy()
        iscallable = callable(growth_rate)
        if iscallable:
            delta_r_tot = self.delta_r.copy()
            delta_r_tot[:] = 0.
        else:
            delta_r_tot = self.delta_r
        # the real-space grid
        for i in range(self.delta_k.ndim):
            for j in range(i,self.delta_k.ndim):
                disp_deriv_k[:] = self.delta_k[:]
                for kslab, slab in zip(disp_deriv_k.slabs.x,disp_deriv_k.slabs):
                    k2 = sum(kk**2 for kk in kslab)
                    k2[k2 == 0.] = 1. # avoid dividing by zero
                    slab[...] *= kslab[i]*kslab[j]/k2
                disp_deriv_k.c2r(out=delta_rsd)
                for rslab, slab in zip(delta_rsd.slabs.x,delta_rsd.slabs):
                    rgrid = [r + o for r,o in zip(rslab,offset)]
                    r2 = np.sum(rr**2 for rr in rgrid)
                    r2[r2 == 0] = 1.
                    slab[...] *= rgrid[i]*rgrid[j]/r2
                factor = 1.
                factor += i != j
                if not iscallable and growth_rate is not None:
                    factor *= growth_rate
                delta_r_tot[:] += factor*delta_rsd[:]
        if iscallable:
            for rslab,slab in zip(delta_r_tot.slabs.x,delta_r_tot.slabs):
                rgrid = [r + o for r,o in zip(rslab,offset)]
                rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                slab[...].flat *= growth_rate(rnorm.flatten())
            self.delta_r[:] += delta_r_tot

    def read_weights(self, positions, field='1+delta', resampler='nnb'):
        # half cell shift already included in resampling
        if field == '1+delta':
            return self.delta_r.readout(positions - self.attrs['BoxCenter'],resampler=resampler) + 1.
        if field == 'nbar*delta':
            return (self.nbar*self.delta_r).readout(positions - self.attrs['BoxCenter'],resampler=resampler)
        if field == 'nbar*(1+delta)':
            return (self.nbar*(self.delta_r + 1)).readout(positions - self.attrs['BoxCenter'],resampler=resampler)
        if field == 'nbar':
            return self.nbar.readout(positions - self.attrs['BoxCenter'],resampler=resampler)

    def set_sampled_selection_function(self, positions, weights=None, resampler='cic', interlaced=False):
        # easiest is just to build a CatalogMesh and call to_real_field()
        # resampler will just expand a bit the survey selection function, but that should not be an issue for our purposes
        from nbodykit.base.catalog import CatalogSource
        from nbodykit.source.mesh.catalog import CatalogMesh
        new = object.__new__(CatalogSource)
        new._size = len(positions) # local size
        CatalogSource.__init__(new,comm=self.comm)
        new['Position'] = positions - self.attrs['BoxCenter']
        if weights is not None: new['Weight'] = weights
        new = CatalogMesh(new,Nmesh=self.pm.Nmesh,BoxSize=self.pm.BoxSize,
                        Position=new['Position'],
                        Weight=new['Weight'] if weights is not None else None,
                        dtype=self.dtype,
                        resampler=resampler,
                        compensated=False,
                        interlaced=interlaced)
        dv = np.prod(self.pm.BoxSize/self.pm.Nmesh)
        self.nbar = new.to_real_field(normalize=False)/dv
        self.attrs.update(**new.attrs)
        if weights is not None:
            shotnoise = self.comm.allreduce(weights**2)
        else:
            shotnoise = self.comm.allreduce(len(positions))
        # these are int dk W(k)^(2p) with p resampler order
        damping_factor = {'nnb':1.,'cic':2./3,'tsc':11./20.,'pcs':151./315}
        self.attrs['norm'] = (self.nbar**2).csum()*dv - damping_factor[resampler]**3*shotnoise/dv
        print(self.attrs['norm'])
        self.attrs['shotnoise'] = shotnoise/self.attrs['norm']*damping_factor[resampler]**3

    def set_analytic_selection_function(self, mask_nbar, interlaced=False):
        H = self.pm.BoxSize/self.pm.Nmesh
        dv = np.prod(H)
        self.attrs['shotnoise'] = 0.
        if not callable(mask_nbar):
            self.nbar = mask_nbar
            self.attrs['norm'] = np.prod(self.pm.Nmesh)*self.nbar**2*dv
            return
        self.nbar = self.delta_r.copy()
        for rslab, slab in zip(self.nbar.slabs.x,self.nbar.slabs):
            dist,ra,dec = cartesian_to_sky([r.flatten() for r in rslab])
            slab[...].flat = mask_nbar(dist,ra,dec)
        if interlaced:
            nbar2 = self.nbar.copy()
            #shifted = pm.affine.shift(0.5)
            offset = 0.5*H
            for rslab, slab in zip(nbar2.slabs.x,nbar2.slabs):
                dist,ra,dec = cartesian_to_sky([(r + o).flatten() for r,o in zip(rslab,offset)])
                slab[...].flat = mask_nbar(dist,ra,dec)
            c1 = self.nbar.r2c()
            c2 = nbar2.r2c()
            # and then combine
            for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                kH = sum(k[i] * H[i] for i in range(self.nbar.ndim))
                s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)
            # FFT back to real-space
            c1.c2r(out=self.nbar)
        self.attrs['norm'] = (self.nbar**2).csum()*dv

    def set_white_noise(self):
        noise = self.pm.generate_whitenoise(self.attrs['seed'],type='real')
        # noise has amplitude of 1
        # multiply by expected density
        dv = np.prod(self.pm.BoxSize/self.pm.Nmesh)
        self.delta_r += noise*self.nbar*dv

    def to_mesh(self):
        toret = FieldMesh(self.delta_r*self.nbar)
        toret.attrs.update(self.attrs)
        return toret















    """
    def set_linear_rsd_local_los_harmonic(self, growth_rate=None):
        BoxCenter = self.attrs['BoxCenter']
        disp_deriv_k = self.delta_k.copy()
        delta_rsd = self.delta_r.copy()
        iscallable = callable(growth_rate)
        delta_r_tot = 1./2.*self.delta_r.copy()

        # the real-space grid
        xgrid = [xx.astype('f8') + BoxCenter[ii] for ii, xx in enumerate(self.delta_r.slabs.optx)]
        xnorm = np.sqrt(sum(xx**2 for xx in xgrid))
        xgrid = [x/xnorm for x in xgrid]

        # the Fourier-space grid
        kgrid = [kk.astype('f8') for kk in self.delta_k.slabs.optx]
        knorm = np.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = np.inf
        kgrid = [k/knorm for k in kgrid]

        from nbodykit.algorithms.convpower.fkp import get_real_Ylm
        Ylms = [get_real_Ylm(2,m) for m in range(-2,3)]

        # the real-space grid
        for Ylm in Ylms:
            # reset the real-space mesh to the original density #2
            disp_deriv_k[:] = self.delta_k[:]
            # apply the Fourier-space Ylm
            for islab, slab in enumerate(disp_deriv_k.slabs):
                slab[:] *= Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])
            # real to complex of field #2
            disp_deriv_k.c2r(out=delta_rsd)
            # apply the config-space Ylm
            for islab, slab in enumerate(delta_rsd.slabs):
                # 4\pi/(2\ell+1)
                slab[:] *= 4*np.pi/5.*Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])
            # add to the total sum
            delta_r_tot[:] += delta_rsd[:]

        if iscallable:
            for rslab,slab in zip(delta_r_tot.slabs.x,delta_r_tot.slabs):
                rgrid = [r + c for r,c in zip(rslab,BoxCenter)]
                rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                slab[...].flat *= growth_rate(rnorm.flatten())
        elif growth_rate is not None:
            delta_r_tot[:] *= growth_rate
        self.delta_r[:] += 2./3.*delta_r_tot
    """
    """
    def _set_linear_rsd_local_los(self, growth_rate=None):
        BoxCenter = self.attrs['BoxCenter']
        disp_deriv_k = self.disp_k[0].copy()
        delta_rsd = self.delta_r.copy()
        iscallable = callable(growth_rate)
        if iscallable:
            delta_r_tot = self.delta_r.copy()
        else:
            delta_r_tot = self.delta_r
        # the real-space grid
        xgrid = [xx + BoxCenter[ii] for ii, xx in enumerate(delta_rsd.slabs.optx)]
        xnorm = np.sum(xx**2 for xx in xgrid)**0.5

        for i in range(len(self.disp_k)):
            for j in range(i,len(self.disp_k)):
                disp_deriv_k[:] = self.disp_k[i][:]
                for kslab, slab in zip(disp_deriv_k.slabs.x,disp_deriv_k.slabs):
                    slab[...] *= 1j*kslab[j]
                disp_deriv_k.c2r(out=delta_rsd)
                for rslab, slab in zip(delta_rsd.slabs.x,delta_rsd.slabs):
                    rgrid = [r + c for r,c in zip(rslab,BoxCenter)]
                    rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                    rnorm[rnorm == 0] = 1.
                    slab[...] *= rgrid[i]*rgrid[j]/rnorm**2
                if not iscallable and growth_rate is not None:
                    delta_rsd[...] *= growth_rate
                delta_r_tot[:] += delta_rsd[:]
        if iscallable:
            for rslab,slab in zip(delta_r_tot.slabs.x,delta_r_tot.slabs):
                rgrid = [r + c for r,c in zip(rslab,BoxCenter)]
                rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                slab[...].flat *= growth_rate(rnorm.flatten())
            self.delta_r[:] += delta_r_tot

    def set_linear_rsd_local_los(self, growth_rate=None):
        BoxCenter = self.attrs['BoxCenter']
        disp_deriv_k = self.disp_k[0].copy()
        delta_rsd = self.delta_r.copy()
        iscallable = callable(growth_rate)
        if iscallable:
            delta_r_tot = self.delta_r.copy()
        else:
            delta_r_tot = self.delta_r
        # the real-space grid
        for i in range(len(self.disp_k)):
            for j in range(i,len(self.disp_k)):
                disp_deriv_k[:] = self.disp_k[i][:]
                for kslab, slab in zip(disp_deriv_k.slabs.x,disp_deriv_k.slabs):
                    slab[...] *= -1j*kslab[j]
                disp_deriv_k.c2r(out=delta_rsd)
                for rslab, slab in zip(delta_rsd.slabs.x,delta_rsd.slabs):
                    rgrid = [r + c for r,c in zip(rslab,BoxCenter)]
                    rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                    rnorm[rnorm == 0] = 1.
                    slab[...] *= rgrid[i]*rgrid[j]/rnorm**2
                factor = 1.
                factor += i != j
                if not iscallable and growth_rate is not None:
                    factor *= growth_rate
                delta_r_tot[:] += factor*delta_rsd[:]
        if iscallable:
            for rslab,slab in zip(delta_r_tot.slabs.x,delta_r_tot.slabs):
                rgrid = [r + c for r,c in zip(rslab,BoxCenter)]
                rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                slab[...].flat *= growth_rate(rnorm.flatten())
            self.delta_r[:] += delta_r_tot

    def to_mesh(self, interlaced=False, return_selection_function=False):
        if return_selection_function:
            toret = self.delta_r * self.nbar
        else:
            toret = self.nbar
        if not interlaced:
            return toret
        shifted = toret.copy()

        # compose the two interlaced fields into the final result.
        c1 = real1.r2c()
        c2 = real2.r2c()

        # and then combine
        for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
            kH = sum(k[i] * H[i] for i in range(3))
            s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)

        # FFT back to real-space
        # NOTE: cannot use "toret" here in case user supplied "out"
        c1.c2r(real1)
    """
