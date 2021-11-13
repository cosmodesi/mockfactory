import logging

import numpy as np

from .gaussian_mock import BaseGaussianMock
from . import utils


class EulerianLinearMock(BaseGaussianMock):
    """
    Extend :class:`BaseGaussianMock` with first order Eulerian bias (and growth) and RSD.

    A standard run would be:

    .. code-block:: python

        mock = EulerianLinearMock(power, boxsize, nmesh)
        mock.set_real_delta_field()
        mock.set_rsd(f, los)
        # from here you have mock.mesh_delta_r
        mock.poisson_sample(seed)
        # from here you have mock.positions

    Attributes
    ----------
    mesh_delta_k : pm.ComplexField
        Density fluctuations in Fourier space.

    mesh_delta_r : pm.RealField
        Density fluctuations in real space.

    nbar : pm.RealField
        Selection function in real space.

    positions : array of shape (N, 3)
        Cartesian positions sampling the density field.
    """
    def set_rsd(self, f, los=None):
        r"""
        Add redshift space distortions to :attr:`mesh_delta_r`.

        Parameters
        ----------
        f : callable, float
            Relation between the velocity divergence :math:`\theta' and the initial density contrast :math:`\delta'.
            If a callable, take the (flattened) distance to the observer as input, i.e. :math:`\theta = f(r) \delta'.
            Else, a float to multiply the :math:`\delta' field, i.e. :math:`\theta = f \delta'.

        los : 'x', 'y', 'z'; int, 3-vector of int, default=None
            Line of sight :math:`\hat{\eta}` for RSD.
            If ``None``, use local line of sight.
        """
        # cartesian product faster than harmonic (I guess due to non-trivial Ymls)
        offset = self.boxcenter #+ 0.5*self.boxsize / self.nmesh
        disp_deriv_k = self.mesh_delta_k.copy()
        iscallable = callable(f)

        mesh_delta_r_tot = self.mesh_delta_r.copy()
        mesh_delta_r_tot[:] = 0.

        if los is not None:
            los = _get_los(los)
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
                k2 = sum(kk**2 for kk in kslab)
                k2[k2 == 0.] = 1. # avoid dividing by zero
                k = k2**0.5
                mu = sum(kk*ll for kk,ll in zip(kslab, los))/k
                slab[...] *= mu**2
            disp_deriv_k.c2r(out=mesh_delta_r_tot)
            if not iscallable:
                mesh_delta_r_tot *= f
        else:
            # the real-space grid
            mesh_delta_rsd = self.mesh_delta_r.copy()
            for i in range(self.ndim):
                for j in range(i,self.ndim):
                    disp_deriv_k[:] = self.mesh_delta_k[:]
                    for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
                        k2 = sum(kk**2 for kk in kslab)
                        k2[k2 == 0.] = 1. # avoid dividing by zero
                        slab[...] *= kslab[i]*kslab[j]/k2
                    disp_deriv_k.c2r(out=mesh_delta_rsd)
                    for rslab, slab in zip(mesh_delta_rsd.slabs.x,mesh_delta_rsd.slabs):
                        # reslab in [-boxsize/2., boxsize/2.]
                        rgrid = [r + o for r,o in zip(rslab, offset)]
                        r2 = np.sum(rr**2 for rr in rgrid)
                        slab[...] *= rgrid[i]*rgrid[j]/r2
                    factor = 1. + (i != j)
                    if not iscallable:
                        factor *= f
                    mesh_delta_r_tot[:] += factor*mesh_delta_rsd[:]
        if iscallable:
            for rslab,slab in zip(mesh_delta_r_tot.slabs.x, mesh_delta_r_tot.slabs):
                rgrid = [r + o for r,o in zip(rslab, offset)]
                rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                slab[...].flat *= f(rnorm.flatten())

        self.mesh_delta_r[:] += mesh_delta_r_tot
