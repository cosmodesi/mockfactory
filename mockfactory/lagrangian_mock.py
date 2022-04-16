from .gaussian_mock import BaseGaussianMock, _get_los
from . import utils


class LagrangianLinearMock(BaseGaussianMock):
    """
    Extend :class:`BaseGaussianMock` with first order Lagrangian bias (and growth) and RSD,
    i.e. the Zeldovich approximation.

    A standard run would be:

    .. code-block:: python

        mock = LagrangianLinearMock(power, boxsize, nmesh)
        mock.set_real_delta_field()
        # from here you have mock.mesh_delta_r and mock.mesh_disp_r
        mock.poisson_sample(seed)
        # from here you have mock.positions and mock.disps
        mock.set_rsd(f, los)

    Attributes
    ----------
    mesh_delta_k : pm.ComplexField
        Density fluctuations in Fourier space.

    mesh_delta_r : pm.RealField
        Density fluctuations in real space.

    nbar : pm.RealField
        Selection function in real space.

    mesh_disp_r : list of 3 pm.RealField
        Zeldovich displacement fields.

    positions : array of shape (N, 3)
        Cartesian positions sampling the density field.
    """
    def set_real_delta_field(self, bias=None, lognormal_transform=True):
        r"""
        Set the density contrast in real space :attr:`mesh_delta_r`
        and the Zeldovich displacement fields :attr:`mesh_disp_r`.

        Parameters
        ----------
        bias : callable, float, default=None
            Lagrangian bias (optionally including growth factor), related to Eulerian bias :math:`b_{E}` by :math:`b_{L} = b_{E} - 1`.
            If a callable, take the (flattened) :math:`\delta' field and (flattened) distance to the observer as input.
            Else, a float to multiply the :math:`\delta' field.

        lognormal_transform : bool, default=True
            Whether to apply a lognormal transform to the (biased) real field, i.e. :math:`\exp{(\delta)}'
        """
        super(LagrangianLinearMock, self).set_real_delta_field(bias=bias, lognormal_transform=lognormal_transform)
        disp_k = [self.pm.create(type='untransposedcomplex') for i in range(self.ndim)]
        slabs = [self.mesh_delta_k.slabs.x, self.mesh_delta_k.slabs] + [d.slabs for d in disp_k]
        for islabs in zip(*slabs):
            kslab, delta_slab = islabs[:2]  # the k arrays and delta slab
            # the square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            mask_zero = k2 == 0.
            k2[mask_zero] = 1.  # avoid dividing by zero
            for i in range(self.ndim):
                disp_slab = islabs[2 + i]
                disp_slab[...] = 1j * kslab[i] / k2 * delta_slab[...]
                # disp_slab[mask_zero] = 0.  # no bulk displacement
        self.mesh_disp_r = [d.c2r() for d in disp_k]

    def readout(self, positions, field='delta', resampler='nnb', compensate=False):
        """
        Read density field at input positions.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        field : string, pm.RealField, default='1+delta'
            Mesh or type of field to read, either 'delta', 'nbar*delta', 'nbar*(1+delta)', 'nbar',
            'disp_x', 'disp_y', 'disp_z'.
            Fields with 'nbar' require calling :meth:`set_analytic_selection_function` first.

        resampler : string, default='nnb'
            Resampler to interpolate the field at input positions.
            e.g. 'ngp', 'cic', 'tsc', 'pcs'...

        compensate : bool, default=False
            Whether to compensate for smooting due to resampling, to make the power spectrum
            of output (positions, values) match that of ``field`` (up to sampling noise).

        Returns
        -------
        values : array of shape (N,)
            Field values interpolated at input positions.
        """
        if isinstance(field, str) and field.startswith('disp_'):
            iaxis = 'xyz'.index(field[len('disp_'):])
            field = self.mesh_disp_r[iaxis]

        return super(LagrangianLinearMock, self).readout(positions, field, resampler=resampler)

    def poisson_sample(self, seed=None, resampler='cic', compensate=False):
        """
        Poisson sample density field and set :attr:`position` and :attr:`disp`,
        Zeldovich displacements interpolated at :attr:`position`.

        Note
        ----
        :meth:`set_real_delta_field` must be called first.

        Parameters
        ----------
        seed : int, default=None
            Random seed.

        resampler : string, default='nnb'
            Resampler to interpolate the displacement field at sampled positions.
            e.g. 'ngp', 'cic', 'tsc', 'pcs'...

        compensate : bool, default=False
            Whether to compensate for smooting due to resampling.
        """
        super(LagrangianLinearMock, self).poisson_sample(seed=seed)
        self.disps = self.position.copy()
        for iaxis, mesh_disp_r in enumerate(self.mesh_disp_r):
            self.disps[:, iaxis] = self.readout(self.position, field=mesh_disp_r, resampler=resampler, compensate=compensate)
        self.position += self.disps

    def set_rsd(self, f, los=None):
        r"""
        Add redshift space distortions to :attr:`position`.

        Note
        ----
        :meth:`poisson_sample` must be called first.

        Parameters
        ----------
        f : callable, float
            Relation between the Zeldovich displacement field :math:`\psi' and the RSD displacement.
            If a callable, take the (flattened) distance to the observer as input, i.e. :math:`f(r) \psi'.
            Else, a float to multiply the :math:`\psi' field, i.e. :math:`f \psi'.

        los : 'x', 'y', 'z'; int, 3-vector of int, default=None
            Line of sight :math:`\hat{\eta}` for RSD.
            If ``None``, use local line of sight.
        """
        if los is None:
            los = self.position / utils.distance(self.position)[:, None]
        else:
            los = _get_los(los)
        rsd = utils.vector_projection(self.disps, los)
        iscallable = callable(f)
        if iscallable:
            rsd *= f(utils.distance(self.position))
        else:
            rsd *= f
        self.position += rsd

    def to_catalog(self):
        """
        Export as :class:`make_survey.BoxCatalog`.

        Note
        ----
        :meth:`poisson_sample` must be called first.
        """
        source = {'Position': self.position, 'Displacement': self.disps}
        from .make_survey import BoxCatalog
        return BoxCatalog(source, position='Position', velocity='Displacement', boxsize=self.boxsize, boxcenter=self.boxcenter, attrs=self.attrs)
