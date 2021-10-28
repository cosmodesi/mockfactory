import numpy
import logging
import warnings
import time

from pmesh.pm import ComplexField
from nbodykit.utils import attrs_to_dict, timer
from nbodykit.source.mesh import MultipleSpeciesCatalogMesh
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges
from nbodykit.algorithms.convpower.catalog import FKPCatalog as _FKPCatalog
from nbodykit.algorithms.convpower.catalogmesh import FKPCatalogMesh as _FKPCatalogMesh
from nbodykit.algorithms.convpower.fkp import ConvolvedFFTPower as _ConvolvedFFTPower
from nbodykit.algorithms.convpower.fkp import _cast_mesh, get_real_Ylm, copy_meta, get_compensation


class ConvolvedFFTPower(_ConvolvedFFTPower):

    def __init__(self, first, poles,
                    second=None,
                    Nmesh=None,
                    edges=None,
                    same_noise=False):

        first = _cast_mesh(first, Nmesh=Nmesh)
        if second is not None:
            second = _cast_mesh(second, Nmesh=Nmesh)
        else:
            second = first

        isauto = second is first

        self.same_noise = {'data':isauto,'randoms':isauto}
        if not isauto:
            if same_noise == 'all':
                self.same_noise['data'] = self.same_noise['randoms'] = True
            elif same_noise:
                self.same_noise[same_noise] = True

        self.first = first
        self.second = second

        # grab comm from first source
        self.comm = first.comm

        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"

        # make a box big enough for both catalogs if they are not equal
        # NOTE: both first/second must have the same BoxCenter to recenter Position
        if not numpy.array_equal(first.attrs['BoxSize'], second.attrs['BoxSize']):

            # stack box coordinates together
            joint = {}
            for name in ['BoxSize', 'BoxCenter']:
                joint[name] = numpy.vstack([first.attrs[name], second.attrs[name]])

            # determine max box length along each dimension
            argmax = numpy.argmax(joint['BoxSize'], axis=0)
            joint['BoxSize'] = joint['BoxSize'][argmax, [0,1,2]]
            joint['BoxCenter'] = joint['BoxCenter'][argmax, [0,1,2]]

            # re-center the box
            first.recenter_box(joint['BoxSize'], joint['BoxCenter'])
            second.recenter_box(joint['BoxSize'], joint['BoxCenter'])

        # make a list of multipole numbers
        if numpy.isscalar(poles):
            poles = [poles]

        # store meta-data
        self.attrs = {}
        self.attrs['poles'] = poles
        self.attrs['edges'] = edges

        # store BoxSize and BoxCenter from source
        self.attrs['Nmesh'] = self.first.attrs['Nmesh'].copy()
        self.attrs['BoxSize'] = self.first.attrs['BoxSize']
        self.attrs['BoxPad'] = self.first.attrs['BoxPad']
        self.attrs['BoxCenter'] = self.first.attrs['BoxCenter']

        # grab some mesh attrs, too
        self.attrs['mesh.resampler'] = self.first.resampler
        self.attrs['mesh.interlaced'] = self.first.interlaced

        # and run
        self.run()

    def run(self):
        """
        Compute the power spectrum multipoles. This function does not return
        anything, but adds several attributes (see below).

        Attributes
        ----------
        edges : array_like
            the edges of the wavenumber bins
        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that behaves similar to a structured array, with
            fancy slicing and re-indexing; it holds the measured multipole
            results, as well as the number of modes (``modes``) and average
            wavenumbers values in each bin (``k``)
        attrs : dict
            dictionary holding input parameters and several important quantites
            computed during execution:

            #. data.N, randoms.N :
                the unweighted number of data and randoms objects
            #. data.W, randoms.W :
                the weighted number of data and randoms objects, using the
                column specified as the completeness weights
            #. alpha :
                the ratio of ``data.W`` to ``randoms.W``
            #. data.norm, randoms.norm :
                the normalization of the power spectrum, computed from either
                the "data" or "randoms" catalog (they should be similar).
                See equations 13 and 14 of arxiv:1312.4611.
            #. data.shotnoise, randoms.shotnoise :
                the shot noise values for the "data" and "random" catalogs;
                See equation 15 of arxiv:1312.4611.
            #. shotnoise :
                the total shot noise for the power spectrum, equal to
                ``data.shotnoise`` + ``randoms.shotnoise``; this should be subtracted from
                the monopole.
            #. BoxSize :
                the size of the Cartesian box used to grid the data and
                randoms objects on a Cartesian mesh.

            For further details on the meta-data, see
            :ref:`the documentation <fkp-meta-data>`.
        """
        pm = self.first.pm

        edges = self.attrs['edges']
        kcoords = None
        if isinstance(edges,dict):
            dk = 2*numpy.pi/pm.BoxSize.min() if edges.get('step',None) is None else edges['step']
            kmin = edges.get('min',0.)
            kmax = edges.get('max',numpy.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2)
            if dk > 0:
                kedges = numpy.arange(kmin, kmax, dk)
            else:
                k = pm.create_coords('complex')
                kedges, kcoords = _find_unique_edges(k, 2 * numpy.pi / pm.BoxSize, kmax, pm.comm)
                if self.comm.rank == 0:
                    self.logger.info('%d unique k values are found' % len(kcoords))
        else:
            kedges = numpy.array(edges)

        # measure the binned 1D multipoles in Fourier space
        result = self._compute_multipoles(kedges)

        # set all the necessary results
        self.poles = BinnedStatistic(['k'], [kedges], result,
                            fields_to_sum=['modes'],
                            coords=[kcoords],
                            **{key:value for key,value in self.attrs.items() if key != 'edges'})

        self.edges = self.attrs['edges'] = kedges

    def _compute_multipoles(self, kedges):
        """
        Compute the window-convoled power spectrum multipoles, for a data set
        with non-trivial survey geometry.

        This estimator builds upon the work presented in Bianchi et al. 2015
        and Scoccimarro et al. 2015, but differs in the implementation. This
        class uses the spherical harmonic addition theorem such that
        only :math:`2\ell+1` FFTs are required per multipole, rather than the
        :math:`(\ell+1)(\ell+2)/2` FFTs in the implementation presented by
        Bianchi et al. and Scoccimarro et al.

        References
        ----------
        * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
          MNRAS, 2015
        * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
        """
        # clear compensation from the actions
        for source in [self.first, self.second]:
            source.actions[:] = []; source.compensated = False
            assert len(source.actions) == 0

        # compute the compensations
        compensation = {}
        for name, mesh in zip(['first', 'second'], [self.first, self.second]):
            compensation[name] = get_compensation(mesh)
            if self.comm.rank == 0:
                if compensation[name] is not None:
                    args = (compensation[name]['func'].__name__, name)
                    self.logger.info("using compensation function %s for source '%s'" % args)
                else:
                    self.logger.warning("no compensation applied for source '%s'" % name)

        rank = self.comm.rank
        pm   = self.first.pm

        # setup the 1D-binning
        muedges = numpy.linspace(-1, 1, 2, endpoint=True)
        edges = [kedges, muedges]

        # make a structured array to hold the results
        cols   = ['k'] + ['power_%d' %l for l in sorted(self.attrs['poles'])] + ['modes']
        dtype  = ['f8'] + ['c8']*len(self.attrs['poles']) + ['i8']
        dtype  = numpy.dtype(list(zip(cols, dtype)))
        result = numpy.empty(len(kedges)-1, dtype=dtype)

        # offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to
        # the original (x,y,z) coords
        offset = self.attrs['BoxCenter'] + 0.5*pm.BoxSize / pm.Nmesh

        # always need to compute ell=0
        poles = sorted(self.attrs['poles'])
        if 0 not in poles:
            poles = [0] + poles
        assert poles[0] == 0

        # spherical harmonic kernels (for ell > 0)
        Ylms = [[get_real_Ylm(l,m) for m in range(-l, l+1)] for l in poles[1:]]

        # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
        rfield1 = self.first.compute(Nmesh=self.attrs['Nmesh'])
        meta1 = rfield1.attrs.copy()
        if rank == 0:
            self.logger.info("%s painting of 'first' done" %self.first.resampler)

        # store alpha: ratio of data to randoms
        self.attrs['second.alpha'] = self.attrs['first.alpha'] = meta1['alpha']

        # FFT 1st density field and apply the resampler transfer kernel
        cfield = rfield1.r2c()
        if compensation['first'] is not None:
            cfield.apply(out=Ellipsis, **compensation['first'])
        if rank == 0: self.logger.info('ell = 0 done; 1 r2c completed')

        # monopole A0 is just the FFT of the FKP density field
        # NOTE: this holds FFT of density field #1
        volume = pm.BoxSize.prod()
        A0_1 = ComplexField(pm)
        A0_1[:] = cfield[:] * volume # normalize with a factor of volume

        # paint second mesh too?
        if self.first is not self.second:

            # paint the second field
            rfield2 = self.second.compute(Nmesh=self.attrs['Nmesh'])
            meta2 = rfield2.attrs.copy()
            self.attrs['second.alpha'] = meta2['alpha']
            if rank == 0: self.logger.info("%s painting of 'second' done" %self.second.resampler)

            # need monopole of second field
            if 0 in self.attrs['poles']:

                # FFT density field and apply the resampler transfer kernel
                A0_2 = rfield2.r2c()
                A0_2[:] *= volume
                if compensation['second'] is not None:
                    A0_2.apply(out=Ellipsis, **compensation['second'])
        else:
            rfield2 = rfield1
            meta2 = meta1

            # monopole of second field is first field
            if 0 in self.attrs['poles']:
                A0_2 = A0_1

        # save the painted density field #2 for later
        density2 = rfield2.copy()

        # initialize the memory holding the Aell terms for
        # higher multipoles (this holds sum of m for fixed ell)
        # NOTE: this will hold FFTs of density field #2
        Aell = ComplexField(pm)

        # the real-space grid
        xgrid = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(density2.slabs.optx)]
        xnorm = numpy.sqrt(sum(xx**2 for xx in xgrid))
        xgrid = [x/xnorm for x in xgrid]

        # the Fourier-space grid
        kgrid = [kk.astype('f8') for kk in cfield.slabs.optx]
        knorm = numpy.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = numpy.inf
        kgrid = [k/knorm for k in kgrid]

        # proper normalization: same as equation 49 of Scoccimarro et al. 2015
        self.attrs['data.norm'] = self.attrs['randoms.norm'] = self.normalization()

        if self.attrs['randoms.norm'] > 0:
            norm = 1.0 / self.attrs['randoms.norm']
            if rank == 0:
                self.logger.info("normalized power spectrum with `randoms.norm = %.6f`" % norm)
        else:
            norm = 1.0
            if rank == 0:
                self.logger.info("normalization of power spectrum is neglected, as no random is provided.")

        # loop over the higher order multipoles (ell > 0)
        start = time.time()
        for iell, ell in enumerate(poles[1:]):

            # clear 2D workspace
            Aell[:] = 0.

            # iterate from m=-l to m=l and apply Ylm
            substart = time.time()
            for Ylm in Ylms[iell]:

                # reset the real-space mesh to the original density #2
                rfield2[:] = density2[:]

                # apply the config-space Ylm
                for islab, slab in enumerate(rfield2.slabs):
                    slab[:] *= Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])

                # real to complex of field #2
                rfield2.r2c(out=cfield)

                # apply the Fourier-space Ylm
                for islab, slab in enumerate(cfield.slabs):
                    slab[:] *= Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])

                # add to the total sum
                Aell[:] += cfield[:]

                # and this contribution to the total sum
                substop = time.time()
                if rank == 0:
                    self.logger.debug("done term for Y(l=%d, m=%d) in %s" %(Ylm.l, Ylm.m, timer(substart, substop)))

            # apply the compensation transfer function
            if compensation['second'] is not None:
                Aell.apply(out=Ellipsis, **compensation['second'])

            # factor of 4*pi from spherical harmonic addition theorem + volume factor
            Aell[:] *= 4*numpy.pi*volume

            # log the total number of FFTs computed for each ell
            if rank == 0:
                args = (ell, len(Ylms[iell]))
                self.logger.info('ell = %d done; %s r2c completed' %args)

            # calculate the power spectrum multipoles, slab-by-slab to save memory
            # NOTE: this computes (A0 of field #1) * (Aell of field #2).conj()
            for islab in range(A0_1.shape[0]):
                Aell[islab,...] = norm * A0_1[islab] * Aell[islab].conj()

            # project on to 1d k-basis (averaging over mu=[0,1])
            proj_result, _ = project_to_basis(Aell, edges)
            result['power_%d' %ell][:] = numpy.squeeze(proj_result[2])

        # summarize how long it took
        stop = time.time()
        if rank == 0:
            self.logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))

        # also compute ell=0
        if 0 in self.attrs['poles']:

            # the 3D monopole
            for islab in range(A0_1.shape[0]):
                A0_1[islab,...] = norm*A0_1[islab]*A0_2[islab].conj()

            # the 1D monopole
            proj_result, _ = project_to_basis(A0_1, edges)
            result['power_0'][:] = numpy.squeeze(proj_result[2])

        # save the number of modes and k
        result['k'][:] = numpy.squeeze(proj_result[0])
        result['modes'][:] = numpy.squeeze(proj_result[-1])

        # compute shot noise
        self.attrs['shotnoise'] = self.shotnoise()

        # copy over any painting meta data
        if self.first is self.second:
            copy_meta(self.attrs, meta1)
        else:
            copy_meta(self.attrs, meta1, prefix='first')
            copy_meta(self.attrs, meta2, prefix='second')

        return result

    def normalization(self):
        r"""
        Compute the power spectrum normalization, using either the
        ``data`` or ``randoms`` source.

        The normalization is given by:

        .. math::

            A = \int d^3x \bar{n}'_1(x) \bar{n}'_2(x) w_{\mathrm{fkp},1} w_{\mathrm{fkp},2}.

        The mean densities are assumed to be the same, so this can be converted
        to a summation over objects in the source, as

        .. math::

            A = \sum w_{\mathrm{comp},1} \bar{n}_2 w_{\mathrm{fkp},1} w_{\mathrm{fkp},2}.

        References
        ----------
        see Eqs. 13,14 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        # the selection (same for first/second)

        vol_per_cell = (self.attrs['BoxSize']/self.attrs['Nmesh']).prod()
        #first = self.first.randoms_field
        #second = self.second.randoms_field

        #for field in [first,second]:
        #    field.resampler = 'tsc'
        #    field.interlaced = True
        #    field.compensated = True
        #    compensation = get_compensation(field)
        first = self.first['randoms']
        compensation = get_compensation(first)
        first = first.to_real_field(normalize=False)
        first = first.r2c()
        first.apply(out=Ellipsis,**compensation)

        total = 0.
        for kslab,slab in zip(first.slabs.x,first.slabs):
            #mask = sum(kk**2 for kk in kslab) == 0
            #slab[mask] = 0.
            total += numpy.sum(slab * slab.conj()).real
        total = self.comm.allreduce(total)*numpy.prod(self.attrs['Nmesh'])
        shotnoise = self.unnormalized_shotnoise(name='randoms')
        A = total/vol_per_cell - shotnoise/vol_per_cell
        return self.attrs['first.alpha']*self.attrs['second.alpha']*A

        """
        damping_factor = [1.,1.,1.,3./4.,2./3.,115./192.,11/20.,5887./11520.,151./315]
        damping_order = {'nnb':1,'cic':2,'tsc':3,'pcs':4}
        damping = damping_factor[6]

        first = first.c2r()
        print((first**2).csum().real)
        #second = second.to_real_field(normalize=False)
        #second = second.r2c()
        #second.apply(out=Ellipsis,**compensation)
        #second = second.c2r()
        shotnoise = self.unnormalized_shotnoise(name='randoms')

        A = (first**2).csum().real/vol_per_cell - damping**3*shotnoise/vol_per_cell
        return self.attrs['first.alpha']*self.attrs['second.alpha']*A
        """
        """
        # these are int dk W(k)^(p) with p resampler order, p = 0...8
        damping_factor = [1.,1.,1.,3./4.,2./3.,115./192.,11/20.,5887./11520.,151./315]
        damping_order = {'nnb':1,'cic':2,'tsc':3,'pcs':4}
        damping = damping_factor[damping_order[first.resampler] + damping_order[second.resampler]]
        shotnoise = self.unnormalized_shotnoise(name='randoms')
        damping = 1.
        A = (first*second).csum().real/vol_per_cell - damping**3*shotnoise/vol_per_cell
        del self.first.randoms_field
        if self.second is not self.first:
            del self.second.randoms_field

        return self.attrs['first.alpha']*self.attrs['second.alpha']*A
        """

    def unnormalized_shotnoise(self, name='data'):
        r"""
        Compute the power spectrum shot noise, using either the
        ``data`` or ``randoms`` source.

        This computes:

        .. math::

            S = \sum (w_\mathrm{comp} w_\mathrm{fkp})^2

        References
        ----------
        see Eq. 15 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        Pshot = 0
        if not self.same_noise[name]:
            return Pshot

        # the selection (same for first/second)
        sel = self.first.source.compute(self.first.source[name][self.first.selection])

        # selected first/second meshes for "name" (data or randoms)
        first = self.first.source[name][sel]
        second = self.second.source[name][sel]

        # completeness weights (assumed same for first/second)
        weight1 = first[self.first.comp_weight]*first[self.first.fkp_weight]
        weight2 = second[self.second.comp_weight]*second[self.second.fkp_weight]

        Pshot = numpy.sum(weight1*weight2)

        # reduce sum across all ranks
        Pshot = self.comm.allreduce(first.compute(Pshot))

        # divide by normalization from randoms
        return Pshot

    def shotnoise(self):
        r"""
        Compute the power spectrum shot noise, using either the
        ``data`` or ``randoms`` source.

        This computes:

        .. math::

            S = \sum (w_\mathrm{comp} w_\mathrm{fkp})^2

        References
        ----------
        see Eq. 15 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        Pshot = 0
        for name in ['data', 'randoms']:

            S = self.unnormalized_shotnoise(name)
            if name == 'randoms':
                alpha2 = self.attrs['first.alpha']*self.attrs['second.alpha']
                S *= alpha2
            Pshot += S # add to total

        # divide by normalization from randoms
        return Pshot / self.attrs['randoms.norm']



class FKPCatalog(_FKPCatalog):

    def to_mesh(self, Nmesh=None, BoxSize=None, BoxCenter=None, dtype='c16', interlaced=False,
                compensated=False, resampler='cic', fkp_weight='FKPWeight',
                comp_weight='Weight', selection='Selection',
                position='Position', bbox_from_species=None, window=None, nbar=None):

        """
        Convert the FKPCatalog to a mesh, which knows how to "paint" the
        FKP density field.

        Additional keywords to the :func:`to_mesh` function include the
        FKP weight column, completeness weight column, and the column
        specifying the number density as a function of redshift.

        Parameters
        ----------
        Nmesh : int, 3-vector, optional
            the number of cells per box side; if not specified in `attrs`, this
            must be provided
        dtype : str, dtype, optional
            the data type of the mesh when painting. dtype='f8' or 'f4' assumes
            Hermitian symmetry of the input field (\delta(x) =
            \delta^{*}(-x)), and stores it as an N x N x N/2+1 real array.
            This speeds evaluation of even multipoles but yields
            incorrect odd multipoles in the presence of the wide-angle effect.
            dtype='c16' or 'c8' stores the field as an N x N x N complex array
            to correctly recover the odd multipoles.
        interlaced : bool, optional
            whether to use interlacing to reduce aliasing when painting the
            particles on the mesh
        compensated : bool, optional
            whether to apply a Fourier-space transfer function to account for
            the effects of the gridding + aliasing
        resampler : str, optional
            the string name of the resampler to use when interpolating the
            particles to the mesh; see ``pmesh.window.methods`` for choices
        fkp_weight : str, optional
            the name of the column in the source specifying the FKP weight;
            this weight is applied to the FKP density field:
            ``n_data - alpha*n_randoms``
        comp_weight : str, optional
            the name of the column in the source specifying the completeness
            weight; this weight is applied to the individual fields, either
            ``n_data``  or ``n_random``
        selection : str, optional
            the name of the column used to select a subset of the source when
            painting
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog
        bbox_from_species: str, optional
            if given, use the species to infer a bbox.
            if not give, will try random, then data (if random is empty)
        window : deprecated.
            use resampler=
        nbar: deprecated.
            deprecated. set nbar in the call to FKPCatalog()
        """
        if window is not None:
            import warnings
            resampler = window
            warnings.warn("the window argument is deprecated. Use resampler= instead", DeprecationWarning)

        # verify that all of the required columns exist
        for name in self.species:
            for col in [fkp_weight, comp_weight]:
                if col not in self[name]:
                    raise ValueError("the '%s' species is missing the '%s' column" %(name, col))

        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError("cannot convert FKP source to a mesh; 'Nmesh' keyword is not "
                                 "supplied and the FKP source does not define one in 'attrs'.")

        # first, define the Cartesian box
        if bbox_from_species is not None:
            BoxSize1, BoxCenter1 = self._define_bbox(position, selection, bbox_from_species)
        else:
            if self['randoms'].csize > 0:
                BoxSize1, BoxCenter1 = self._define_bbox(position, selection, "randoms")
            else:
                BoxSize1, BoxCenter1 = self._define_bbox(position, selection, "data")

        if BoxSize is None:
            BoxSize = BoxSize1

        if BoxCenter is None:
            BoxCenter = BoxCenter1

        # log some info
        if self.comm.rank == 0:
            self.logger.info("BoxSize = %s" %str(BoxSize))
            self.logger.info("BoxCenter = %s" %str(BoxCenter))

        # initialize the FKP mesh
        kws = {'Nmesh':Nmesh, 'BoxSize':BoxSize, 'BoxCenter': BoxCenter, 'dtype':dtype, 'selection':selection}
        return FKPCatalogMesh(self,
                              comp_weight=comp_weight,
                              fkp_weight=fkp_weight,
                              position=position,
                              value='Value',
                              interlaced=interlaced,
                              compensated=compensated,
                              resampler=resampler,
                              **kws)



class FKPCatalogMesh(_FKPCatalogMesh):

    def __init__(self, source, BoxSize, BoxCenter, Nmesh, dtype, selection,
                    comp_weight, fkp_weight, value='Value',
                    position='Position', interlaced=False,
                    compensated=False, resampler='cic'):

        if not isinstance(source, FKPCatalog):
            raise TypeError("the input source for FKPCatalogMesh must be a FKPCatalog")

        uncentered_position = position
        position = '_RecenteredPosition'
        weight = '_TotalWeight'

        self.attrs.update(source.attrs)

        self.recenter_box(BoxSize, BoxCenter)

        MultipleSpeciesCatalogMesh.__init__(self, source=source,
                        BoxSize=BoxSize, Nmesh=Nmesh,
                        dtype=dtype, weight=weight, value=value, selection=selection, position=position,
                        interlaced=interlaced, compensated=compensated, resampler=resampler)

        self._uncentered_position = uncentered_position
        self.comp_weight = comp_weight
        self.fkp_weight = fkp_weight

    def to_real_field(self):
        r"""
        Paint the FKP density field, returning a ``RealField``.

        Given the ``data`` and ``randoms`` catalogs, this paints:

        .. math::

            F(x) = w_\mathrm{fkp}(x) * [w_\mathrm{comp}(x)*n_\mathrm{data}(x) -
                        \alpha * w_\mathrm{comp}(x)*n_\mathrm{randoms}(x)]


        This computes the following meta-data attributes in the process of
        painting, returned in the :attr:`attrs` attributes of the returned
        RealField object:

        - randoms.W, data.W :
            the weighted sum of randoms and data objects; see
            :func:`weighted_total`
        - alpha : float
            the ratio of ``data.W`` to ``randoms.W``
        - randoms.norm, data.norm : float
            the power spectrum normalization; see :func:`normalization`
        - randoms.shotnoise, data.shotnoise: float
            the shot noise for each sample; see :func:`shotnoise`
        - shotnoise : float
            the total shot noise, equal to the sum of ``randoms.shotnoise``
            and ``data.shotnoise``
        - randoms.num_per_cell, data.num_per_cell : float
            the mean number of weighted objects per cell for each sample
        - num_per_cell : float
            the mean number of weighted objects per cell

        For further details on the meta-data, see
        :ref:`the documentation <fkp-meta-data>`.

        Returns
        -------
        :class:`~pmesh.pm.RealField` :
            the field object holding the FKP density field in real space
        """

        attrs = {}

        # determine alpha, the weighted number ratio
        for name in self.source.species:
            attrs[name+'.W'] = self.weighted_total(name)

        attrs['alpha'] = attrs['data.W'] / attrs['randoms.W']

        # paint the data
        real = self['data'].to_real_field(normalize=False)
        real.attrs.update(attrs_to_dict(real, 'data.'))
        if self.comm.rank == 0:
            self.logger.info("data painted.")

        if self.source['randoms'].csize > 0:

            # paint the randoms
            rfield2 = self['randoms'].to_real_field(normalize=False)

            if self.comm.rank == 0:
                self.logger.info("randoms painted.")

            real[:] += -1. * attrs['alpha'] * rfield2[:]
            real.attrs.update(attrs_to_dict(rfield2, 'randoms.'))

        # divide by volume per cell to go from number to number density
        vol_per_cell = (self.pm.BoxSize/self.pm.Nmesh).prod()
        real[:] /= vol_per_cell

        if self.comm.rank == 0:
            self.logger.info("volume per cell is %g" % vol_per_cell)

        # remove shot noise estimates (they are inaccurate in this case)
        real.attrs.update(attrs)
        real.attrs.pop('data.shotnoise', None)
        real.attrs.pop('randoms.shotnoise', None)

        return real
