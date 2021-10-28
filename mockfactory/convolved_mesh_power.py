import numpy
import logging
import time
import warnings

from pmesh.pm import ComplexField
from nbodykit import CurrentMPIComm
from nbodykit.utils import timer
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.source.mesh.catalog import get_compensation
from nbodykit.algorithms.fftpower import project_to_basis, _find_unique_edges
from nbodykit.algorithms.convpower.fkp import ConvolvedFFTPower, get_real_Ylm, copy_meta


class ConvolvedMeshFFTPower(ConvolvedFFTPower):

    logger = logging.getLogger('ConvolvedMeshFFTPower')

    def __init__(self, first, poles,
                    second=None,
                    kmin=0.,
                    kmax=None,
                    dk=None,
                    norm=None):

        if not isinstance(first, FieldMesh):
            attrs = getattr(first,'attrs',{})
            first = FieldMesh(first)
            first.attrs.update(attrs)
        if second is None:
            second = first
        if not isinstance(second, FieldMesh):
            attrs = getattr(second,'attrs',{})
            second = FieldMesh(second)
            second.attrs.update(attrs)

        self.first = first
        self.second = second

        # grab comm from first source
        self.comm = first.comm

        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"

        # make a list of multipole numbers
        if numpy.isscalar(poles):
            poles = [poles]

        # store meta-data
        self.attrs = {}
        self.attrs['poles'] = poles
        self.attrs['dk'] = dk
        self.attrs['kmin'] = kmin
        self.attrs['kmax'] = kmax
        if norm is not None:
            self.attrs['norm'] = norm
        else:
            self.attrs['norm'] = self.first.attrs['norm']

        # store BoxSize and BoxCenter from source
        self.attrs['Nmesh'] = self.first.attrs['Nmesh'].copy()
        self.attrs['BoxSize'] = self.first.attrs['BoxSize']
        self.attrs['BoxCenter'] = self.first.attrs['BoxCenter']

        # and run
        self.run()

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
            if mesh.attrs.get('compensated',False):
                compensation[name] = get_compensation(mesh.attrs.get('interlaced',False),mesh.attrs['resampler'])
            else:
                compensation[name] = None
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
        rfield1 = self.first.field
        meta1 = self.first.attrs.copy()

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
            rfield2 = self.second.field
            meta2 = self.second.attrs.copy()

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
                Aell[islab,...] = 1./self.attrs['norm'] * A0_1[islab] * Aell[islab].conj()

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
                A0_1[islab,...] = 1./self.attrs['norm']*A0_1[islab]*A0_2[islab].conj()

            # the 1D monopole
            proj_result, _ = project_to_basis(A0_1, edges)
            result['power_0'][:] = numpy.squeeze(proj_result[2])

        # save the number of modes and k
        result['k'][:] = numpy.squeeze(proj_result[0])
        result['modes'][:] = numpy.squeeze(proj_result[-1])

        # compute shot noise
        self.attrs['shotnoise'] = self.first.attrs.get('shotnoise',0.)
        #self.attrs['shotnoise'] = 3000.

        # copy over any painting meta data
        if self.first is self.second:
            copy_meta(self.attrs, meta1)
        else:
            copy_meta(self.attrs, meta1, prefix='first')
            copy_meta(self.attrs, meta2, prefix='second')

        return result
