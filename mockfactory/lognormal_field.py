import logging

import numpy as np

from nbodykit.mpirng import MPIRandomState
import mpsort
from nbodykit.source.catalog import ArrayCatalog

from .gaussian_field import GaussianFieldMesh


class LogNormalFieldMesh(GaussianFieldMesh):

    def set_displacement_field(self, growth_rate=None):
        iscallable = callable(growth_rate)
        disp_k = [self.pm.create(type='untransposedcomplex') for i in range(self.delta_k.ndim)]
        for i in range(self.delta_k.ndim): disp_k[i][:] = 1j
        slabs = [self.delta_k.slabs.x, self.delta_k.slabs] + [d.slabs for d in disp_k]
        for islabs in zip(*slabs):
            kslab, delta_slab = islabs[:2] # the k arrays and delta slab
            # the square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            zero_idx = k2 == 0.
            k2[zero_idx] = 1. # avoid dividing by zero
            with np.errstate(invalid='ignore',divide='ignore'):
                for i in range(self.delta_k.ndim):
                    disp_slab = islabs[2+i]
                    disp_slab[...] *= kslab[i] / k2 * delta_slab[...]
                    disp_slab[zero_idx] = 0. # no bulk displacement
        self.disp_r = [d.c2r() for d in disp_k]
        if iscallable:
            offset = self.attrs['BoxCenter'] + 0.5*self.pm.BoxSize / self.pm.Nmesh
            for d in self.disp_r:
                for rslab,slab in zip(d.slabs.x,d.slabs):
                    rgrid = [r + o for r,o in zip(rslab,offset)]
                    rnorm = np.sum(rr**2 for rr in rgrid)**0.5
                    slab[...].flat *= growth_rate(rnorm.flatten())
        elif growth_rate is not None:
            for d in self.disp_r:
                d[...] *= growth_rate

    def poisson_sample(self, seed=None, displacement=True, resampler='nnb'):
        # seed1 used for poisson sampling
        # seed2 used for uniform shift within a cell.
        seed1, seed2 = np.random.RandomState(seed).randint(0, 0xfffffff, size=2)

        # mean number of objects per cell
        H = self.pm.BoxSize / self.pm.Nmesh
        dv = np.prod(H)

        # number of objects in each cell (per rank, as a RealField)
        cellmean = (self.delta_r + 1.) * self.nbar*dv

        # create a random state with the input seed
        rng = MPIRandomState(seed=seed1, comm=self.comm, size=self.delta_r.size)

        # generate poissons. Note that we use ravel/unravel to
        # maintain MPI invariane.
        Nravel = rng.poisson(lam=cellmean.ravel())
        N = self.pm.create(type='real')
        N.unravel(Nravel)

        Ntot = N.csum()
        if self.comm.rank == 0:
            self.logger.info("Poisson sampling done, total number of objects is %d" % Ntot)

        pos_mesh = self.pm.generate_uniform_particle_grid(shift=0.0)
        # no need to do decompose because pos_mesh is strictly within the
        # local volume of the RealField.
        N_per_cell = N.readout(pos_mesh, resampler=resampler)
        # fight round off errors, if any
        N_per_cell = np.int64(N_per_cell + 0.5)
        pos = pos_mesh.repeat(N_per_cell, axis=0)

        if displacement:
            disp_mesh = np.empty_like(pos_mesh)
            for i in range(N.ndim):
                disp_mesh[:, i] = self.disp_r[i].readout(pos_mesh, resampler=resampler)
            disp = disp_mesh.repeat(N_per_cell, axis=0)
            del disp_mesh
        else:
            disp = None

        del pos_mesh

        if self.comm.rank == 0:
            self.logger.info("Catalog produced. Assigning in cell shift.")

        # generate linear ordering of the positions.
        # this should have been a method in pmesh, e.g. argument
        # to genereate_uniform_particle_grid(return_id=True);

        # FIXME: after pmesh update, remove this
        orderby = np.int64(pos[:, 0] / H[0] + 0.5)
        for i in range(1, self.delta_r.ndim):
            orderby[...] *= self.delta_r.Nmesh[i]
            orderby[...] += np.int64(pos[:, i] / H[i] + 0.5)

        # sort by ID to maintain MPI invariance.
        pos = mpsort.sort(pos, orderby=orderby, comm=self.comm)
        if displacement:
            disp = mpsort.sort(disp, orderby=orderby, comm=self.comm)

        if self.comm.rank == 0:
            self.logger.info("sorting done")

        rng_shift = MPIRandomState(seed=seed2, comm=self.comm, size=len(pos))
        in_cell_shift = rng_shift.uniform(0, H[i], itemshape=(self.delta_r.ndim,))

        pos[...] += in_cell_shift
        pos[...] %= self.pm.BoxSize
        pos[...] += self.attrs['BoxCenter']

        if self.comm.rank == 0:
            self.logger.info("catalog shifted.")

        return pos, disp

    def to_catalog(self, seed=None, displacement=True, resampler='nnb'):
        position, displacement = self.poisson_sample(seed=seed,displacement=displacement,resampler=resampler)
        source = {'Position':position}
        if displacement is not None:
            source['Displacement'] = displacement
        return ArrayCatalog(source,**self.attrs)
