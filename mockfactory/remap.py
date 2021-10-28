import sys
import logging
import itertools

import numpy as np


"""This is just a vectorisation of Duncan Campbell's script https://github.com/duncandc/cuboid_remap,
base on Jordan Carlson and Martin White of arXiv:1003.3178."""


def vec3(*args):
    if len(args) == 1: args = args[0]
    return np.array(args,dtype=np.float64)

def dot(u, v):
    return np.sum([u_*v_ for u_,v_ in zip(u,v)],axis=0)

def square(v):
    return dot(v,v)

def norm(v):
    return np.sqrt(square(v))

def triple_scalar_product(u, v, w):
    return u[0]*(v[1]*w[2] - v[2]*w[1]) + u[1]*(v[2]*w[0] - v[0]*w[2]) + u[2]*(v[0]*w[1] - v[1]*w[0])


class Plane(object):

    """Plane in 3-D cartesian space."""

    def __init__(self, p, n):
        """
        Parameters
        ----------
        p : array_like
            a point in a plane
        n : array_like
            a vector normal to the plane
        """
        self.n = np.asarray(n)
        self.d = -dot(p,n)

    @property
    def normal(self):
        """Unit vector normal to the plane"""
        return self.n/norm(self.n)

    def test(self, position):
        """
        Compare points to a plane.

        Parameters
        ----------
        position : array_like (N,3)
             positions

        Returns
        -------
        above : float
            value is positive, negative, or zero depending on whether
            the point lies above, below, or on the plane.
        """
        return dot(self.n,position.T) + self.d

    def test_unit_cube(self):
        """
        Compare a unit cube to the plane.

        Returns
        -------
        u : int
            [+1, 0, -1] if the unit cube is above, below, or intersecting the plane.
        """
        position = vec3(list(itertools.product((0,1),(0,1),(0,1))))
        s = self.test(position)
        above = (s>0).any()
        below = (s<0).any()
        return int(above) - int(below)


class Cell(object):

    """A cell."""

    def __init__(self, ipos=(0,0,0)):
        self.ipos = vec3(ipos)
        self.faces = []

    def isin(self, position):
        mask = np.ones(len(position),dtype=np.bool_)
        for face in self.faces:
            mask &= face.test(position) >= 0
        return mask

    def __repr__(self):
        return 'Cell at {} with {:d} non-trivial planes'.format(self.ipos,len(self.faces))


class Cuboid(object):

    """Cuboid remapping class."""

    logger = logging.getLogger('Cuboid')

    def __init__(self, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        """
        Parameters
        ----------
        u1 : array_like, (3,)
            lattice vector
        u2 : array_like, (3,)
            lattice vector
        u3 : array_like, (3,)
            lattice vector

        Note
        ----
        ``u1``, ``u2``, ``u3`` form an unimodular invertable 3x3 integer matrix.
        """
        self.e1,self.e2,self.e3 = self.new_base(u1,u2,u3)

        self.logger.info('e1 = {}, e2 = {}, e3 = {}'.format(self.e1,self.e2,self.e3))

        self.L1 = norm(self.e1)
        self.L2 = norm(self.e2)
        self.L3 = norm(self.e3)
        self.n1 = self.e1/self.L1
        self.n2 = self.e2/self.L2
        self.n3 = self.e3/self.L3
        self.cells = []

        v0 = vec3(0,0,0)
        self.v = [v0,
                  v0 + self.e3,
                  v0 + self.e2,
                  v0 + self.e2 + self.e3,
                  v0 + self.e1,
                  v0 + self.e1 + self.e3,
                  v0 + self.e1 + self.e2,
                  v0 + self.e1 + self.e2 + self.e3]

        # Compute bounding box of cuboid
        vmin = np.min(self.v,axis=0)
        vmax = np.max(self.v,axis=0)

        # Extend to nearest integer coordinates
        iposmin = np.floor(vmin).astype(int)
        iposmax = np.ceil(vmax).astype(int)

        # Determine which cells (and which faces within those cells) are non-trivial
        iranges = [np.arange(min_,max_) for min_,max_ in zip(iposmin,iposmax)]
        for ipos in itertools.product(*iranges):
            shift = -vec3(ipos)
            faces = [Plane(self.v[0] + shift, +self.n1),
                    Plane(self.v[4] + shift, -self.n1),
                    Plane(self.v[0] + shift, +self.n2),
                    Plane(self.v[2] + shift, -self.n2),
                    Plane(self.v[0] + shift, +self.n3),
                    Plane(self.v[1] + shift, -self.n3)]
            cell = Cell(ipos)
            skipcell = False
            for face in faces:
                r = face.test_unit_cube()
                if r == 1:
                    # Unit cube is completely above this plane; this cell is empty
                    continue
                if r == 0:
                    # Unit cube intersects this plane; keep track of it
                    cell.faces.append(face)
                if r == -1:
                    skipcell = True
                    break
            if skipcell or len(cell.faces) == 0:
                self.logger.info('Skipping cell at {}'.format(ipos))
                continue
            else:
                self.cells.append(cell)
                self.logger.info('Adding cell at {}'.format(ipos))

        # For the identity remapping, use exactly one cell
        if len(self.cells) == 0:
            self.cells.append(Cell())

        # Print the full list of cells
        self.logger.info('Found {} non-empty cells:'.format(len(self.cells)))
        for cell in self.cells:
            self.logger.info(str(cell))

    def transform(self, position, BoxSize=None):
        position = vec3(position)
        isscalar = position.ndim == 1
        position = np.atleast_2d(position)
        if position.shape[-1] != 3:
            raise ValueError('Input position should be of shape (...,3)')
        if BoxSize is not None:
            position /= BoxSize
        mask = np.zeros(position.shape[0],dtype=np.bool_)
        for cell in self.cells:
            mask_ = cell.isin(position)
            mask |= mask_
            position[mask_] += cell.ipos
        if not mask.all():
            raise RuntimeError('Elements not contained in any cell')
        toret = vec3([dot(position.T,n) for n in [self.n1,self.n2,self.n3]]).T
        if BoxSize is not None:
            toret *= BoxSize
        if isscalar:
            toret = toret[0]
        return toret

    def inverse_transform(self, position, BoxSize=None):
        position = vec3(position)
        isscalar = position.ndim == 1
        position = np.atleast_2d(position)
        if position.shape[-1] != 3:
            raise ValueError('Input position should be of shape (...,3)')
        if BoxSize is not None:
            position /= BoxSize
        position = vec3([dot(position.T,n) for n in [self.n1,self.n2,self.n3]]).T
        toret = np.fmod(position, 1) + (position < 0)
        if BoxSize is not None:
            toret *= BoxSize
        if isscalar:
            toret = toret[0]
        return toret

    @classmethod
    def new_base(cls, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        u1 = vec3(u1)
        u2 = vec3(u2)
        u3 = vec3(u3)
        if triple_scalar_product(u1, u2, u3) != 1.:
            raise ValueError('Invalid lattice vectors: u1 = {}, u2 = {}, u3 = {}'.format(u1,u2,u3))

        s1 = square(u1)
        s2 = square(u2)
        d12 = dot(u1,u2)
        d23 = dot(u2,u3)
        d13 = dot(u1,u3)
        alpha = -d12/s1
        gamma = -(alpha*d13 + d23)/(alpha*d12 + s2)
        beta = -(d13 + gamma*d12)/s1
        return [u1, u2 + alpha*u1, u3 + beta*u1 + gamma*u2]

    @classmethod
    def new_cuboid(cls, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        return [norm(e) for e in cls.new_base(u1,u2,u3)]

    @classmethod
    def generate_lattice_vectors(cls, maxint=2, maxcomb=sys.maxsize):
        """Brute-force approach, fast enough for reasonable ``maxint``."""
        triplets = itertools.product(*[itertools.product(*[range(-maxint,maxint+1) for i in range(3)]) for i in range(3)])
        toret = {}
        for u in triplets:
            if triple_scalar_product(*u) == 1:
                # calculate cuboid side lengths
                box = tuple(sorted(cls.new_cuboid(*u))[::-1])
                if box not in toret:
                    toret[box] = [u]
                elif len(toret[box]) < maxcomb:
                    toret[box].append(u)
        return toret
