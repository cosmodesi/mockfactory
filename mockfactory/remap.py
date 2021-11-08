"""
A vectorisation of Duncan Campbell's script https://github.com/duncandc/cuboid_remap,
based on Jordan Carlson and Martin White algorithm of arXiv:1003.3178.
"""

import logging
import itertools

import numpy as np


from .utils import BaseClass


def vec3(*args):
    """Create 3D vector from input ``args``."""
    if len(args) == 1: args = args[0]
    return np.asarray(args, dtype='f8')


def dot(u, v):
    """Dot product between vectors ``u`` and ``v``."""
    return sum(uu*vv for uu,vv in zip(u,v))


def square(v):
    """Square norm of vector ``v``."""
    return dot(v,v)


def norm(v):
    """Norm of vector ``v``."""
    return np.sqrt(square(v))


def det3(u, v, w):
    """Determinant of 3x3 matrix formed by input 3-vectors ``u``, ``v`` and ``w``."""
    return u[0]*(v[1]*w[2] - v[2]*w[1]) + u[1]*(v[2]*w[0] - v[0]*w[2]) + u[2]*(v[0]*w[1] - v[1]*w[0])


def orthogonalize(u1, u2, u3):
    """Orthognalize input base of 3-vectors."""
    u1, u2, u3 = vec3(u1), vec3(u2), vec3(u3)
    s1 = square(u1)
    s2 = square(u2)
    d12 = dot(u1,u2)
    d23 = dot(u2,u3)
    d13 = dot(u1,u3)
    alpha = -d12/s1
    gamma = -(alpha*d13 + d23)/(alpha*d12 + s2)
    beta = -(d13 + gamma*d12)/s1
    return [u1, u2 + alpha*u1, u3 + beta*u1 + gamma*u2]


class Plane(BaseClass):

    """Plane in 3-D cartesian space."""

    def __init__(self, p, n):
        """
        Initialize :class:`Plane`.

        Parameters
        ----------
        p : array_like
            A point in a plane.

        n : array_like
            A vector normal to the plane.
        """
        self.n = np.asarray(n)
        self.d = -dot(p,n)

    @property
    def normal(self):
        """Unit vector normal to the plane."""
        return self.n/norm(self.n)

    def test(self, position):
        """
        Compare points to a plane.

        Parameters
        ----------
        position : array_like (N, 3)
             Cartesian positions.

        Returns
        -------
        above : array (N,)
            Positive, negative, or zero depending on whether
            the point lies above, below, or on the plane.
        """
        return dot(self.n,position.T) + self.d

    def test_unit_cube(self):
        """
        Compare a unit cube to the plane.

        Returns
        -------
        u : int
            +1 if the unit cube is above, -1 if below, or 0 if intersecting the plane.
        """
        position = vec3(list(itertools.product((0,1),(0,1),(0,1))))
        s = self.test(position)
        above = (s>0).any()
        below = (s<0).any()
        return int(above) - int(below)


class Cell(BaseClass):
    """
    A cell, i.e. the intersection between the cuboid and a tile (replication of the unit cube).
    Convex polyhedron bounded by 12 planes: the 6 faces of the tile and the 6 faces of the cuboid.
    """
    def __init__(self, ipos=(0,0,0)):
        """
        Initialize :class:`Cell`.

        Parameters
        ----------
        ipos : array_like
            Label of the tile where the cell lives in.
        """
        self.ipos = vec3(ipos)
        self.faces = []

    def isin(self, position):
        """
        Test whether input ``position`` is in cell.
        """
        mask = np.ones(len(position), dtype=np.bool_)
        for face in self.faces:
            mask &= face.test(position) >= 0
        return mask

    def __repr__(self):
        return 'Cell at {} with {:d} non-trivial planes'.format(self.ipos, len(self.faces))


class Cuboid(BaseClass):

    """Cuboid remapping class."""

    def __init__(self, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        """
        Initialize :class:`Cuboid`.

        Parameters
        ----------
        u1 : array_like of shape (3,)
            Lattice vector.

        u2 : array_like of shape (3,)
            Lattice vector.

        u3 : array_like of shape (3,)
            Lattice vector.

        Note
        ----
        ``u1``, ``u2``, ``u3`` must form an unimodular invertable 3x3 integer matrix;
        i.e. ``det3(u1, u2, u3) == 1.``.
        """
        if det3(u1, u2, u3) != 1.:
            raise ValueError('Invalid lattice vectors (u1, u2, u3) = ({}, {}, {})'.format(u1, u2, u3))

        self.e1, self.e2, self.e3 = orthogonalize(u1, u2, u3)

        self.log_info('Orthogonal base (e1, e2, e3) = ({}, {}, {})'.format(self.e1, self.e2, self.e3))

        self.l1 = norm(self.e1)
        self.l2 = norm(self.e2)
        self.l3 = norm(self.e3)
        self.n1 = self.e1/self.l1
        self.n2 = self.e2/self.l2
        self.n3 = self.e3/self.l3
        self.cells = []

        v0 = vec3(0,0,0)
        # Coordinates of the 8 vertices of the cuboid
        self.v = [v0,
                  v0 + self.e3,
                  v0 + self.e2,
                  v0 + self.e2 + self.e3,
                  v0 + self.e1,
                  v0 + self.e1 + self.e3,
                  v0 + self.e1 + self.e2,
                  v0 + self.e1 + self.e2 + self.e3]

        # Compute bounding box of cuboid
        vmin = np.min(self.v, axis=0)
        vmax = np.max(self.v, axis=0)

        # Extend to nearest integer coordinates
        iposmin = np.floor(vmin).astype(int)
        iposmax = np.ceil(vmax).astype(int)

        # Determine which cells (and which faces within those cells) are non-trivial
        iranges = [np.arange(imin, imax) for imin,imax in zip(iposmin,iposmax)]
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
                self.log_debug('Skipping cell at {}'.format(ipos))
                continue
            self.cells.append(cell)
            self.log_debug('Adding cell at {}'.format(ipos))

        # For the identity remapping, use exactly one cell
        if len(self.cells) == 0:
            self.cells.append(Cell())

        # Print the full list of cells
        self.log_debug('Found {:d} non-empty cells:'.format(len(self.cells)))
        for cell in self.cells:
            self.log_debug(str(cell))

    def transform(self, position, boxsize=1.):
        """
        Transform input positions in box to remapped positions.

        Parameters
        ----------
        position : array_like (N, 3)
            Cartesian positions in box, in [0, boxsize].

        boxsize : float, array
            Box size.

        Returns
        -------
        toret : array_like (N, 3)
            Remapped positions.
        """
        position = vec3(position)
        isscalar = position.ndim == 1
        position = np.atleast_2d(position)
        if position.shape[-1] != 3:
            raise ValueError('Input position should be of shape (...,3)')
        position /= boxsize
        mask = np.zeros(position.shape[0], dtype=np.bool_)
        for cell in self.cells:
            mask_ = cell.isin(position)
            mask |= mask_
            position[mask_] += cell.ipos
        if not mask.all():
            raise RuntimeError('Elements not contained in any cell')
        toret = vec3([dot(position.T,n) for n in [self.n1,self.n2,self.n3]]).T * boxsize
        if isscalar:
            toret = toret[0]
        return toret

    def inverse_transform(self, position, boxsize=1.):
        """
        Transform remapped positions to positions in box.

        Parameters
        ----------
        position : array_like (N, 3)
            Cartesian remapped positions.

        boxsize : float, array
            Box size.

        Returns
        -------
        toret : array_like (N, 3)
            Positions in box.
        """
        position = vec3(position)
        isscalar = position.ndim == 1
        position = np.atleast_2d(position)
        if position.shape[-1] != 3:
            raise ValueError('Input position should be of shape (...,3)')
        position /= boxsize
        position = vec3([dot(position.T,n) for n in [self.n1,self.n2,self.n3]]).T
        toret = (position % 1) * boxsize
        if isscalar:
            toret = toret[0]
        return toret

    @classmethod
    def generate_lattice_vectors(cls, maxint=2, maxcomb=None):
        """
        Generate lattice vectors.
        Brute-force approach, fast enough for reasonable ``maxint``.

        Parameters
        ----------
        maxint : int
            Maximum integer coordinate.

        maxcomb : int, default=None
            Maximum number of combinations.
        """
        triplets = itertools.product(*[itertools.product(*[range(-maxint, maxint+1) for i in range(3)]) for i in range(3)])
        toret = {}
        for u in triplets:
            if det3(*u) == 1:
                # calculate cuboid side lengths
                box = tuple(sorted([norm(e) for e in orthogonalize(*u)])[::-1])
                if box not in toret:
                    toret[box] = [u]
                elif maxcomb is None or len(toret[box]) < maxcomb:
                    toret[box].append(u)
        return toret
