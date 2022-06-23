"""
A vectorisation of Duncan Campbell's script https://github.com/duncandc/cuboid_remap,
based on Jordan Carlson and Martin White's algorithm of arXiv:1003.3178.
"""
import itertools

import numpy as np

from .utils import BaseClass


def _make_array(value, shape, dtype='f8'):
    # Return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret


def vec3(*args):
    """Create 3D vector from input ``args``."""
    if len(args) == 1: args = args[0]
    return np.asarray(args, dtype='f8')


def dot(u, v):
    """Dot product between vectors ``u`` and ``v``."""
    return sum(uu * vv for uu, vv in zip(u, v))


def square(v):
    """Square norm of vector ``v``."""
    return dot(v, v)


def norm(v):
    """Norm of vector ``v``."""
    return np.sqrt(square(v))


def det3(u, v, w):
    """Determinant of 3x3 matrix formed by input 3-vectors ``u``, ``v`` and ``w``."""
    return u[0] * (v[1] * w[2] - v[2] * w[1]) + u[1] * (v[2] * w[0] - v[0] * w[2]) + u[2] * (v[0] * w[1] - v[1] * w[0])


def orthogonalize(u1, u2, u3):
    """Orthognalize input base of 3-vectors."""
    u1, u2, u3 = vec3(u1), vec3(u2), vec3(u3)
    s1 = square(u1)
    s2 = square(u2)
    d12 = dot(u1, u2)
    d23 = dot(u2, u3)
    d13 = dot(u1, u3)
    alpha = - d12 / s1
    gamma = - (alpha * d13 + d23) / (alpha * d12 + s2)
    beta = - (d13 + gamma * d12) / s1
    return [u1, u2 + alpha * u1, u3 + beta * u1 + gamma * u2]


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
        self.d = - dot(p, n)

    @property
    def normal(self):
        """Unit vector normal to the plane."""
        return self.n / norm(self.n)

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
        return dot(self.n, position.T) + self.d

    def test_unit_cube(self):
        """
        Compare a unit cube to the plane.

        Returns
        -------
        u : int
            +1 if the unit cube is above, -1 if below, or 0 if intersecting the plane.
        """
        position = vec3(list(itertools.product((0, 1), (0, 1), (0, 1))))
        s = self.test(position)
        above = (s > 0).any()
        below = (s < 0).any()
        return int(above) - int(below)


class Cell(BaseClass):
    """
    A cell, i.e. the intersection between the cuboid and a tile (replication of the unit cube).
    Convex polyhedron bounded by 12 planes: the 6 faces of the tile and the 6 faces of the cuboid.
    """
    def __init__(self, ipos=(0, 0, 0)):
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


class CuboidError(Exception):

    """Error raised when issue with :class:`Cuboid`."""


class Cuboid(BaseClass):

    """Cuboid remapping class."""

    def __init__(self, u1=(1, 0, 0), u2=(0, 1, 0), u3=(0, 0, 1), boxsize=1.):
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

        boxsize : float, array_like of shape (3,)
            Size of the initial box.

        Note
        ----
        ``u1``, ``u2``, ``u3`` must form an unimodular invertable 3x3 integer matrix;
        i.e. ``det3(u1, u2, u3) == 1.``.
        """
        if det3(u1, u2, u3) != 1.:
            raise ValueError('Invalid lattice vectors (u1, u2, u3) = ({}, {}, {})'.format(u1, u2, u3))

        self.e1, self.e2, self.e3 = orthogonalize(u1, u2, u3)

        # self.log_info('Orthogonal base (e1, e2, e3) = ({}, {}, {})'.format(self.e1, self.e2, self.e3))

        self.boxsize = _make_array(boxsize, 3, dtype='f8')
        ebox = np.array([norm(e) for e in [self.e1, self.e2, self.e3]])
        self.n1 = self.e1 / ebox[0]
        self.n2 = self.e2 / ebox[1]
        self.n3 = self.e3 / ebox[2]
        self.cuboidresize = np.array([norm(self.boxsize * n) for n in [self.n1, self.n2, self.n3]])
        self.cuboidsize = self.cuboidresize * ebox

        self.cells = []

        v0 = vec3(0, 0, 0)
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
        iranges = [np.arange(imin, imax) for imin, imax in zip(iposmin, iposmax)]
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

    def transform(self, position):
        """
        Transform input positions in box to remapped positions.

        Parameters
        ----------
        position : array_like (N, 3)
            Cartesian positions in box, in [0, boxsize].

        Returns
        -------
        toret : array_like (N, 3)
            Remapped positions.
        """
        position = np.array(position)
        isscalar = position.ndim == 1
        position = np.atleast_2d(position)
        if position.shape[-1] != 3:
            raise ValueError('Input position should be of shape (...,3)')
        position /= self.boxsize
        mask = np.zeros(position.shape[0], dtype=np.bool_)
        for cell in self.cells:
            mask_ = cell.isin(position)
            mask |= mask_
            position[mask_] += cell.ipos
        if not mask.all():
            raise CuboidError('Elements not contained in any cell')
        toret = vec3([dot(position.T, n) for n in [self.n1, self.n2, self.n3]]).T * self.cuboidresize
        if isscalar:
            toret = toret[0]
        return toret

    def inverse_transform(self, position):
        """
        Transform remapped positions to positions in box.

        Parameters
        ----------
        position : array_like (N, 3)
            Cartesian remapped positions.

        Returns
        -------
        toret : array_like (N, 3)
            Positions in box.
        """
        position = np.array(position)
        isscalar = position.ndim == 1
        position = np.atleast_2d(position)
        position /= self.cuboidresize
        if position.shape[-1] != 3:
            raise ValueError('Input position should be of shape (...,3)')
        position = sum(p[:, None] * n for p, n in zip(position.T, [self.n1, self.n2, self.n3]))
        toret = (position % 1) * self.boxsize
        if isscalar:
            toret = toret[0]
        return toret

    @classmethod
    def generate_lattice_vectors(cls, maxint=2, maxcomb=None, boxsize=1., cuboidranges=None, sort=False):
        """
        Generate lattice vectors.
        Brute-force approach, fast enough for reasonable ``maxint``.

        Parameters
        ----------
        maxint : int
            Maximum integer coordinate.

        maxcomb : int, default=None
            Maximum number of combinations of lattice vectors for each cuboid size.

        boxsize : float, array_like of shape (3,)
            Size of the initial box.

        cuboidranges : array_like
            List of 3 ranges (min, max) for the cuboid size.
            If ``None``, no selection is performed.

        sort : bool, default=False
            Whether to sort cuboid size, such that lattice vectors corresponding to
            e.g. (1, 2, 0.5) and (0.5, 1, 2) will be stored in the (2, 1, 0.5) entry.

        Returns
        -------
        toret : dict
            Dictionary of cuboidsize: list of lattice vectors.
        """
        boxsize = _make_array(boxsize, 3, dtype='f8')
        if not cuboidranges:
            cuboidranges = None
        else:
            if np.ndim(cuboidranges) == 1:
                cuboidranges = [cuboidranges]
            cuboidranges = list(cuboidranges)
            # Fill with unconstraining ranges
            cuboidranges += [(0, np.inf)] * (3 - len(cuboidranges))
        # We can restrict to coprimes, following from:
        # i) determinant calculation (expansion following a culumn/line)
        # ii) unimodularity (determinant is 1)
        # iii) Bezout's theorem (for three integer)
        coprimes = coprime_triples(range(- maxint, maxint + 1))
        triplets = []
        for coprime in coprimes:
            triplets += [t for t in itertools.permutations(coprime)]
        itervectors = []
        if cuboidranges is not None:
            # Take advantage of e = u to restrict from the beginning to the box range
            itervector1 = []
            for e in triplets:
                if cuboidranges[0][0] <= norm(e * boxsize) <= cuboidranges[0][1]:
                    itervector1.append(e)
            itervectors.append(itervector1)
        else:
            itervectors += [triplets]
        itervectors += [triplets] * 2
        triplets = itertools.product(*itervectors)
        toret = {}
        for u in triplets:
            if det3(*u) == 1:
                # Calculate cuboid side lengths
                cuboidsize = tuple(norm(e * boxsize) for e in orthogonalize(*u))
                if cuboidranges is None or (cuboidranges[1][0] <= cuboidsize[1] <= cuboidranges[1][1] and cuboidranges[2][0] <= cuboidsize[2] <= cuboidranges[2][1]):
                    box = cuboidsize
                    if sort:
                        box = tuple(sorted(cuboidsize)[::-1])
                    if box not in toret:
                        toret[box] = [u]
                    elif maxcomb is None or len(toret[box]) < maxcomb:
                        toret[box].append(u)
        return toret


def gcd(*args):
    """Return the greatest common divisor of input integers."""
    # Return self if a single number is passed
    if len(args) == 1:
        return args[0]
    # Pairwise case
    elif len(args) == 2:
        a, b = args
        if a < 0: a = -a
        if b < 0: b = -b
        while(b != 0):
            tmp = b
            b = a % b
            a = tmp
        return a
    # If greater than two arguments, recurse
    a = args[0]
    b = gcd(*args[1:])
    return gcd(a, b)


def coprime_triples(range):
    """
    Return all integer coprime triples within a range.

    Parameters
    ----------
    range : array_like
        Sequence of integers.

    Returns
    -------
    toret : set
        A set of coprime triples.
    """
    toret = set()

    for i in range:
        for j in range:
            # If a pair is coprime, a triple must be coprime
            if gcd(i, j) == 1:
                for k in range:
                    toret.add(tuple(sorted([i, j, k])))
            # If not, check to see if triple is coprime
            else:
                for k in range:
                    if gcd(i, j, k) == 1:
                        toret.add(tuple(sorted([i, j, k])))

    return toret
