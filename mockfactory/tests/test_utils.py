import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from mockfactory.utils import trunccauchy, truncnorm
from mockfactory import setup_logging


def test_rvs():

    x, loc, scale, a, b = np.linspace(-3, 3, 1000), 0.2, 0.8, -0.6, 1.1

    names = ['norm', 'cauchy']
    plt.figure(figsize=(10, 4))

    for ii, name in enumerate(names):
        plt.subplot(1, len(names), ii + 1)
        rv = getattr(stats, name)(loc=loc, scale=scale)
        trv = globals()['trunc' + name](a=a, b=b, loc=loc, scale=scale)
        assert np.allclose(trv.cdf([a - 1., b + 1.]), [0., 1.])
        assert np.allclose(trv.ppf([0., 1.]), [a, b])
        plt.plot(x, trv.pdf(x), label='truncated law')
        plt.plot(x, rv.pdf(x) * trv.pdf(loc) / rv.pdf(loc), label='untruncated law')
        plt.hist(trv.rvs(size=100000), range=(-3, 3), bins=100, alpha=0.6, density=True, label='samples')
        plt.legend()
        plt.title(f'{name} with loc = {loc:.2f} and scale = {scale:.2f}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    setup_logging()
    test_rvs()
