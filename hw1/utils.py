from scipy.stats import truncnorm, norm, uniform
import numpy as np


def generate_distribution(size, distribution_type, data_type, min_val=0, max_val=1, **kwargs):
    """
    Generate different types of distribution. Implement uniform, normal and truncated distribution now.
    :param size:Output array size
    :param distribution_type: Distribution type, should be in 'normal', 'truncated_normal' and 'uniform'
    :param data_type: Output data type
    :param min_val: Output seq minimum value
    :param max_val: Output seq maximum value
    :param kwargs: Other input parameter
    :return: Generated sequence
    """
    assert distribution_type in ['uniform', 'normal', 'truncated_normal'], '''distribution type should
                                                                              be 'uniform', 'normal' or 
                                                                              'truncated_normal'''
    assert data_type in ['int', 'float'], "data_type should be 'int' or 'float'"
    if distribution_type == 'uniform':
        if data_type == 'int':
            seq = np.random.randint(low=min_val, high=max_val + 1, size=size)
        else:
            seq = uniform.rvs(loc=min_val, scale=max_val, size=size).astype(float)
    elif type == 'normal':
        assert 'mean' in kwargs.keys(), 'specify the mean of the distribution'
        assert 'std' in kwargs.keys(), 'specify the standard deviation of the distribution'
        if data_type == 'int':
            raise NotImplementedError('hasn\'t implemented normal int')
        else:
            seq = norm.rvs(loc=kwargs['mean'], scale=kwargs['std'], size=size)
    else:
        assert 'mean' in kwargs.keys(), 'specify the mean of the distribution'
        assert 'std' in kwargs.keys(), 'specify the standard deviation of the distribution'
        if data_type == 'int':
            raise NotImplementedError('hasn\'t implemented truncated normal int')
        else:
            seq = truncnorm.rvs(a=(min_val - kwargs['mean']) / kwargs['std'],
                                b=(max_val - kwargs['mean']) / kwargs['std'],
                                loc=kwargs['mean'],
                                scale=kwargs['std'],
                                size=size)
    return seq


def generate_location(size, distribution_type, data_type, min_val, max_val_x, max_val_y, **kwargs):
    """
    Generate initial 2D location.
    :param size:
    :param distribution_type:
    :param data_type:
    :param min_val:
    :param max_val_x:
    :param max_val_y:
    :param kwargs:
    :return: A [size, 2] ndarray. Each row is a location.
    """
    loc_seq = np.zeros((size, 2))
    loc_seq[:, 0] = generate_distribution(size, distribution_type, data_type, min_val, max_val_x, **kwargs)
    loc_seq[:, 1] = generate_distribution(size, distribution_type, data_type, min_val, max_val_y, **kwargs)
    return loc_seq
