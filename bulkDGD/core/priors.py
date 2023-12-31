#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    priors.py
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#    
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2023 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#                       Viktoria Schuster
#                       <viktoria.schuster@sund.ku.dk>
#                       Inigo Prada Luengo
#                       <inlu@diku.dk>
#                       Anders Krogh
#                       <akrogh@di.ku.dk>
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program. 
#    If not, see <http://www.gnu.org/licenses/>.


# Description of the module
__doc__ = \
    "This module contains the classes implementing some of the " \
    "prior distributions used in the DGD model. The 'softball' " \
    "distribution (:class:`core.priors.SoftballPrior`) is used " \
    "as a prior over the means of the components of the " \
    "Gaussian mixture, while a Gaussian distribution " \
    "(:class:`core.priors.GaussianPrior`) is used as a prior " \
    "over the log-variance of the Gaussian mixture."


# Standard library
import logging as log
import math
# Third-party packages
import torch


# Get the module's logger
logger = log.getLogger(__name__)


class SoftballPrior:
    
    """
    Class implementing a "softball" prior distribution.

    It is an almost uniform prior for an m-dimensional ball, with
    the logistic function making a soft (differentiable) boundary.
    """
    
    def __init__(self,
                 dim,
                 radius,
                 sharpness):
        """Initialize an instance of the softball prior
        distribution.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the distribution.

        radius : ``float``
            The radius of the soft ball.

        sharpness : ``int``
            The sharpness of the soft boundary of the ball.
        """
        
        # Set the dimensionality of the prior
        self._dim = dim

        # Set the radius of the ball
        self._radius = radius

        # Set the sharpness of the boundary
        self._sharpness = sharpness


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the softball distribution.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify
        the value of ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def radius(self):
        """The radius of the soft ball.
        """

        return self._radius


    @radius.setter
    def radius(self,
               value):
        """Raise an exception if the user tries to modify
        the value of ``radius`` after initialization.
        """
        
        errstr = \
            "The value of 'radius' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def sharpness(self):
        """The sharpness of the soft boundary of the ball.
        """

        return self._sharpness


    @sharpness.setter
    def sharpness(self,
                  value):
        """Raise an exception if the user tries to modify
        the value of ``sharpness`` after initialization.
        """
        
        errstr = \
            "The value of 'sharpness' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#
    

    def sample(self,
               n_samples):
        """Get ``n`` random samples from the softball distribution.
        The sampling is uniform from the ``dim``-dimensional ball,
        and approximate.
        
        Parameters
        ----------
        n_samples : ``int``
            The number of samples to be drawn.

        Returns
        -------
        samples : ``torch.Tensor``
            The samples drawn from the softball distribution.
        """

        # Disable gradient calculation
        with torch.no_grad():
            
            # Get a tensor filled with random numbers sampled from a 
            # normal distribution with mean of 0 and a standard
            # deviation of 1 - 'sample' is a tensor with dimensions
            # [n_samples, dim]
            samples = torch.randn((n_samples, self.dim))
            
            # Get the norm of the tensor calculated on the last
            # dimension of the tensor. Retain 'dim' in the output
            # tensor. Divide the first element of the norm by
            # the second element of the norm.
            # In brief, get 'n' random directions.
            samples.div_(samples.norm(dim = -1,
                                      keepdim = True))
            
            # Get 'n' random lengths
            local_len = \
                self.radius * \
                torch.pow(torch.rand((n_samples, 1)), 1.0 / self.dim)
            
            # ???
            samples.mul_(local_len.expand(-1, self.dim))
        
        # Return the new samples
        return samples
    

    def log_prob(self,
                 x):
        """Return the log of the probability density function
        evaluated at ``x``.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor.

        Returns
        -------
        log_prob : ``torch.Tensor``
            The log of the probability density function evaluated
            at ``x``.
        """

        # Compute the norm
        norm = \
            math.lgamma(1 + self.dim * 0.5) - \
            self.dim * (math.log(self.radius) + \
            0.5 * math.log(math.pi))
        
        # Return the log of the probability density function
        return (norm - \
                torch.log(1 + \
                          torch.exp(\
                            self.sharpness * (x.norm(dim = -1) / \
                            self.radius-1))))


class GaussianPrior:
    
    """
    Class implementing a Gaussian prior distribution.
    """

    def __init__(self,
                 dim,
                 mean,
                 stddev):
        """Initialize an instance of the Gaussian distribution.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the distribution.
        
        mean : ``float``
            The mean of the Gaussian distribution.

        stddev : ``float``
            The standard deviation of the Gaussian distribution.
        """

        # Set the dimensionality
        self._dim = dim
        
        # Set the mean
        self._mean = mean

        # Set the standard deviation
        self._stddev = stddev

        # Set a normal distribution with the given mean
        # and standard deviation  
        self._dist = \
            torch.distributions.normal.Normal(self.mean,
                                              self.stddev)


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the Gaussian distribution.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify
        the value of ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def mean(self):
        """The mean of the Gaussian distribution.
        """

        return self._mean


    @mean.setter
    def mean(self,
             value):
        """Raise an exception if the user tries to modify
        the value of ``mean`` after initialization.
        """
        
        errstr = \
            "The value of 'mean' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def stddev(self):
        """The standard deviation of the Gaussian distribution.
        """

        return self._stddev


    @stddev.setter
    def stddev(self,
               value):
        """Raise an exception if the user tries to modify
        the value of ``stddev`` after initialization.
        """
        
        errstr = \
            "The value of 'stddev' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#


    def sample(self,
               n_samples):
        """Sample from the Gaussian distribution.

        Parameters
        ----------
        n_samples : ``int``
            The number of samples to be drawn.

        Returns
        -------
        ``torch.Tensor``
            The samples drawn from the Gaussian distribution.
        """
        
        return self._dist.sample((n_samples, self.dim))

    
    def log_prob(self,
                 x):
        """Return the log of the probability density function
        evaluated at ``x``.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor.

        Returns
        -------
        log_prob : ``torch.Tensor``
            The log of the probability density function evaluated
            at ``x``.
        """
        
        return self._dist.log_prob(x)