#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    decoder.py
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
    "This module contains the classes defining the decoder " \
    "(:class:`core.decoder.Decoder`) and the layer representing " \
    "the negative binomial distributions used to model the " \
    "expression of the genes " \
    "(:class:`core.decoder.NBLayer`). " \
    "This layer is the output layer of the decoder but, " \
    "because of its complex behavior, is implemented as " \
    "a separate class."


# Standard library
import logging as log
import math
# Third-party packages
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Helper functions --------------------------#


def reshape_scaling_factors(x,
                            out_dim):
    """Reshape the scaling factor (a tensor).

    Parameters
    ----------
    x : ``torch.Tensor``
        The input tensor to be reshaped.

    out_dim : ``int``
        The dimensionality of the output tensor.

    Returns
    -------
    x : ``torch.Tensor``
        The reshaped tensor.
    """
    
    # Get the dimensionality of the input tensor
    start_dim = len(x.shape)

    # For each extra dimension that the output tensor needs to
    # have with respect to the input tensor
    for i in range(out_dim - start_dim):

        # Add a singleton dimension
        x = x.unsqueeze(1)
    
    # Return the reshaped tensor
    return x


#---------------------- Negative binomial layer ----------------------#


class NBLayer(nn.Module):
    
    """
    Class implementing an output layer representing the means
    of the negative binomial distributions modeling the outputs
    (e.g., the means of the gene expression counts). One negative
    binomial distribution with trainable parameters is used for
    each gene.

    The negative binomial models the number of "successes" in a
    sequence of independent and identically distributed Bernoulli
    trials before a specified number of "failures" occur.
    
    A Bernoulli trial is a trial where there are only two
    possible mutually exclusive outcomes. 
    """

    # Supported activation functions
    ACTIVATION_FUNCTIONS = ["sigmoid", "softplus"]

    def __init__(self,
                 dim,
                 r_init,
                 activation = "softplus"):
        """Initialize an instance of the class.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the negative binomial
            distributions.

        r_init : ``int``
            The initial value for ``r``, representing the "number
            of failures" after which the "trials" stop.

        activation : ``str``, ``"softplus"``
            The name of the activation function to be used.
        """
        
        # Initialize the class
        super().__init__()

        # Set the dimensionality of the NBLayer
        self._dim = dim

        # Initialize the value of the log of r. Real-valued positive
        # parameters are usually used as their log equivalent.
        self._log_r = \
            self._get_log_r(r_init = r_init,
                            dim = self.dim)

        # Get the name of the activation that will be used
        self._activation = \
            self._get_activation(activation = activation)


    #-------------------- Initialization methods ---------------------#
    

    def _get_log_r(self,
                   r_init,
                   dim):
        """Get a tensor with dimensions (1, ``dim``) filled with the
        natural logarithm of the initial value of ``r`` ("number of
        failures" after which the "trials" stop).

        Parameters
        ----------
        r_init : ``int``
            The initial value for ``r``, representing the "number
            of failures" after which the "trials" stop.

        dim : ``int``
            The dimensionality of the negative binomial
            distributions.

        Returns
        -------
        ``torch.Tensor``
            Tensor containing the ``r_init`` value as many times
            as the number of dimensions of the space the
            negative binomials live in.
        """

        return \
            nn.Parameter(torch.full(fill_value = math.log(r_init),
                                    size = (1, dim)),
                         requires_grad = True)


    def _get_activation(self,
                        activation):
        """Get the name of the activation function after checking
        that it is supported.

        Parameters
        ----------
        activation : ``str``
            The name of the activation function to be used.
        """
        
        # If the provided activation function is not supported
        if activation not in self.ACTIVATION_FUNCTIONS:

            # Raise an exception
            errstr = \
                f"Unknown 'activation' ({activation}). " \
                f"Supported activation functions are: " \
                f"{', '.join(self.ACTIVATION_FUNCTIONS)}."
            raise ValueError(errstr)

        # Return the name of the activation function
        return activation


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the ``NBLayer``.
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
    def log_r(self):
        """The natural logarithm of the ``r`` values associated
        with the negative binomial distributions (the "number
        of failures" after which the "trials" end).
        """

        return self._log_r


    @log_r.setter
    def log_r(self,
              value):
        """Raise an exception if the user tries to modify
        the value of ``log_r`` after initialization.
        """
        
        errstr = \
            "The value of 'log_r' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def activation(self):
        """The activation function used in the ``NBLayer``,
        which depends on the scaling type.
        """

        return self._activation


    @activation.setter
    def activation(self,
                   value):
        """Raise an exception if the user tries to modify
        the value of ``activation`` after initialization.
        """
        
        errstr = \
            "The value of 'activation' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#


    @staticmethod
    def rescale(means,
                scaling_factors):
        """Rescale the means of the negative binomial
        distributions.

        Parameters
        ----------
        means : ``torch.Tensor``
            A 1D tensor containing the means of the negative
            binomials o be rescaled.

            In the tensor, each value represents the 
            mean of a different negative binomial.

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length is equal to the
            number of scaling factors to be used to rescale
            the means.

        Returns
        -------
        ``torch.Tensor``
            The rescaled means.

            This is a 1D tensor whose length is equal to the number
            of negative binomials whose means were rescaled.
        """
        
        # Return the rescaled values by multiplying the means
        # by the scaling factors
        return means * scaling_factors


    @staticmethod
    def log_prob_mass(k,
                      m,
                      r):
        """Compute the logarithm of the probability mass
        for a set of negative binomial distribution.

        Thr formula used to compute the logarithm of the
        probability mass is:

        .. math::

           logPDF_{NB(k,m,r)} &=
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\ 
           &+ k \\cdot log(m \\cdot c + \\epsilon) +
           r \\cdot log(r \\cdot c)

        Where :math:`\\epsilon` is a small value to prevent
        underflow/overflow, and :math:`c` is equal to
        :math:`\\frac{1}{r+m+\\epsilon}`.

        The derivation of this formula from the non-logarithmic
        formulation of the probability mass function of the
        negative binomial distribution can be found below.

        Parameters
        ----------
        k : ``torch.Tensor``
            "Number of successes" seen before
            stopping the trials (each value in the
            tensor corresponds to the number of successes
            of a different negative binomial).

        m : ``torch.Tensor``
            Means of the negative binomials (each value in
            the tensor corresponds to the mean of a
            different negative binomial).

        r : ``torch.Tensor``
            "Number of failures" after which the trials
            end (each value in the tensor corresponds to
            the number of failures of a different negative
            binomial).

        Returns
        -------
        x : ``torch.Tensor``
            The log-probability mass of the negative binomials.
            This is a 1D tensor whose length corresponds to the
            number of negative binomials distributions considered,
            and each value in the tensor corresponds to the
            log-probability mass of a different negative binomial.

        Notes
        -----
        Here, we show how we derived the formula for the logarithm
        of the probability mass of the negative binomial
        distribution.

        We start from the non-logarithmic version of the
        probability mass for the negative binomial, which is:

        .. math::

           PDF_{NB(k,m,r)} = \
           \\binom{k+r-1}{k} (1-p)^{k} p^{r}

        However, since:

        * :math:`1-p` is equal to :math:`\\frac{m}{r+m}`
        * :math:`p` is equal to :math:`\\frac{r}{r+m}`
        * :math:`k+r-1` can be rewritten in terms of the
          gamma function as :math:`\\Gamma(k+r)`
        * :math:`k` can also be rewritten as
          :math:`\\Gamma(r) \\cdot k!`

        The formula becomes:

        .. math::

           PDF_{NB(k,m,r)} = \
           \\binom{\\Gamma(k+r)}{\\Gamma(r) \\cdot k!}
           \\left( \\frac{m}{r+m} \\right)^k
           \\left( \\frac{r}{r+m} \\right)^r

        However, :math:`k!` can be also be rewritten as
        :math:`\\Gamma(k+1)`, resulting in:

        .. math::

           PDF_{NB(k,m,r)} = \
           \\binom{\\Gamma(k+r)}{\\Gamma(r) \\cdot 
           \\Gamma(k+1)}
           \\left( \\frac{m}{r+m} \\right)^k
           \\left( \\frac{r}{r+m} \\right)^r

        Then, we get the natural logarithm:
        
        .. math::

           logPDF_{NB(k,m,r)} &= \
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
           &+ k \\cdot log \\left( \\frac{m}{r+m} \\right) +
           r \\cdot log \\left( \\frac{r}{r+m} \\right)
        
        Here, we are adding a small value :math:`\\epsilon` to
        prevent underflow/overflow:

        .. math::

           logPDF_{NB(k,m,r)} &= \
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
           &+ k \\cdot
           log \\left( m \\cdot \\frac{1}{r+m+\\epsilon} 
           + \\epsilon \\right) +
           r \\cdot
           log \\left( r \\cdot \\frac{1}{r+m+\\epsilon}
           \\right)

        Finally, we substitute :math:`\\frac{1}{r+m+\\epsilon}`
        with :math:`c` and we obtain:

        .. math::

           logPDF_{NB(k,m,r)} &= \
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
           &+ k \\cdot
           log \\left( m \\cdot c + \\epsilon \\right) +
           r \\cdot
           log \\left( r \\cdot c \\right)
        """

        # Convert the "number of successes" to a double-precision
        # floating point number
        k = k.double()
        
        # Set a small value used to prevent underflow and
        # overflow
        eps = 1.e-10
        
        # Set a constant used later in the equation defining
        # the log-probability mass
        c = 1.0 / (r + m + eps)
        
        # Get the log-probability mass of the negative binomials.
        #
        # The non-log version would be:
        #
        # NB(k,m,r) = \
        #   gamma(k+r) / (gamma(r) * k!) *
        #   (m/(r+m))^k *
        #   (r/(r+m))^r
        #
        # Since k! can be rewritten as gamma(k+1):
        #
        # NB(k,m,r) = \
        #   gamma(k+r) / (gamma(r) * gamma(k+1)) *
        #   (m/(r+m))^k *
        #   (r/(r+m))^r
        #
        # Getting the natural logarithm:
        #
        # log(NB(k,m,r)) = \
        #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
        #   k * log(m * 1/(r+m)) +
        #   r * log(r * 1/(r+m))
        #
        # Here, we are adding the small ``eps`` to
        # prevent underflow/overflow:
        #
        # log(NB(k,m,r)) = \
        #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
        #   k * log(m * 1/(r+m+eps) + eps) +
        #   r * log(r * 1/(r+m+eps))
        #
        # Substituting 1/(r+m+eps) with c:
        #
        # log(NB(k,m,r)) = \
        #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
        #   k * log(m * c + eps) +
        #   r * log(r * c)
        x = \
            torch.lgamma(k+r) - torch.lgamma(r) - \
            torch.lgamma(k+1) + k*torch.log(m*c+eps) + \
            r*torch.log(r*c)
        
        # Return the log-probability mass
        return x


    def forward(self,
                x):
        """Forward pass. Pass the input tensor through the
        activation function and return the result.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor.

        Returns
        -------
        ``torch.Tensor``
            A tensor containing the result of passing the input
            tensor through the activation function. This tensor
            has the same shape as the input tensor.
        """
        
        # If the activation function is a sigmoid
        if self.activation == "sigmoid":
            
            # Pass the input through the sigmoid
            # function
            return torch.sigmoid(x)
        
        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Pass the input through the softplus
            # function
            return F.softplus(x)
    

    def log_prob(self,
                 obs_counts,
                 pred_means,
                 scaling_factors):
        """Get the log-probability mass.

        Parameters
        ----------
        obs_counts : ``torch.Tensor``
            The observed gene counts.

            The first dimension of this tensor must have a length
            equal to the number of samples whose counts are
            reported.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``obs_counts`` and
            ``pred_means``.
        
        Returns
        -------
        ``torch.Tensor``
            The log-probability mass.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """
        
        return self.__class__.log_prob_mass(\
                    k = obs_counts,
                    m = self.__class__.rescale(\
                            means = pred_means,
                            scaling_factors = scaling_factors),
                    r = torch.exp(self.log_r))


    def loss(self,
             obs_counts,
             pred_means,
             scaling_factors):
        """Compute the loss given observed the means ``obs_counts``
        and predicted means ``pred_mean``, the latter rescaled by
        ``scaling_factors``.

        The loss corresponds to the negative log-probability density.

        Parameters
        ----------
        obs_counts : ``torch.Tensor``
            The observed gene counts.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``obs_counts`` and
            ``pred_means``.

        Returns
        -------
        ``torch.Tensor``
            The loss associated with the input ``x``.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """  
            
        # Return a tensor with as many values as the
        # dimensions of the input 'x' (the loss for each
        # of the negative binomials associated with 'x')
        return - self.log_prob(obs_counts = obs_counts,
                               pred_means = pred_means,
                               scaling_factors = scaling_factors)


    def sample(self,
               n,
               pred_means,
               scaling_factors):
        """Get ``n`` samples from the negative binomials.

        Parameters
        ----------
        n : ``int``
            The number of samples to get.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

        scaling_factors : ``torch.Tensor``
            A tensor containing the scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``pred_means``.
        
        Returns
        -------
        ``torch.Tensor``
            The samples drawn from the negative binomial distributions.
            
            The shape of this tensor depends on the shape of ``n``
            and ``pred_means``, but the first dimension always
            has a length equal to the number of samples drawn from
            the negative binomial distribution.
        """
        
        # Disable gradient calculation
        with torch.no_grad():
            
            # Rescale the means
            m = self.__class__.rescale(\
                    means = pred_means,
                    scaling_factors = scaling_factors)
            
            # Get the probabilities from the means
            # m = p * r / (1-p), so p = m / (m+r)
            probs = m / (m + torch.exp(self.log_r))

            # Sample from the negative binomials with the
            # calculated probabilities
            nb = \
                dist.NegativeBinomial(\
                    torch.exp(self.log_r),
                    probs = probs)
            
            # Get 'n' samples from the distributions
            return nb.sample([n]).squeeze()


#------------------------------ Decoder ------------------------------#


class Decoder(nn.Module):
    
    """
    Class implementing the decoder.
    """

    def __init__(self,
                 n_units_input_layer,
                 n_units_hidden_layers,
                 n_units_output_layer,
                 r_init = 2,
                 activation_output = "softplus"):
        """Initialize an instance of the neural network representing
        the decoder.

        Parameters
        ----------
        n_units_input_layer : ``int``
            The mumber of neurons in the input layer.

        n_units_hidden_layers : ``list``
            The number of units in each of the hidden layers. As
            many hidden layers as the number of items in the list
            will be created.

        n_units_output_layer : ``init``
            The number of units in the output layer.

        r_init : ``int``
            The initial value for ``r``, representing the "number
            of failures" after which the "trials" stop in the
            ``NBLayer``.

        activation_output : ``str``, {``"sigmoid"``, ``"softplus"``}
            The name of the activation function to be used in
            the output layer.
        """

        # Initialize the class
        super().__init__()

        # Create the layers
        self.main = \
            self._get_layers(\
                n_units_input_layer = n_units_input_layer,
                n_units_hidden_layers = n_units_hidden_layers,
                n_units_output_layer = n_units_output_layer)

        # Create the output layer
        self.nb = \
            NBLayer(dim = n_units_output_layer,
                    r_init = r_init,
                    activation = activation_output)


    #-------------------- Initialization methods ---------------------#


    def _get_layers(self,
                    n_units_input_layer,
                    n_units_hidden_layers,
                    n_units_output_layer):
        """Get the decoder's layers.

        Parameters
        ----------
        n_units_input_layer : ``int``
            The mumber of neurons in the input layer.

        n_units_hidden_layers : ``list``
            The number of units in each of the hidden layers. As
            many hidden layers as the number of items in the list
            will be created.

        n_units_output_layer : ``init``
            The number of units in the output layer.

        Returns
        -------
        ``torch.nn.ModuleList``
            The list of layers.
        """

        # Create a ModuleList to store the layers
        layers = nn.ModuleList()

        # Get number of groups of connections (one group of
        # connections connect two layers, so they are one
        # less than the total number of layers)
        n_connects = 1 + len(n_units_hidden_layers)

        # For each group of connections
        for n_connect in range(n_connects):

            # If it is the first hidden layer
            if n_connect == 0:

                # Add full connections between the input layer and
                # the first hidden layer using the ReLu activation
                # function
                layers.extend(\
                    [nn.Linear(n_units_input_layer,
                               n_units_hidden_layers[n_connect]),
                     nn.ReLU(True)])
                
                # Set the previous number of units (used in the
                # next step of the loop) as the number of units in
                # the first hidden layer
                prev_n_units = n_units_hidden_layers[n_connect]

                # Go to the next step
                continue

            # If it is the last hidden layer
            elif n_connect == n_connects-1:

                # Add full connections between the last hidden
                # layer and the output layer using the ReLu
                # activation function
                layers.append(\
                    nn.Linear(prev_n_units, 
                              n_units_output_layer))

                # Return the list of layers
                return layers

            # If it is an intermediate hidden layer
            else:
                
                # Add full connections between the previous
                # hidden layer and the current hidden layer
                # using the ReLu activation function
                layers.extend(\
                    [nn.Linear(\
                        prev_n_units,
                        n_units_hidden_layers[n_connect]),
                     nn.ReLU(True)])

                # Set the previous number of units (used in the
                # next step of the loop) as the number of units
                # in the current hidden layer
                prev_n_units = n_units_hidden_layers[n_connect]


    #------------------------ Public methods -------------------------#


    def forward(self,
                z):
        """Perform the forward pass through the neural network.

        Parameters
        ----------
        z : ``torch.Tensor``
            A tensor holding the representations to pass
            through the decoder.

        Returns
        -------
        ``torch.Tensor``
            A tensor holding the outputs of the decoder
            for the given representations.
        """

        # For each layer of the neural network
        for i in range(len(self.main)):
            
            # Pass through the layer and find the
            # intermediate (or final) representation
            z = self.main[i](z)

        # Return the final representations
        return self.nb(z)