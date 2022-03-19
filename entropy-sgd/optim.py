from torch.optim import Optimizer
from copy import deepcopy
import numpy as np


class EntropySGD(Optimizer):
    def __init__(self, params, config={}):

        defaults = dict(lr=0.01, momentum=0, damp=0, weight_decay=0,
                        nesterov=True, L=0, eps=1e-4, g0=1e-2, g1=0)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):

        assert (closure is not None) and (model is not None) and \
               (criterion is not None), \
            'attach closure for Entropy-SGD, model and criterion'

        mf, merr = closure()

        # Parameters of the optimizer
        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        # Object self.param_groups is a list of length 1.
        # The unique element (self.param_groups[0]) is a dictionary.
        # The keys are 'params' + config keys (wd, lr, mom, etc).
        # The key 'params' is set by the Optimizer class.
        # self.param_groups[0]['params']: parameters for all the active layers.
        params = self.param_groups[0]['params']

        # The state is created as an empty dictionary by the Optimizer class.
        state = self.state

        # If t is not in state, this is the first iteration. Therefore, we must
        # define the langevin parameters. In order to do that, we must capture
        # the desired dimensions of the tensors
        if 't' not in state:

            # Controls the gamma scooping
            state['t'] = 0

            # List of parameters for each active layer
            state['wc'] = []

            # List of gradients with momentum for each active layer
            state['mdw'] = []

            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))

            state['langevin'] = dict(
                # mw: Approximation of the expected value over the Gibbs
                # distribution. mw is defined by letter \mu on the original
                # paper
                mw=deepcopy(state['wc']),
                # Stores the gradients with momentum, if applicable
                mdw=deepcopy(state['mdw']),
                # eta: Gaussian noise to be added to the gradient
                # (same dimension of mdw)
                eta=deepcopy(state['mdw']),
                # langevin learning rate
                lr=0.1,
                # beta1 = 1 - alpha on the original paper
                # By default, alpha = 0.75
                beta1=0.25)

        lp = state['langevin']

        # Gets values for each step
        for i, w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
        llr, beta1 = lp['lr'], lp['beta1']
        g = g0 * (1 + g1)**state['t']

        for i in range(L):

            # Sample a new batch to calculate the gradient of x'
            # Each new batch accessed is used to approximate the
            # expected value over a Gibbs distribution
            f, _ = closure()

            # loop over each layer of parameters (list with all active layers)
            # wc: parameters of the network (x), not modified by Langevin
            # w: x' parameters on the original paper
            for wc, w, mw, mdw, eta in zip(state['wc'], params,
                                           lp['mw'], lp['mdw'], lp['eta']):

                # Gradient of each layer of parameters
                dw = w.grad.data

                if wd > 0:
                    dw.add_(other=w.data, alpha=wd)
                if mom > 0:
                    # Acumulates the gradiet with momentum
                    mdw.mul_(mom).add_(other=dw, alpha=1 - damp)
                    if nesterov:
                        dw.add_(other=mdw, alpha=mom)
                    else:
                        dw = mdw

                # Add noise
                eta.normal_()
                # Comparing the paper with this equation, we see that to
                # add noise, we need to add a negative factor on eps (-eps)
                dw.add_(other=wc - w.data,
                        alpha=-g).add_(other=eta,
                                       alpha=-eps / np.sqrt(0.5 * llr))

                # Update x' weights
                # Inner loop of SGD
                w.data.add_(other=dw, alpha=-llr)
                # Update the expected value over the Gibbs distribution
                mw.mul_(beta1).add_(other=w.data, alpha=1 - beta1)

        if L > 0:
            # Copy model back
            for i, w in enumerate(params):
                # Replace x' parameters by x parameters
                # State before langevin
                w.data.copy_(state['wc'][i])
                # Gradient of Local Entropy normalized by Gamma
                w.grad.data.copy_(w.data - lp['mw'][i])

        for w, mdw, mw in zip(params, state['mdw'], lp['mw']):
            dw = w.grad.data

            if wd > 0:
                dw.add_(other=w.data, alpha=wd)
            if mom > 0:
                mdw.mul_(mom).add_(other=dw, alpha=1 - damp)
                if nesterov:
                    dw.add_(other=mdw, alpha=mom)
                else:
                    dw = mdw

            # Outer loop of SGD
            w.data.add_(other=dw, alpha=-lr)

        # Update state t for gamma scoping
        if self.state['gamma_scoping']:
            self.state['t'] += 1

        return mf, merr
