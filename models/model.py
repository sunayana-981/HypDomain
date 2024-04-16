import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import networks
import copy
import numpy as np

class ERM(nn.module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, args):
        super(ERM, self).__init__(input_shape, num_classes, num_domains)
        self.featurizer = networks.Featurizer(input_shape)
        self.classifier,out = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes)
        self.final_domain = networks.final(out,num_domains)
        self.final_classes = networks.final(out,num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-4,
        )
        self.num_domains = num_domains
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class IDFM(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes
        self.num_domains = num_domains

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        # all_x = torch.cat([x for x, y in minibatches])
        # # labels
        # all_y = torch.cat([y for _, y in minibatches])
        images,class_labels,domain_labels = minibatches
        # one-hot labels
        all_o = torch.nn.functional.one_hot(class_labels, self.num_domains)
        # features
        #all_f = self.featurizer(images)
        all_f = self.network(images)
        # predictions
        #all_p = self.classifier(all_f)
        all_p = self.final_domain(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.final_classes(all_f_muted)

        # Section 3.3: Batch Percentage
        # all_s = F.softmax(all_p, dim=1)
        # all_s_muted = F.softmax(all_p_muted, dim=1)
        # changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        # percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        # mask_b = changes.lt(percentile).float().view(-1, 1)
        # mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        #all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

