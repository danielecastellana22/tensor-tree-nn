import torch as th
import dgl
import dgl.init
from preprocessing.utils import ConstValues
import models.prob.th_logprob as thlp
from experiments.config import create_object_from_config


class BottomUpHMM(thlp.ProbModule):
    def __init__(self, h_size, only_root_state, state_transition_config, x_emission_config, y_emission_config=None):
        super(BottomUpHMM, self).__init__()
        self.x_emission = create_object_from_config(x_emission_config, h_size=h_size)
        self.y_emission = create_object_from_config(y_emission_config, h_size=h_size) if y_emission_config is not None else None
        self.state_transition = create_object_from_config(state_transition_config, h_size=h_size)

        self.only_root_state = only_root_state

    def forward(self, *t_list):
        eta_list = []

        for t in t_list:

            ############################################################################################################
            # UPWARD
            ############################################################################################################

            # register upward functions
            t.set_n_initializer(dgl.init.zero_initializer)
            t.register_message_func(self.state_transition.up_message_func)
            t.register_reduce_func(self.state_transition.up_reduce_func)
            t.register_apply_node_func(self.state_transition.up_apply_node_func)

            # set evidence
            if self.x_emission is not None:
                evid = self.x_emission.set_evidence(t.ndata['x'])
            else:
                evid = th.zeros(t.number_of_nodes(), self.state_transition.h_size)

            if self.training and self.y_emission is not None:
                evid = thlp.mul(evid, self.y_emission.set_evidence(t.ndata['y']))

            t.ndata['evid'] = evid

            # TODO: leave -1 to indicate last element?
            # modify types to consider bottom
            t.ndata['t'][t.ndata['t'] == ConstValues.NO_ELEMENT] = self.state_transition.num_types-1

            # remove -1 in position
            t.ndata['pos'] = t.ndata['pos'] * (t.ndata['pos'] != ConstValues.NO_ELEMENT) #t.ndata['pos_mask']

            # start propagation
            dgl.prop_nodes_topo(t)
            ############################################################################################################

            ############################################################################################################
            # DOWNWARD
            ############################################################################################################

            leaf_ids = [i for i in range(t.number_of_nodes()) if t.in_degree(i) == 0]
            t.ndata['is_leaf'] = th.zeros_like(t.ndata['t'], dtype=th.bool)
            t.ndata['is_leaf'][leaf_ids] = 1

            t_rev = self.__reverse_dgl_batch__(t)

            # downward
            t_rev.set_n_initializer(dgl.init.zero_initializer)
            t_rev.register_message_func(self.state_transition.down_message_func)
            t_rev.register_reduce_func(self.state_transition.down_reduce_func)
            t_rev.register_apply_node_func(self.state_transition.down_apply_node_func)

            # propagate
            dgl.prop_nodes_topo(t_rev)

            # return the posterior
            eta = t_rev.ndata['eta']
            t.ndata['eta'] = eta
            ############################################################################################################

            # append posterior
            if self.only_root_state:
                root_ids = [i for i in range(t.number_of_nodes()) if t.out_degree(i) == 0]
                eta_list.append(eta[root_ids])
            else:
                eta_list.append(eta)

        # compute the returned value
        if self.training:
            return None  # return nothing
        else:
            if self.y_emission is None:
                return eta_list
            else:
                return self.y_emission(*eta_list)  # return p(y_i|X)

    def accumulate_posterior(self, t_list, out_data):
        if self.training:
            likelihood_list = []
            eta_list = []
            # accumulate posterior on x_emission
            for t in t_list:
                self.x_emission.accumulate_posterior(t.ndata['eta'], t.ndata['x'])
                likelihood_list.append(t.ndata['N_u'].sum(0))

                if not self.only_root_state:
                    self.y_emission.accumulate_posterior(t.ndata['eta'], t.ndata['y'])
                    eta_list.append(t.ndata['eta'])
                else:
                    root_ids = [i for i in range(t.number_of_nodes()) if t.out_degree(i) == 0]
                    eta_list.append(t.ndata['eta'][root_ids])

            if self.only_root_state:
                # TODO: compute joint probability of element in eta_list (i.e. outer product)
                joint_prob = None
                self.y_emission.accumulate_posterior(joint_prob, out_data)
                # TODO: compute also p(y | x_r1, x_r2, ...)
            else:
                return th.stack(likelihood_list).sum()

    @staticmethod
    def __reverse_dgl_batch__(b):
        return dgl.batch([x.reverse(True, True) for x in dgl.unbatch(b)])