import torch as th
import dgl
import dgl.init
from exputils.datasets import ConstValues
import thlogprob as thlp
from exputils.configurations import create_object_from_config


class HRM(thlp.ProbModule):

    def __init__(self, h_size, only_root_state, state_transition_config, x_emission_config, y_emission_config=None,
                 x_embedding_config=None):

        super(HRM, self).__init__()

        self.x_embedding = create_object_from_config(x_embedding_config) if x_embedding_config is not None else None

        if self.x_embedding is not None:
            self.x_emission = create_object_from_config(x_emission_config, h_size=h_size, out_size=self.x_embedding.embedding_dim)
        else:
            self.x_emission = create_object_from_config(x_emission_config, h_size=h_size)

        self.y_emission = create_object_from_config(y_emission_config, h_size=h_size) if y_emission_config is not None else None
        self.state_transition = create_object_from_config(state_transition_config, h_size=h_size)

        self.only_root_state = only_root_state

    def forward(self, *t_list, out_data=None):

        ################################################################################################################
        # UPWARD
        ################################################################################################################
        beta_root_list = []
        loglike_list = []
        for t in t_list:
            # register upward functions
            t.set_n_initializer(dgl.init.zero_initializer)

            # set evidence
            if self.x_emission is not None:
                if self.x_embedding is not None:
                    x_mask = (t.ndata['x'] != ConstValues.NO_ELEMENT)
                    t.ndata['x_embs'] = self.x_embedding(t.ndata['x'] * x_mask) * x_mask.view(-1, 1)
                    t.ndata['x_mask'] = (t.ndata['x'] != ConstValues.NO_ELEMENT)
                    evid = self.x_emission.set_evidence(t.ndata['x_embs']) * x_mask.view(-1, 1)
                else:
                    evid = self.x_emission.set_evidence(t.ndata['x'])
            else:
                evid = th.zeros(t.number_of_nodes(), self.state_transition.h_size)

            if self.training and not self.only_root_state and self.y_emission is not None:
                evid = thlp.mul(evid, self.y_emission.set_evidence(out_data))

            t.ndata['evid'] = evid

            # modify types to consider bottom
            t.ndata['t'][t.ndata['t'] == ConstValues.NO_ELEMENT] = self.state_transition.num_types-1

            # remove -1 in position
            t.ndata['pos'] = t.ndata['pos'] * (t.ndata['pos'] != ConstValues.NO_ELEMENT) #t.ndata['pos_mask']

            # start propagation
            dgl.prop_nodes_topo(t,
                                message_func=self.state_transition.up_message_func,
                                reduce_func=self.state_transition.up_reduce_func,
                                apply_node_func=self.state_transition.up_apply_node_func)

            root_ids = [i for i in range(t.number_of_nodes()) if t.out_degree(i) == 0]
            beta_root_list.append(t.ndata['beta'][root_ids])

            loglike_list.append(t.ndata['N_u'].sum())
        ################################################################################################################
        eta_root_list = beta_root_list

        if self.only_root_state:
            if self.training:
                if self.y_emission is not None:
                    joint_prob = self.y_emission.set_evidence(out_data) # bs x h x ... x h
                    bs = joint_prob.size(0)
                    n_vars = len(beta_root_list)
                    for i, beta_i in enumerate(beta_root_list):
                        joint_prob = thlp.mul(joint_prob, beta_i.view(*([bs] + [1] * (i) + [-1] + [1] * (n_vars - i - 1))))

                    y_eta, y_Z = thlp.normalise(joint_prob, list(range(1, joint_prob.ndim)), get_Z=True)

                    # accumulate posterior
                    self.y_emission.accumulate_posterior(y_eta, out_data)

                    loglike_list.append(y_Z.sum())

                    eta_root_list = []
                    if len(beta_root_list) == 1:
                        # only one tree, no var elimination is needed
                        eta_root_list.append(y_eta)
                    else:
                        for i in range(len(beta_root_list)):
                            sum_over_vars = list(set(range(1, y_eta.ndim)) - {i+1})
                            eta_root_list.append(thlp.sum_over(y_eta, sum_over_vars))
            else:
                # we do not need downward
                if self.y_emission is not None:
                    return self.y_emission(*beta_root_list)
                else:
                    return beta_root_list
        ################################################################################################################
        # DOWNWARD
        ################################################################################################################
        all_eta_list = []
        for idx_t, t in enumerate(t_list):
            leaf_ids = [i for i in range(t.number_of_nodes()) if t.in_degrees(i) == 0]
            t.ndata['is_leaf'] = th.zeros_like(t.ndata['t'], dtype=th.bool)
            t.ndata['is_leaf'][leaf_ids] = 1

            # set base case for downward recursion
            root_ids = [i for i in range(t.number_of_nodes()) if t.out_degree(i) == 0]
            t.ndata['beta'][root_ids] = eta_root_list[idx_t]

            t_rev = self.__reverse_dgl_batch__(t)

            # downward
            t_rev.set_n_initializer(dgl.init.zero_initializer)

            # propagate
            dgl.prop_nodes_topo(t_rev,
                                message_func=self.state_transition.down_message_func,
                                reduce_func=self.state_transition.down_reduce_func,
                                apply_node_func=self.state_transition.down_apply_node_func)

            # return the posterior
            eta = t_rev.ndata['eta']
            t.ndata['eta'] = eta

            # append posterior
            all_eta_list.append(eta)

            if self.training:
                # accumulate posterior
                if self.x_embedding is not None:
                    x_mask = t.ndata['x_mask']
                    self.x_emission.accumulate_posterior(t.ndata['eta'][x_mask], t.ndata['x_embs'][x_mask])
                else:
                    self.x_emission.accumulate_posterior(t.ndata['eta'], t.ndata['x'])
                if not self.only_root_state and self.y_emission is not None:
                    self.y_emission.accumulate_posterior(t.ndata['eta'], out_data)
        ################################################################################################################

        # compute the returned value
        if self.training:
            return th.stack(loglike_list).sum()
        else:
            # here only_root_state is false
            if self.y_emission is None:
                return all_eta_list
            else:
                return self.y_emission(*all_eta_list)  # return p(y_i|X)

    def __m_step__(self):
        pass


    @staticmethod
    def __reverse_dgl_batch__(t):
        t_rev = dgl.reverse(t, copy_edata=True, copy_ndata=True)
        t_rev.set_batch_num_nodes(t.batch_num_nodes())
        t_rev.set_batch_num_edges(t.batch_num_edges())
        return t_rev
