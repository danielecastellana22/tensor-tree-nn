import torch.nn as nn
import torch as th
import models.prob.th_logprob as thlp


# TODO: pos_stationarity cannot be implemented due to the saving child info into node.

class BaseStateTransition(thlp.CategoricalProbModule):

    # n_aggr allows to speed up the computation computing more aggregation in parallel. USEFUL FOR LSTM
    def __init__(self, h_size, pos_stationarity, max_output_degree, num_types):
        super(BaseStateTransition, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity

        if self.pos_stationarity:
            raise ValueError('Pos stationarity must be false!')

        if num_types is None:
            num_types = 1
        else:
            # +1 is for the bottom
            num_types += 1

        self.num_types = num_types

    def forward(self):
        pass

    def up_message_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def up_reduce_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def up_apply_node_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def down_message_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def down_reduce_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def down_apply_node_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    @staticmethod
    def __gather_param__(param: th.Tensor, types=None, pos=None):
        # we assume first dimension contains type information
        if types is not None:  # has shape (bs x t)
            p = param.gather(0, types.view(-1, *([1 for x in range(param.ndim - 1)])).expand(-1, *param.shape[1:]))

            if pos is not None:
                p = p.gather(1, pos.view(-1, 1, *([1 for x in range(p.ndim - 2)])).expand(-1, 1, *p.shape[2:])).squeeze(1)  # has shape (bs x h)
        else:
            if pos is not None:
                p = param.gather(0, pos.view(-1, *([1 for x in range(param.ndim - 1)])).expand(-1, *param.shape[1:]))
            else:
                p = param.unsqueeze(0)

        return p

    def accumulate_posterior(self, param: th.Tensor, posterior, types=None, pos=None):
        if self.training:
            if types is not None:
                if pos is not None:
                    idx_tensor = th.stack((types, pos))
                    param.grad += th.sparse.FloatTensor(idx_tensor, posterior.exp(), param.grad.shape).coalesce().to_dense()
                else:
                    param.grad.index_add_(0, types, posterior.exp())
            else:
                if pos is not None:
                    param.grad.index_add_(0, pos, posterior.exp())
                else:
                    param.grad += th.sum(posterior.exp(), 0)


class SwitchingParent(BaseStateTransition):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, num_types=None):
        super(SwitchingParent, self).__init__(h_size, pos_stationarity, max_output_degree, num_types)

        if self.pos_stationarity:
            raise ValueError('Switching Parent cannot be implemented with positional stationarity')
            # self.U = nn.Parameter(th.empty(self.num_types, self.h_size, self.h_size).squeeze(0), requires_grad=False)
            # self.p = nn.Parameter(th.empty(self.num_types, self.h_size).squeeze(0), requires_grad=False)
        else:
            self.Sp = nn.Parameter(th.empty(self.num_types, self.max_output_degree).squeeze(0), requires_grad=False)
            self.U = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size, self.h_size).squeeze(0),
                                  requires_grad=False)
            self.p = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size).squeeze(0),
                                  requires_grad=False)

        self.init_parameters()
        self.reset_posterior()

    def up_message_func(self, edges):
        return {'beta_ch': edges.src['beta']}

    def up_reduce_func(self, nodes):
        bs = nodes.mailbox['beta_ch'].shape[0]
        n_ch = nodes.mailbox['beta_ch'].shape[1]
        beta_ch = thlp.zeros(bs, self.max_output_degree, self.h_size)
        beta_ch[:, :n_ch, :] = nodes.mailbox['beta_ch']

        U = self.__gather_param__(self.U,
                                  types=nodes.data['t'] if self.num_types > 1 else None)
        # has shape bs x L x h x h
        Sp = self.__gather_param__(self.Sp,
                                  types=nodes.data['t'] if self.num_types > 1 else None)
        # has shape bs x L
        gamma_ch = thlp.mul(thlp.mul(beta_ch.unsqueeze(3), U), Sp.unsqueeze(2).unsqueeze(3))  # has shape (bs x L x h x h)
        gamma_p_ch = thlp.sum_over(gamma_ch, 2)  # has shape (bs x L x h)

        gamma_r = thlp.sum_over(gamma_p_ch, 1)  # has shape (bs x h)

        return {'gamma_r': gamma_r, 'gamma_ch': gamma_ch, 'gamma_p_ch': gamma_p_ch}

    def up_apply_node_func(self, nodes):
        x = nodes.data['evid']  # represents P(x_u | Q_u) have size bs x h

        if 'gamma_r' in nodes.data:
            beta = nodes.data['gamma_r']
        else:
            beta = self.__gather_param__(self.p,
                                         types=nodes.data['t'] if self.num_types > 1 else None,
                                         pos=nodes.data['pos'])

        beta = thlp.mul(x, beta)  # has shape (bs x h)
        beta, N_u = thlp.normalise(beta, 1, get_Z=True)

        return {'beta': beta, 'N_u': N_u}

    def down_message_func(self, edges):
        pos = edges.dst['pos']
        return {'eta_ch': edges.src['eta_ch'].gather(1, pos.view(-1, 1, 1).expand(-1, 1, self.h_size)).squeeze(1)}

    def down_reduce_func(self, nodes):
        eta_ch = nodes.mailbox['eta_ch'].squeeze(1)  # has shape (bs x h)

        return {'eta': eta_ch}

    def down_apply_node_func(self, nodes):
        if 'eta' in nodes.data:
            eta_u = nodes.data['eta']
        else:
            # root
            eta_u = nodes.data['beta']

        gamma_ch = nodes.data['gamma_ch']  # has shape (bs x L x h x h)
        gamma_p_ch = nodes.data['gamma_p_ch'].unsqueeze(2) # has shape (bs x L x 1 x h)
        gamma_r = nodes.data['gamma_r'].unsqueeze(1).unsqueeze(2)  # has shape (bs x 1 x 1 x h)
        n_ch_mask = gamma_p_ch.exp().sum((2, 3), keepdim=True)
        #a = thlp.div(gamma_ch, gamma_r * n_ch_mask)
        # a = thlp.mul(a, gamma_r)
        #b = thlp.sum_over(a, 2, keepdim=True)
        # P(Q_l, Q_ch_l, Sp=l| X)
        eta_u_chl = thlp.div(thlp.mul(gamma_ch, eta_u.unsqueeze(1).unsqueeze(2)), gamma_r * n_ch_mask)  # has shape (bs x L x h x h)
        eta_u_chl, eta_sp = thlp.normalise(eta_u_chl, [2, 3], get_Z=True)

        is_leaf = nodes.data['is_leaf']
        is_internal = th.logical_not(is_leaf)
        self.accumulate_posterior(self.U, eta_u_chl[is_internal],
                                  types=nodes.data['t'][is_internal] if self.num_types > 1 else None)

        self.accumulate_posterior(self.Sp, eta_sp[is_internal].squeeze(3).squeeze(2),
                                  types=nodes.data['t'][is_internal] if self.num_types > 1 else None)

        self.accumulate_posterior(self.p, eta_u[is_leaf],
                                  types=nodes.data['t'][is_leaf] if self.num_types > 1 else None,
                                  pos=nodes.data['pos'][is_leaf] if not self.pos_stationarity else None)

        return {'eta_ch': thlp.sum_over(eta_u_chl, 3), 'eta': eta_u}


class SumChild(BaseStateTransition):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, num_types=None):
        super(SumChild, self).__init__(h_size, pos_stationarity, max_output_degree, num_types)

        if self.pos_stationarity:
            self.U = nn.Parameter(th.empty(self.num_types, self.h_size, self.h_size).squeeze(0), requires_grad=False)
            self.p = nn.Parameter(th.empty(self.num_types, self.h_size).squeeze(0), requires_grad=False)
        else:
            self.U = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size, self.h_size).squeeze(0),
                                  requires_grad=False)
            self.p = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size).squeeze(0),
                                  requires_grad=False)

        self.init_parameters()
        self.reset_posterior()

    def up_message_func(self, edges):
        return {'beta_ch': edges.src['beta']}

    def up_reduce_func(self, nodes):
        bs = nodes.mailbox['beta_ch'].shape[0]
        n_ch = nodes.mailbox['beta_ch'].shape[1]
        beta_ch = thlp.zeros(bs, self.max_output_degree, self.h_size)
        beta_ch[:, :n_ch, :] = nodes.mailbox['beta_ch']

        U = self.__gather_param__(self.U,
                                  types=nodes.data['t'] if self.num_types > 1 else None)
        # has shape bs x L x h x h)
        gamma_ch = thlp.mul(beta_ch.unsqueeze(3), U)  # has shape (bs x L x h x h)
        gamma_p_ch = thlp.sum_over(gamma_ch, 2)  # has shape (bs x L x h)

        gamma_r = th.sum(gamma_p_ch[:, :n_ch, :], 1) # has shape (bs x h)

        return {'gamma_r': gamma_r, 'gamma_ch': gamma_ch, 'gamma_p_ch': gamma_p_ch}

    def up_apply_node_func(self, nodes):
        x = nodes.data['evid']  # represents P(x_u | Q_u) have size bs x h

        if 'gamma_r' in nodes.data:
            beta = nodes.data['gamma_r']
        else:
            beta = self.__gather_param__(self.p,
                                         types=nodes.data['t'] if self.num_types > 1 else None,
                                         pos=nodes.data['pos'])

        beta = thlp.mul(x, beta)  # has shape (bs x h)
        beta, N_u = thlp.normalise(beta, 1, get_Z=True)

        return {'beta': beta, 'N_u': N_u}

    def down_message_func(self, edges):
        pos = edges.dst['pos']
        return {'eta_ch': edges.src['eta_ch'].gather(1, pos.view(-1, 1, 1).expand(-1, 1, self.h_size)).squeeze(1)}

    def down_reduce_func(self, nodes):
        eta_ch = nodes.mailbox['eta_ch'].squeeze(1)  # has shape (bs x h)

        return {'eta': eta_ch}

    def down_apply_node_func(self, nodes):
        if 'eta' in nodes.data:
            eta_u = nodes.data['eta']
        else:
            # root
            eta_u = nodes.data['beta']

        gamma_ch = nodes.data['gamma_ch']  # has shape (bs x L x h x h)
        gamma_p_ch = nodes.data['gamma_p_ch'].unsqueeze(2) # has shape (bs x L x 1 x h)
        gamma_r = nodes.data['gamma_r'].unsqueeze(1).unsqueeze(2)  # has shape (bs x 1 x 1 x h)
        n_ch_mask = gamma_p_ch.exp().sum((2, 3), keepdim=True)
        a = thlp.div(gamma_ch, gamma_p_ch * n_ch_mask)
        a = thlp.mul(a, gamma_r)
        b = thlp.sum_over(a, 2, keepdim=True)
        # P(Q_l, Q_ch_l | X)
        eta_u_chl = thlp.div(thlp.mul(a, eta_u.unsqueeze(1).unsqueeze(2)), b * n_ch_mask)  # has shape (bs x L x h x h)

        is_leaf = nodes.data['is_leaf']
        is_internal = th.logical_not(is_leaf)
        self.accumulate_posterior(self.U, eta_u_chl[is_internal],
                                  types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.p, eta_u[is_leaf],
                                  types=nodes.data['t'][is_leaf] if self.num_types > 1 else None,
                                  pos=nodes.data['pos'][is_leaf] if not self.pos_stationarity else None)

        return {'eta_ch': thlp.sum_over(eta_u_chl, 3), 'eta': eta_u}


class Canonical(BaseStateTransition):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, num_types=None, rank=None):
        super(Canonical, self).__init__(h_size, pos_stationarity, max_output_degree, num_types)

        self.rank = rank

        if self.pos_stationarity:
            self.U = nn.Parameter(th.empty(self.num_types, self.h_size, self.rank).squeeze(0), requires_grad=False)
            self.p = nn.Parameter(th.empty(self.num_types, self.h_size).squeeze(0), requires_grad=False)
        else:
            self.U = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size, self.rank).squeeze(0),
                                  requires_grad=False)
            self.p = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size).squeeze(0),
                                  requires_grad=False)

        self.U_output = nn.Parameter(th.empty(self.num_types, self.rank, self.h_size).squeeze(0), requires_grad=False)

        self.init_parameters()
        self.reset_posterior()

    def up_message_func(self, edges):
        return {'beta_ch': edges.src['beta']}

    def up_reduce_func(self, nodes):
        bs = nodes.mailbox['beta_ch'].shape[0]
        n_ch = nodes.mailbox['beta_ch'].shape[1]
        beta_ch = thlp.zeros(bs, self.max_output_degree, self.h_size)
        beta_ch[:, :n_ch, :] = nodes.mailbox['beta_ch']

        U = self.__gather_param__(self.U,
                                  types=nodes.data['t'] if self.num_types > 1 else None)
        # has shape bs x L x h x rank
        gamma_ch = thlp.mul(beta_ch.unsqueeze(3), U)  # has shape (bs x L x h x rank)
        gamma_p_ch = thlp.sum_over(gamma_ch, 2)  # has shape (bs x L x rank)

        gamma_r = th.sum(gamma_p_ch[:, :n_ch, :], 1) # has shape (bs x rank)

        return {'gamma_r': gamma_r, 'gamma_ch': gamma_ch, 'gamma_p_ch': gamma_p_ch}

    def up_apply_node_func(self, nodes):
        x = nodes.data['evid']  # represents P(x_u | Q_u) have size bs x h

        if 'gamma_r' in nodes.data:
            U_out = self.__gather_param__(self.U_output, types=nodes.data['t'] if self.num_types > 1 else None)
            # U_out has shape bs x rank x h
            beta = thlp.sum_over(thlp.mul(nodes.data['gamma_r'].unsqueeze(2), U_out), 1)
        else:
            beta = self.__gather_param__(self.p,
                                         types=nodes.data['t'] if self.num_types > 1 else None,
                                         pos=nodes.data['pos'] if not self.pos_stationarity else None)

        beta = thlp.mul(x, beta)  # has shape (bs x h)
        beta, N_u = thlp.normalise(beta, 1, get_Z=True)

        return {'beta': beta, 'N_u': N_u}

    def down_message_func(self, edges):
        pos = edges.dst['pos']
        return {'eta_ch': edges.src['eta_ch'].gather(1, pos.view(-1, 1, 1).expand(-1, 1, self.h_size)).squeeze(1)}

    def down_reduce_func(self, nodes):
        eta_ch = nodes.mailbox['eta_ch'].squeeze(1)  # has shape (bs x h)

        return {'eta': eta_ch}

    def down_apply_node_func(self, nodes):
        if 'eta' in nodes.data:
            eta_u = nodes.data['eta']
        else:
            # root
            eta_u = nodes.data['beta']

        U_out = self.__gather_param__(self.U_output, types=nodes.data['t'] if self.num_types>1 else None)

        gamma_r = nodes.data['gamma_r']  # has shape (bs x rank)
        a = thlp.mul(U_out, gamma_r.unsqueeze(2))  # has shape bs x rank x h
        b = thlp.sum_over(a, 1, keepdim=True)
        # P(Q_u, R_u | X)
        eta_ur = thlp.mul(thlp.div(a, b), eta_u.unsqueeze(1))  # has shape bs x rank x h

        eta_r = thlp.sum_over(eta_ur, 2).unsqueeze(1).unsqueeze(2)  # has shape (bs x 1 x 1 x rank)
        gamma_ch = nodes.data['gamma_ch']  # has shape (bs x L x h x rank)
        gamma_p_ch = nodes.data['gamma_p_ch'].unsqueeze(2) # has shape (bs x L x 1 x rank)
        gamma_r = gamma_r.unsqueeze(1).unsqueeze(2)  # has shape (bs x 1 x 1 x rank)
        n_ch_mask = gamma_p_ch.exp().sum((2, 3), keepdim=True)
        a = thlp.div(gamma_ch, gamma_p_ch * n_ch_mask)
        a = thlp.mul(a, gamma_r)
        b = thlp.sum_over(a, 2, keepdim=True)
        # P(Q_l, R_l | X)
        eta_ur_ch = thlp.div(thlp.mul(a, eta_r), b * n_ch_mask)  # has shape (bs x L x h x rank)

        is_leaf = nodes.data['is_leaf']
        is_internal = th.logical_not(is_leaf)
        self.accumulate_posterior(self.U, eta_ur_ch[is_internal],
                                  types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.U_output, eta_ur[is_internal],
                                  types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.p, eta_u[is_leaf],
                                  types=nodes.data['t'][is_leaf] if self.num_types > 1 else None,
                                  pos=nodes.data['pos'][is_leaf] if not self.pos_stationarity else None)

        return {'eta_ch': thlp.sum_over(eta_ur_ch, 3), 'eta': eta_u}


class Full(BaseStateTransition):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, num_types=None):
        super(Full, self).__init__(h_size, pos_stationarity, max_output_degree, num_types)

        if self.pos_stationarity:
            raise ValueError('Full cannot be pos stationary.')

        dim = [self.h_size+1 for i in range(max_output_degree)]
        dim.insert(0, self.num_types)
        dim.append(self.h_size)
        self.U = nn.Parameter(th.empty(*dim).squeeze(0), requires_grad=False)
        self.p = nn.Parameter(th.empty(self.num_types, h_size).squeeze(0), requires_grad=False)

        self.init_parameters()
        self.reset_posterior()

    def up_message_func(self, edges):
        return {'beta_ch': edges.src['beta']} # , 'pos': edges.data['pos']}

    def up_reduce_func(self, nodes):
        beta_ch = nodes.mailbox['beta_ch']  # has shape (bs x n_ch x h)
        n_ch = beta_ch.shape[1]
        bs = beta_ch.shape[0]

        U = self.__gather_param__(self.U, types=nodes.data['t'] if self.num_types > 1 else None)

        for i in range(self.max_output_degree):
            # TODO: we are assuming last are bottom
            if i < n_ch:
                btm = thlp.zeros(bs, 1)
                x = th.cat((beta_ch[:, i, :], btm), 1)
            else:
                x = thlp.zeros(1, self.h_size+1)
                x[:, -1] = 0

            new_shape = [x.shape[0]] + [1]*i + [self.h_size+1] + [1]*(self.max_output_degree-i)
            U = thlp.mul(U, x.view(*new_shape))

        beta = thlp.sum_over(U, list(range(1, self.max_output_degree+1)))
        return {'beta_np': beta, 'beta_ch': U}

    def up_apply_node_func(self, nodes):
        x = nodes.data['evid']  # represents P(x_u | Q_u) have size bs x h

        if 'beta_np' in nodes.data:
            beta = nodes.data['beta_np']  # has shape (bs x h)
        else:
            beta = self.__gather_param__(self.p, types=nodes.data['t'] if self.num_types > 1 else None)

        beta = thlp.mul(x, beta)  # has shape (bs x h)

        # normalise
        beta, N_u = thlp.normalise(beta, 1, get_Z=True)

        return {'beta': beta, 'N_u': N_u}

    def down_message_func(self, edges):
        pos = edges.dst['pos']
        return {'eta_ch': edges.src['eta_ch'].gather(1, pos.view(-1, 1, 1).expand(-1, 1, self.h_size)).squeeze(1)}

    def down_reduce_func(self, nodes):
        eta_ch = nodes.mailbox['eta_ch'].squeeze(1)  # has shape (bs x h)

        return {'eta': eta_ch}

    def down_apply_node_func(self, nodes):
        if 'eta' in nodes.data:
            eta_u = nodes.data['eta']
        else:
            # root
            eta_u = nodes.data['beta']

        beta_ch = nodes.data['beta_ch']  # has shape (bs x h x ... x h x h)
        beta_np = nodes.data['beta_np']
        new_shape = [-1] + [1] * self.max_output_degree + [self.h_size]
        # P(Q_u, Q_1, ..., Q_L | X)
        eta_uch = thlp.div(thlp.mul(beta_ch, eta_u.view(*new_shape)), beta_np.view(*new_shape))
        # P(Q_1, ..., Q_L | X)
        eta_joint_ch = thlp.sum_over(eta_uch, -1)

        bs = eta_u.shape[0]
        eta_ch = th.empty(bs, self.max_output_degree, self.h_size+1)
        for i in range(self.max_output_degree):
            sum_over_var = list(set(range(1, self.max_output_degree + 1)) - {i+1})
            eta_ch[:, i, :] = thlp.sum_over(eta_joint_ch, sum_over_var)

        # accumulate posterior
        is_leaf = nodes.data['is_leaf']
        is_internal = th.logical_not(is_leaf)
        self.accumulate_posterior(self.U, eta_uch[is_internal], types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.p, eta_u[is_leaf],
                                  types=nodes.data['t'][is_leaf] if self.num_types > 1 else None)

        return {'eta_ch': eta_ch[:, :, :-1], 'eta': eta_u}


class HOSVD(BaseStateTransition):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, num_types=None, rank=None):
        super(HOSVD, self).__init__(h_size, pos_stationarity, max_output_degree, num_types)

        self.rank = rank

        if self.pos_stationarity:
            raise ValueError('HOSVD cannot be pos stationary.')

        # input mode matrices
        self.U = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size, self.rank).squeeze(0),
                              requires_grad=False)

        # priori on leaves
        self.p = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size).squeeze(0),
                              requires_grad=False)

        # core tensor
        dim = [self.rank + 1 for i in range(max_output_degree)]
        dim.insert(0, self.num_types)
        dim.append(self.rank)
        self.G = nn.Parameter(th.empty(*dim).squeeze(0), requires_grad=False)

        # output mode matrix
        self.U_output = nn.Parameter(th.empty(self.num_types, self.rank, self.h_size).squeeze(0), requires_grad=False)

        self.init_parameters()
        self.reset_posterior()

    def up_message_func(self, edges):
        return {'beta_ch': edges.src['beta']}

    def up_reduce_func(self, nodes):
        bs = nodes.mailbox['beta_ch'].shape[0]
        n_ch = nodes.mailbox['beta_ch'].shape[1]
        beta_ch = thlp.zeros(bs, self.max_output_degree, self.h_size)
        beta_ch[:, :n_ch, :] = nodes.mailbox['beta_ch']

        # compute beta_r
        U = self.__gather_param__(self.U,
                                  types=nodes.data['t'] if self.num_types > 1 else None,)
        # U has shape bs x L x h x rank
        gamma_ch_rl = thlp.sum_over(thlp.mul(U, beta_ch.unsqueeze(3)), 2)  # has shape bs x L x rank

        G = self.__gather_param__(self.G, types=nodes.data['t'] if self.num_types > 1 else None)

        for i in range(self.max_output_degree):
            # TODO: we are assuming last are bottom
            if i < n_ch:
                btm = thlp.zeros(bs, 1)
                x = th.cat((gamma_ch_rl[:, i, :], btm), 1)
            else:
                x = thlp.zeros(1, self.rank+1)
                x[:, -1] = 0

            new_shape = [x.shape[0]] + [1]*i + [self.rank+1] + [1]*(self.max_output_degree-i)
            G = thlp.mul(G, x.view(*new_shape))

        gamma_r = thlp.sum_over(G, list(range(1, self.max_output_degree+1)))
        return {'gamma_r': gamma_r, 'gamma_ch_all': G, 'beta_ch': beta_ch}

    def up_apply_node_func(self, nodes):
        x = nodes.data['evid']  # represents P(x_u | Q_u) have size bs x h

        if 'gamma_r' in nodes.data:
            U_out = self.__gather_param__(self.U_output, types=nodes.data['t'] if self.num_types > 1 else None)
            # U_out has shape bs x rank x h
            beta = thlp.sum_over(thlp.mul(nodes.data['gamma_r'].unsqueeze(2), U_out), 1)
        else:
            beta = self.__gather_param__(self.p,
                                         types=nodes.data['t'] if self.num_types > 1 else None,
                                         pos=nodes.data['pos'])

        beta = thlp.mul(x, beta)  # has shape (bs x h)
        # normalise
        beta, N_u = thlp.normalise(beta, 1, get_Z=True)

        return {'beta': beta, 'N_u': N_u}

    def down_message_func(self, edges):
        pos = edges.dst['pos']
        return {'eta_ch': edges.src['eta_ch'].gather(1, pos.view(-1, 1, 1).expand(-1, 1, self.h_size)).squeeze(1)}

    def down_reduce_func(self, nodes):
        eta_ch = nodes.mailbox['eta_ch'].squeeze(1)  # has shape (bs x h)

        return {'eta': eta_ch}

    def down_apply_node_func(self, nodes):
        if 'eta' in nodes.data:
            eta_u = nodes.data['eta']
        else:
            # root
            eta_u = nodes.data['beta']

        bs = eta_u.shape[0]

        gamma_r = nodes.data['gamma_r']  # has shape (bs x rank)
        U_out = self.__gather_param__(self.U_output, types=nodes.data['t'] if self.num_types > 1 else None)
        # U_out has shape bs x rank x h
        a = thlp.mul(gamma_r.unsqueeze(2), U_out)
        b = thlp.sum_over(a, 1, keepdim=True)
        # P(Q_u, R_u | X)
        eta_ur = thlp.div(thlp.mul(a, eta_u.unsqueeze(1)), b)  # has shape bs x rank x h

        eta_r = thlp.sum_over(eta_ur, 2)  # has shape bs x rank

        gamma_ch_all = nodes.data['gamma_ch_all']  # has shape bs x r x ... x r x r
        new_shape = [-1] + [1] * self.max_output_degree + [self.rank]
        # P(R_u, R_1, ..., R_L | X)
        eta_ru_rch = thlp.div(thlp.mul(gamma_ch_all, eta_r.view(*new_shape)), gamma_r.view(*new_shape))
        # P(R_1, ..., R_L | X)
        eta_rch = thlp.sum_over(eta_ru_rch, -1)  # has shape bs x r+1 x ... r+1

        eta_rl = thlp.zeros(bs, self.max_output_degree, self.rank+1)
        for i in range(self.max_output_degree):
            sum_over_var = list(set(range(1, self.max_output_degree + 1)) - {i+1})
            eta_rl[:, i, :] = thlp.sum_over(eta_rch, sum_over_var)

        U = self.__gather_param__(self.U,
                                  types=nodes.data['t'] if self.num_types > 1 else None)
        #  U has shape bs x L x h x rank
        a = thlp.mul(U, nodes.data['beta_ch'].unsqueeze(3))
        b = thlp.sum_over(a, 2, keepdim=True)
        # P(Q_l, R_l | X)
        eta_rql = thlp.div(thlp.mul(a, eta_rl[:, :, :-1].unsqueeze(2)), b)  # has shape bs x L x h x rank

        # accumulate posterior
        is_leaf = nodes.data['is_leaf']
        is_internal = th.logical_not(is_leaf)

        self.accumulate_posterior(self.U, eta_rql[is_internal], types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.U_output, eta_ur[is_internal], types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.G, eta_ru_rch[is_internal], types=nodes.data['t'][is_internal] if self.num_types > 1 else None)
        self.accumulate_posterior(self.p, eta_u[is_leaf],
                                  types=nodes.data['t'][is_leaf] if self.num_types > 1 else None,
                                  pos=nodes.data['pos'][is_leaf])

        return {'eta_ch': thlp.sum_over(eta_rql, 3), 'eta': eta_u}


class TensorTrain(BaseStateTransition):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, num_types=None, rank=None):
        super(TensorTrain, self).__init__(h_size, pos_stationarity, max_output_degree, num_types)

        self.rank = rank

        if self.pos_stationarity:
            # 3d tensor
            self.U = nn.Parameter(th.empty(self.num_types, self.h_size, self.rank+1, self.rank).squeeze(0), requires_grad=False)

            # priori on leaves
            self.p = nn.Parameter(th.empty(self.num_types, self.h_size).squeeze(0), requires_grad=False)
        else:
            # 3d tensor
            self.U = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size, self.rank+1, self.rank).squeeze(0),
                                  requires_grad=False)

            # priori on leaves
            self.p = nn.Parameter(th.empty(self.num_types, self.max_output_degree, self.h_size).squeeze(0),
                                  requires_grad=False)

        # output rank
        # self.R_output = nn.Parameter(th.empty(self.num_types, self.rank, self.rank).squeeze(0), requires_grad=False)

        # output mode matrix
        self.U_output = nn.Parameter(th.empty(self.num_types, self.rank, self.h_size).squeeze(0), requires_grad=False)

        self.init_parameters()
        self.reset_posterior()

    def up_message_func(self, edges):
        return {'beta_ch': edges.src['beta']}

    def up_reduce_func(self, nodes):
        bs = nodes.mailbox['beta_ch'].shape[0]
        n_ch = nodes.mailbox['beta_ch'].shape[1]
        beta_ch = thlp.zeros(bs, self.max_output_degree, self.h_size)
        beta_ch[:, :n_ch, :] = nodes.mailbox['beta_ch']
        # TODO: we are assuming beta_ch is ordered accoridng pos. It allows bottom at the end

        # compute beta_r
        U = self.__gather_param__(self.U,
                                  types=nodes.data['t'] if self.num_types > 1 else None)
        # U has shape bs x L x h x rank+1 x rank

        gamma_ch_rl = thlp.sum_over(thlp.mul(U, beta_ch.unsqueeze(3).unsqueeze(4)), 2)
        # has shape bs x L x rank+1 x rank

        gamma_less_l = thlp.zeros(bs, self.max_output_degree, self.rank+1)
        gamma_less_l[:, 0, :-1] = gamma_ch_rl[:, 0, -1, :]  # has shape bs x rank
        for i in range(1, n_ch):
            gamma_i = gamma_ch_rl[:, i, :, :]  # has shape bs x (rank+1) x rank
            gamma_prev = gamma_less_l[:, i-1, :].unsqueeze(2)  # has shape bs x (rank+1) x 1
            gamma_less_l[:, i, :-1] = thlp.sum_over(thlp.mul(gamma_i, gamma_prev), 1)
        gamma_less_l[:, n_ch:, -1] = 0

        # R_out = self.__gather_param__(self.R_output, types=nodes.data['t'] if self.num_types > 1 else None)
        # gamma_r = thlp.sum_over(thlp.mul(R_out, gamma_less_l[:, n_ch-1, :-1].unsqueeze(2)), 1)  # has shape bs x rank
        gamma_r = gamma_less_l[:, n_ch-1, :-1]

        return {'gamma_less_l': gamma_less_l, 'gamma_r': gamma_r, 'beta_ch': beta_ch,
                'n_ch': th.full((bs, 1), n_ch, dtype=th.long).squeeze(1)}

    def up_apply_node_func(self, nodes):
        x = nodes.data['evid']  # represents P(x_u | Q_u) have size bs x h

        if 'gamma_r' in nodes.data:
            U_out = self.__gather_param__(self.U_output, types=nodes.data['t'] if self.num_types > 1 else None)
            # U_out has shape bs x rank x h
            a = thlp.sum_over(thlp.mul(nodes.data['gamma_r'].unsqueeze(2), U_out), 1)
        else:
            a = self.__gather_param__(self.p,
                                         types=nodes.data['t'] if self.num_types > 1 else None,
                                         pos=nodes.data['pos'])

        beta = thlp.mul(x, a)  # has shape (bs x h)
        # normalise
        beta, N_u = thlp.normalise(beta, 1, get_Z=True)

        return {'beta': beta, 'N_u': N_u}

    def down_message_func(self, edges):
        pos = edges.dst['pos']
        return {'eta_ch': edges.src['eta_ch'].gather(1, pos.view(-1, 1, 1).expand(-1, 1, self.h_size)).squeeze(1)}

    def down_reduce_func(self, nodes):
        eta_ch = nodes.mailbox['eta_ch'].squeeze(1)  # has shape (bs x h)

        return {'eta': eta_ch}

    def down_apply_node_func(self, nodes):
        if 'eta' in nodes.data:
            eta_u = nodes.data['eta']
        else:
            # root
            eta_u = nodes.data['beta']

        is_leaf = nodes.data['is_leaf']
        is_internal = th.logical_not(is_leaf)
        n_ch_list = nodes.data['n_ch'][is_internal]-1
        gamma_r = nodes.data['gamma_r'][is_internal]  # has shape (bs x rank)
        gamma_L = nodes.data['gamma_less_l'][is_internal].gather(1, n_ch_list.view(-1, 1, 1).expand(-1, 1, self.rank + 1)).squeeze(1)
        # has shape bs x rank+1
        gamma_less_l = nodes.data['gamma_less_l'][is_internal]  # has shape bs x L x rank+1
        beta_ch = nodes.data['beta_ch'][is_internal]  # has shape bs x L x h
        t = nodes.data['t'][is_internal]

        self.accumulate_posterior(self.p, eta_u[is_leaf],
                                  types=nodes.data['t'][is_leaf] if self.num_types > 1 else None,
                                  pos=nodes.data['pos'][is_leaf])

        eta_u_ch_all = thlp.zeros(eta_u.shape[0], self.max_output_degree, self.h_size)
        if th.any(is_internal):
            # computation only on internal nodes
            # compute P(Q_u, R_u | X)
            U_out = self.__gather_param__(self.U_output, types=t if self.num_types > 1 else None)
            # U_out has shape bs x rank x h
            a = thlp.mul(gamma_r.unsqueeze(2), U_out)
            b = thlp.sum_over(a, 1, keepdim=True)
            eta_ur = thlp.div(thlp.mul(a, eta_u[is_internal].unsqueeze(1)), b)  # has shape bs x rank x h

            # compute P(R_u, R_L | X)
            eta_r = thlp.sum_over(eta_ur, 2)  # has shape bs x rank_U
            # R_out = self.__gather_param__(self.R_output, types=t if self.num_types > 1 else None)
            # # R_out has shape bs x rank_L x rank_U
            # a = thlp.mul(R_out, gamma_L[:, :-1].unsqueeze(2))  # has shape bs x rank_L x rank_U
            # b = thlp.sum_over(a, 1, keepdim=True)
            # eta_rul = thlp.div(thlp.mul(a, eta_r.unsqueeze(1)), b)  # has shape bs x rank_L x rank_U

            # compute P(R_l, R_l-1, Q_l | X)
            # eta_rL = thlp.sum_over(eta_rul, 2)  # has shape bs x rank
            eta_rL = eta_r
            U = self.__gather_param__(self.U, types=t if self.num_types > 1 else None)
            # U has shape bs x L x h x rank+1  x rank
            eta_u_ch = thlp.zeros(eta_rL.shape[0], self.max_output_degree, self.h_size)
            last_eta = eta_rL  # has shape bs x rank
            for i in range(self.max_output_degree-1, -1, -1):
                pos_flag = i <= n_ch_list
                if th.any(pos_flag):
                    if i > 0:
                        a = thlp.mul(U[pos_flag, i, :, :, :], gamma_less_l[pos_flag, i-1, :].unsqueeze(1).unsqueeze(3))
                        a = thlp.mul(a, beta_ch[pos_flag, i, :].unsqueeze(2).unsqueeze(3))
                    else:
                        a = thlp.zeros(*(U.shape[:1] + U.shape[2:]))
                        a[:, :, -1, :] = thlp.mul(U[:, i, :, -1, :], beta_ch[:, i, :].unsqueeze(2))
                    b = thlp.sum_over(a, (1, 2), keepdim=True)
                    eta_rul_rlprec = thlp.div(thlp.mul(a, last_eta[pos_flag, :].unsqueeze(1).unsqueeze(2)), b)
                    self.accumulate_posterior(self.U, eta_rul_rlprec,
                                              types=t[pos_flag] if self.num_types > 1 else None,
                                              pos=th.full((eta_rul_rlprec.shape[0], 1), i, dtype=th.long).squeeze(1))
                    eta_u_ch[pos_flag, i, :] = thlp.sum_over(eta_rul_rlprec, (2, 3))
                    last_eta[pos_flag, :] = thlp.sum_over(eta_rul_rlprec, (1, 3))[:, :-1]

            # accumulate posterior
            self.accumulate_posterior(self.U_output, eta_ur, types=t if self.num_types > 1 else None)
            # self.accumulate_posterior(self.R_output, eta_rul, types=t if self.num_types > 1 else None)
            eta_u_ch_all[is_internal] = eta_u_ch

        return {'eta_ch': eta_u_ch_all, 'eta': eta_u}