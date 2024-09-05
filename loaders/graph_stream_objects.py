import torch
from torch import nn

class RR_GCN(nn.Module):
    '''
    Implimenting the untrained random relational GCN from

        https://arxiv.org/abs/2203.02424

    to act as a low-ovehead encoder and combining that idea with a
    jumping knowledge net (https://people.csail.mit.edu/keyulux/pdf/JK-Net.pdf)
    so message passing is accomplished in a single aggr
    '''
    def __init__(self, layers, in_dim, enc_dim, decay=1.):
        super().__init__()

        # Don't really need to put this in a nn.ModuleList because
        # this net isn't trained. It's just used as a naive hash fn
        self.nets = [nn.Linear(in_dim, enc_dim)] + [nn.Linear(enc_dim, enc_dim)] * (layers-1)
        self.nonlin = nn.Tanh()

        self.in_dim = in_dim
        self.enc_dim = enc_dim
        self.layers = layers
        self.decay = decay

        # Non-linear activations need to be invertable so we can update
        # vectors over time.
        #   z_t = sigma(sum(N)), if new edges observed x_{t+1} then
        #   z_{t+1} = sigma(sum(N) + x_{t+1})  =  sigma(inv_sigma(z_t) + x_{t+1})
        self.inv = torch.atanh

    def load_saved(self, params):
        for i in range(len(self.nets)):
            self.nets[i].load_state_dict(params[i])

    @torch.no_grad()
    def forward(self, x, neighbors, num_neigh, cnt=0, old_z=None):
        '''
        Given node's own features x, and embedding of neighbors formatted as
        [ n_x , 1-hop z, 2-hop z, ..., k-1-hop z ]
        Convolve them together

        If an old vector to update, need to know how many samples it was made up of
        cnt, and the vector itself, old_z. Assumes old_z is 1 dimensional like the return
        vector from this function.
        '''
        assert (old_z is None and cnt==0) or (isinstance(old_z, torch.Tensor) and cnt > 0)

        vec = torch.zeros(self.in_dim)
        vec[x] = 1.
        div_by = num_neigh+cnt

        layers = [vec]

        # If an old_z exists, get it ready to be added in the convolutions
        # otherwise, just use zeros so we can have all the same ops
        if cnt > 0:
            old_z = self.inv(old_z)
            old_z *= cnt

            # Older edges affect the final embedding slightly less than
            # newer ones (set to 0.9885 for about 1/2 decay per hour)
            old_z *= self.decay
            neighbors *= (1+ (1-self.decay))
        else:
            old_z = torch.zeros(self.enc_dim * self.layers)

        # First layer is indexed differently so doing it separate from the loop
        l = self.nets[0](
            layers[0] +
            neighbors[:self.in_dim]
        ) + old_z[:self.enc_dim]
        l /= div_by

        layers.append(self.nonlin(l))

        # Get everything alligned along enc_dim chunks
        neighbors = neighbors[self.in_dim:]
        old_z = old_z[self.enc_dim:]

        for i in range(self.layers-1):
            idx = self.enc_dim*i

            # W/n (x1+x2+x3+...) = (Wx1 + Wx2 + ...)/n
            # May as well factor out the mat mul and amt to divide by for the average
            l = self.nets[i+1](
                layers[-1] +
                neighbors[idx:idx+self.enc_dim]
            ) + old_z[idx:idx+self.enc_dim]

            l /= div_by
            layers.append(self.nonlin(l))

        return torch.cat(layers[1:], dim=0)

    @torch.no_grad()
    def singleton(self, x):
        '''
        Assumes x is just node feat int, not concatted to a relation.
        Will append a zero vector to it for this reason
        '''
        vec = torch.zeros(self.in_dim)
        vec[x] = 1.

        layers = [vec]
        for i in range(self.layers):
            layers.append(
                self.nonlin(
                    self.nets[i](layers[-1])
                )
            )

        return torch.cat(layers[1:], dim=0)


class NodeEmb():
    SRC=0
    DST=1

    def __init__(self, x, first_seen, mode, e_size, z=None):
        self.z = z
        self.x = x
        self.first_seen = self.last_seen = first_seen
        self.mode = mode

        self.edges = None
        self.unproc = 0                     # Number of edges added to self.edges but unprocessed
        self.cnt = 0 if z is None else 1    # Number of edges convolved over the full lifetime of Node

    def add_edge(self, edge):
        # Avoid storing a bunch of empty tensors after
        # edges are aggregated
        if self.unproc:
            self.edges += edge
            self.unproc += 1
        else:
            self.edges = edge
            self.unproc = 1

    def purge(self):
        self.cnt += self.unproc
        self.unproc = 0
        self.edges = None