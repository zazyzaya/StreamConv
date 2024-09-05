import math
import platform
import shelve
import socket

import torch
from tqdm import tqdm

from loaders.graph_stream_objects import NodeEmb, RR_GCN
from loaders.record_reader import RecordReader
from loaders.e5_types import *

ENC_DIM = 128

# Please fill these in with the path to the files
DATA_HOME = None        # Raw E5 data
SAVE_TO = None          # Where to save embeddings
MOVED_TO = None         # Need to copy E5 files to speed up processing

if DATA_HOME is None:
    exit()



def add_id(uuid, idx, nmap):
        if uuid not in nmap:
            nmap[uuid] = idx[0]
            idx[0] += 1
            return True
        return False


def aggregate_time_decay(z_str, cur_t, prev_t, updated_ids, updated_vec, updated_t, last_z, last_n, decay):
    # Unchanged portions of the graph just decay more
    unchanged = ((last_n * last_z) - updated_vec)
    unchanged *= math.exp(-(cur_t-prev_t)*decay)
    last_n = math.exp(-(cur_t-prev_t)*decay)*(last_n - updated_t)

    # Need to calculate updated changes to the graph
    changed = torch.zeros(unchanged.size())
    with shelve.open(z_str, 'r') as db:
        cur_n = last_n
        for k in updated_ids:
            t = db[k].last_seen
            z = db[k].z

            delta = cur_t - t
            decay = math.exp(-delta * decay)
            cur_n += decay
            changed += z * decay

    # Finally, average together and return
    return (changed + unchanged).true_divide(cur_n), cur_n

def aggregate_first_time_decay(z_str, last_t, dim, decay):
    '''
    It's memory intensive to do this as a full mat mul, so let's
    instead do this as iterative addition so as not to overwhealm
    the RAM
    '''
    db = shelve.open(z_str, 'r')
    sketch = torch.zeros(1,dim*3, requires_grad=False)
    n = 0

    for k,v in db.items():
        t = v.last_seen
        z = v.z

        delta = last_t - t
        decay = math.exp(-delta * decay)
        sketch += decay * z

        n += decay

    db.close()
    return sketch.true_divide(n), n

def aggregate_updates(z_str, updated_ids, dim):
    '''
    Sketches only include data from nodes that were just updated
    '''
    db = shelve.open(z_str, 'r')
    sketch = torch.zeros(1, dim*3, requires_grad=False)

    for uuid in updated_ids:
        sketch += db[uuid].z

    return sketch / len(updated_ids)


# For the purposes of this project, most of these values never change
def modeswitch_label_stream(fname: str, n: int, rr_gcn: RR_GCN, fsize=FEATS, rsize=RELS, start_t=None, start_i=0, zs=dict(), verbose=True):
    recs = RecordReader(fname)
    e_size = fsize+rsize + rr_gcn.enc_dim * rr_gcn.layers

    def mode_switch(uuid, cur_mode):
        '''
        Check if a node has switched from being src to being dst
        '''
        if zs[uuid].mode == cur_mode:
            return False
        return True

    def build_edge(x, r, z):
        edge = torch.zeros(fsize+rsize)
        edge[[x, fsize+r]] = 1.
        return torch.cat([edge,z])

    # Use timestamp of first record as start time
    if start_t is None:
        start_t = recs[0][-1]

    # Every dst becomes a new node so convs preserve temporal dependance
    # Do-while. Just check condition at the end
    i = start_i
    freq = 0

    prog = tqdm(total=n, disable=(not verbose))
    while freq < n and i < len(recs):
        sid,did,sx,dx,rel,ts = recs[i]

        # Skip nodes of indeterminate type
        if sx == fsize or dx == fsize:
            i += 1
            continue

        freq += 1

        '''
         __________   ____
        / ____   _ \ / ___|
        \___ \| |_) | |
         ___) |  _ <| |___
        |____/|_| \_______|
        '''
        # If src is new, create singleton embedding
        if sid not in zs:
            zs[sid] = NodeEmb(sx, ts, NodeEmb.SRC, e_size, z=rr_gcn.singleton(sx))

        # Otherwise, check if src used to be a dst
        # If src has switched from dst, it needs to update its emb to prop
        # any new edges it's encountered
        elif mode_switch(sid, NodeEmb.SRC):
            zs[sid].mode = NodeEmb.SRC
            emb_data = zs[sid]

            # First time its being embedded
            if not emb_data.cnt:
                zs[sid].z = rr_gcn.forward(emb_data.x, emb_data.edges, emb_data.unproc)
            # A previous embedding needs to be updated
            else:
                zs[sid].z = rr_gcn.forward(emb_data.x, emb_data.edges, emb_data.unproc, emb_data.cnt, emb_data.z)

            # Clear out the neighbors, update count of edges injested
            zs[sid].purge()


        '''
         ____  ____ _____
        |  _ \/ _____   _|
        | | | \___ \ | |
        | |_| |___) || |
        |____/_____/ |_|
        '''
        # If dst is new, add src_z to list of neighbors
        if did not in zs:
            zs[did] = NodeEmb(dx, ts, NodeEmb.DST, e_size)
            edge = build_edge(zs[sid].x, rel, zs[sid].z)
            zs[did].add_edge(edge)

        # If a src is switching to a dst node, not much really changes
        # main difference is if we're adding src edge to ongoing group
        # or creating a new one (previous if block)
        else:
            if mode_switch(did, NodeEmb.DST):
                zs[did].mode = NodeEmb.DST

            zs[did].add_edge(build_edge(zs[sid].x, rel, zs[sid].z))

        i += 1
        prog.update()

        # No matter what, update last seen time
        zs[did].last_seen = zs[sid].last_seen = i

    prog.close()

    if verbose:
        print("Aggregating remaining dst nodes")

    '''
        _    _____ _________
       / \  / ___/  ___/  _ \
      / _ \| |  _| |  _| |_) |
     / ___ \ |_| | |_| |  _ <__
    /_/   \_____/\____/_| \___/
    '''
    z_out = [[],[]]
    for uuid in tqdm(zs.keys(), disable=(not verbose)):
        emb = zs[uuid]
        if emb.mode == NodeEmb.DST and emb.unproc:
            # First time its being embedded
            if not emb.cnt:
                z = rr_gcn.forward(emb.x, emb.edges, emb.unproc)
            # A previous embedding needs to be updated
            else:
                z = rr_gcn.forward(emb.x, emb.edges, emb.unproc, emb.cnt, emb.z)

            # May as well set them as src so we don't repeat this too often
            # It takes some computation to go from dst->src which we're already
            # doing here; it's free to go from src->dst
            emb.z = z
            emb.mode = NodeEmb.SRC
            emb.purge()

        else:
            z = emb.z

        z_out[0].append(z)
        z_out[1].append(emb.last_seen)

    z_out = (torch.stack(z_out[0]), torch.tensor([z_out[1]]))
    return zs, z_out, i, freq


def modeswitch_label_stream_ooc(fname, n, rr_gcn, zs_fname, decay, start_t=None,
            start_i=0, verbose=True, fsize=FEATS, rsize=RELS, unk=True, ignore=[], RR=RecordReader):
    '''
    As above, but uses out-of-core storage to save memory. A little slower, but safer for
    devices with less memory
    '''
    recs = RR(fname, as_str=True)
    e_size = fsize+rsize + rr_gcn.enc_dim * rr_gcn.layers
    zs = shelve.DbfilenameShelf(zs_fname, 'w', writeback=True)

    updated_vec = torch.zeros(rr_gcn.layers * rr_gcn.enc_dim)
    updated_t = 0
    updated_ids = set()

    def mode_switch(uuid, cur_mode):
        '''
        Check if a node has switched from being src to being dst
        '''
        if zs[uuid].mode == cur_mode:
            return False
        return True

    def build_edge(x, r, z):
        edge = torch.zeros(fsize+rsize)
        edge[[x, fsize+r]] = 1.
        return torch.cat([edge,z])

    # Use timestamp of first record as start time
    if start_t is None:
        start_t = recs[start_i][-1]

    # Every dst becomes a new node so convs preserve temporal dependance
    # Do-while. Just check condition at the end
    cur_ts = start_t
    i = start_i
    freq = 0

    while (cur_ts-start_t) < n and i < len(recs):
        sid,did,sx,dx,rel,ts = recs[i]

        # Skip nodes of indeterminate type
        if unk and (sx == FEATS or dx == FEATS):
            i += 1
            continue

        # Keep track of what portion of old emb has changed
        # Note: only tracks ids that were already there and have subsequently
        # been updated, not new ones.
        if sid not in updated_ids and sid in zs:
            dec = math.exp(-(start_t-zs[sid].last_seen)*decay)
            updated_vec += zs[sid].z * dec
            updated_t += dec

        if did not in updated_ids and did in zs:
            dec = math.exp(-(start_t-zs[did].last_seen)*decay)
            updated_vec += zs[did].z * dec
            updated_t += dec

        updated_ids.add(sid); updated_ids.add(did)

        cur_ts = ts
        freq += 1

        '''
         __________   ____
        / ____   _ \ / ___|
        \___ \| |_) | |
         ___) |  _ <| |___
        |____/|_| \_______|
        '''
        # If src is new, create singleton embedding
        if sid not in zs:
            zs[sid] = NodeEmb(sx, ts, NodeEmb.SRC, e_size, z=rr_gcn.singleton(sx))

        # Otherwise, check if src used to be a dst
        # If src has switched from dst, it needs to update its emb to prop
        # any new edges it's encountered
        elif mode_switch(sid, NodeEmb.SRC):
            zs[sid].mode = NodeEmb.SRC
            emb_data = zs[sid]

            # First time its being embedded
            if not emb_data.cnt:
                zs[sid].z = rr_gcn.forward(emb_data.x, emb_data.edges, emb_data.unproc)
            # A previous embedding needs to be updated
            else:
                zs[sid].z = rr_gcn.forward(emb_data.x, emb_data.edges, emb_data.unproc, emb_data.cnt, emb_data.z)

            # Clear out the neighbors, update count of edges injested
            zs[sid].purge()


        '''
         ____  ____ _____
        |  _ \/ _____   _|
        | | | \___ \ | |
        | |_| |___) || |
        |____/_____/ |_|
        '''
        # If dst is new, add src_z to list of neighbors
        if did not in zs:
            zs[did] = NodeEmb(dx, ts, NodeEmb.DST, e_size)
            edge = build_edge(zs[sid].x, rel, zs[sid].z)
            zs[did].add_edge(edge)

        # If a src is switching to a dst node, not much really changes
        # main difference is if we're adding src edge to ongoing group
        # or creating a new one (previous if block)
        else:
            if mode_switch(did, NodeEmb.DST):
                zs[did].mode = NodeEmb.DST

            zs[did].add_edge(build_edge(zs[sid].x, rel, zs[sid].z))

        i += 1

        # No matter what, update last seen time
        zs[did].last_seen = zs[sid].last_seen = ts

    if verbose:
        print("Aggregating remaining dst nodes")

    '''
        _    _____ _________
       / \  / ___/  ___/  _ \
      / _ \| |  _| |  _| |_) |
     / ___ \ |_| | |_| |  _ <__
    /_/   \_____/\____/_| \___/
    '''
    # Only cached items have been updated
    for uuid in zs.cache.keys():
        emb = zs[uuid]
        if emb.mode == NodeEmb.DST and emb.unproc:
            # First time its being embedded
            if not emb.cnt:
                z = rr_gcn.forward(emb.x, emb.edges, emb.unproc)
            # A previous embedding needs to be updated
            else:
                z = rr_gcn.forward(emb.x, emb.edges, emb.unproc, emb.cnt, emb.z)

            # May as well set them as src so we don't repeat this too often
            # It takes some computation to go from dst->src which we're already
            # doing here; it's free to go from src->dst
            emb.z = z
            emb.mode = NodeEmb.SRC
            emb.purge()

    # Synchronize changes to disc
    zs.close()
    return i, freq, ts, updated_ids, updated_vec, updated_t


def modeswitch_label_stream_ooc_event_count(fname, n, rr_gcn, zs_fname, decay, start_i=0, elapsed=0, verbose=True, fsize=FEATS, rsize=RELS, RR=RecordReader, unk=True):
    '''
    As above, but uses out-of-core storage to save memory. A little slower, but safer for
    devices with less memory
    '''
    recs = RR(fname, as_str=True)
    e_size = fsize+rsize + rr_gcn.enc_dim * rr_gcn.layers
    zs = shelve.DbfilenameShelf(zs_fname, 'w', writeback=True)

    updated_vec = torch.zeros(rr_gcn.layers * rr_gcn.enc_dim)
    updated_t = 0
    updated_ids = set()

    def mode_switch(uuid, cur_mode):
        '''
        Check if a node has switched from being src to being dst
        '''
        if zs[uuid].mode == cur_mode:
            return False
        return True

    def build_edge(x, r, z):
        edge = torch.zeros(fsize+rsize)
        edge[[x, fsize+r]] = 1.
        return torch.cat([edge,z])

    # Every dst becomes a new node so convs preserve temporal dependance
    # Do-while. Just check condition at the end
    i = start_i
    freq = 0

    while freq < n and i < len(recs):
        sid,did,sx,dx,rel,ts = recs[i]

        # Skip nodes/rels of indeterminate type
        if unk and (sx == FEATS or dx == FEATS):
            i += 1
            continue

        # Keep track of what portion of old emb has changed
        # Note: only tracks ids that were already there and have subsequently
        # been updated, not new ones.
        if sid not in updated_ids and sid in zs:
            dec = math.exp(-((i+elapsed)-zs[sid].last_seen)*decay)
            updated_vec += zs[sid].z * dec
            updated_t += dec

        if did not in updated_ids and did in zs:
            dec = math.exp(-((i+elapsed)-zs[did].last_seen)*decay)
            updated_vec += zs[did].z * dec
            updated_t += dec

        updated_ids.add(sid); updated_ids.add(did)
        freq += 1

        '''
         __________   ____
        / ____   _ \ / ___|
        \___ \| |_) | |
         ___) |  _ <| |___
        |____/|_| \_______|
        '''
        # If src is new, create singleton embedding
        if sid not in zs:
            zs[sid] = NodeEmb(sx, ts, NodeEmb.SRC, e_size, z=rr_gcn.singleton(sx))

        # Otherwise, check if src used to be a dst
        # If src has switched from dst, it needs to update its emb to prop
        # any new edges it's encountered
        elif mode_switch(sid, NodeEmb.SRC):
            zs[sid].mode = NodeEmb.SRC
            emb_data = zs[sid]

            # First time its being embedded
            if not emb_data.cnt:
                zs[sid].z = rr_gcn.forward(emb_data.x, emb_data.edges, emb_data.unproc)
            # A previous embedding needs to be updated
            else:
                zs[sid].z = rr_gcn.forward(emb_data.x, emb_data.edges, emb_data.unproc, emb_data.cnt, emb_data.z)

            # Clear out the neighbors, update count of edges injested
            zs[sid].purge()


        '''
         ____  ____ _____
        |  _ \/ _____   _|
        | | | \___ \ | |
        | |_| |___) || |
        |____/_____/ |_|
        '''
        # If dst is new, add src_z to list of neighbors
        if did not in zs:
            zs[did] = NodeEmb(dx, ts, NodeEmb.DST, e_size)
            edge = build_edge(zs[sid].x, rel, zs[sid].z)
            zs[did].add_edge(edge)

        # If a src is switching to a dst node, not much really changes
        # main difference is if we're adding src edge to ongoing group
        # or creating a new one (previous if block)
        else:
            if mode_switch(did, NodeEmb.DST):
                zs[did].mode = NodeEmb.DST

            zs[did].add_edge(build_edge(zs[sid].x, rel, zs[sid].z))

        # No matter what, update last seen time
        zs[did].last_seen = zs[sid].last_seen = i+elapsed
        i += 1

    if verbose:
        print("Aggregating remaining dst nodes")

    '''
        _    _____ _________
       / \  / ___/  ___/  _ \
      / _ \| |  _| |  _| |_) |
     / ___ \ |_| | |_| |  _ <__
    /_/   \_____/\____/_| \___/
    '''
    # Only cached items have been updated
    for uuid in zs.cache.keys():
        emb = zs[uuid]
        if emb.mode == NodeEmb.DST and emb.unproc:
            # First time its being embedded
            if not emb.cnt:
                z = rr_gcn.forward(emb.x, emb.edges, emb.unproc)
            # A previous embedding needs to be updated
            else:
                z = rr_gcn.forward(emb.x, emb.edges, emb.unproc, emb.cnt, emb.z)

            # May as well set them as src so we don't repeat this too often
            # It takes some computation to go from dst->src which we're already
            # doing here; it's free to go from src->dst
            emb.z = z
            emb.mode = NodeEmb.SRC
            emb.purge()

    # Synchronize changes to disc
    zs.close()
    return i, recs[i-1][-1], updated_ids, updated_vec, updated_t