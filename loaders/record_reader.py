import os
import struct

class RecordReader():
    SIZE = 40
    def __init__(self, fname, as_str=False):
        self.fname = fname

        self.as_str = as_str
        self.f = open(fname, 'rb')
        self.len = os.path.getsize(fname)//self.SIZE

    def get_bin(self, i):
        self.f.seek(i*self.SIZE)
        return self.f.read(self.SIZE)

    def decode(self,bin):
        '''
        Given a binary record produced by encode() split into its component parts
        '''
        uuid_src = bin[:16]
        uuid_dst = bin[16:32]
        x_src = bin[32] >> 4
        x_dst = bin[32] & 0b1111
        rel = bin[33]
        ts = struct.unpack('>Q', bytes(2) + bin[34:])[0] # Interperate as unsigned long long

        return uuid_src,uuid_dst,x_src,x_dst,rel,ts

    def __next__(self):
        '''
        Scan the file backward and decode each 40B line
        '''
        if self.loc < self.len:
            ret = self.decode(self.f.read(self.SIZE))
            self.loc += self.SIZE
            self.f.seek(self.loc)
            return ret
        else:
            raise StopIteration

    def __iter__(self):
        '''
        Reset the counting variables
        '''
        self.loc = 0
        self.f.seek(self.loc)
        return self

    def __getitem__(self, idx):
        '''
        Returns record by index (without scanning through each line)
        But recall, files have records in reverse order so, adjusts for that too

        For now, slices will return bytes b.c. that's all I need them to return
        '''
        if isinstance(idx, slice):
            start = idx.start if idx.start else 0
            end = idx.stop if idx.stop else self.len - 1

            if start < 0:
                start = self.len+start-1
            if end < 0:
                end = self.len+end-1

            if end<=start:
                return bytes(0)

            self.f.seek(start * self.SIZE)
            bytes_read = (end-start) * self.SIZE
            return self.f.read(bytes_read)

        if idx < 0:
            # Support for backward indexing
            idx = self.len + idx - 1

        self.f.seek(idx*self.SIZE)
        if not self.as_str:
            return self.decode(self.f.read(self.SIZE))

        # Added because shelve likes string keys, but wanted
        # code to be backward compatible with just byte return
        # types also
        s,d,xs,xd,r,ts = self.decode(self.f.read(self.SIZE))
        return str(s),str(d),xs,xd,r,ts

    def __len__(self):
        return self.len


class E3RecordReader(RecordReader):
    '''
    Needed an extra byte for the node features
    '''
    SIZE = 41
    def decode(self, bin):
        '''
        Given a binary record produced by encode() split into its component parts
        '''
        uuid_src = bin[:16]
        uuid_dst = bin[16:32]
        x_src = bin[32]
        x_dst = bin[33]
        rel = bin[34]
        ts = struct.unpack('>Q', bytes(2) + bin[35:])[0] # Interperate as unsigned long long

        return uuid_src,uuid_dst,x_src,x_dst,rel,ts