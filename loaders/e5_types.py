# Entity Types
# Subjects:
PROCESS = 0
THREAD = 1
UNIT = 2 # Specific to TRACE. A group of threads

def parse_file(x):
    x = x['predicateObjectPath']

    if x is None:
        return 3

    x = x.upper()

    if x.startswith('\\REGISTRY'):
        return REG(x)
    elif x.endswith('.DLL'):
        return DLL_FILE(x)

    # Hopefully the first condition is narrow and frequent
    # enough for this `in` not to hang the program
    elif x.endswith('TMP') or 'TMP' in x:
        return TMP_FILE(x)

    # Only works on windows envs
    elif x.startswith('\\Device\\NamedPipe'):
        return PIPE(0)

    else:
        return 3 # Normal FILE

def parse_flow(x):
    x = x['remoteAddress']

    # Default to local IP
    if x is None:
        return 8

    # Local IP range
    if x.startswith('128.55.12') or x.startswith('fe80'):
        return 8

    # Reserved/special addresses
    if x.startswith('ff') or x.endswith('.255'):
        return 9

    return 10


def parse_unk(x):
    '''
    Often CREATE_OBJECT is referring to files
    '''
    if x is None:
        return 13
    return parse_file(x)


# Objects
FILE=parse_file # 3,4,5,6, or 7
TMP_FILE=lambda x : 4
DLL_FILE=lambda x : 5
PIPE=lambda x : 6
REG=lambda x : 7
FLOW=lambda x : 8     # If unknown, assume it's local
SOCKET=lambda x : 11  # Unless specified otherwise, all net events are flows
SRC_SNK=lambda x : 12

UNK=parse_unk

# Human-readable version
NODE_TYPES = {
    0: 'Process', 1: 'Thread', 2: 'Unit',
    3: 'File', 4: 'Tmp File', 5: 'DLL', 6: 'Pipe',
    7: 'Registry Key', 8: 'Local Flow',
    9: 'Special Flow', 10: 'Extern Flow', 11: 'Socket',
    12: 'Src Sink', 13: 'Unknown'
}

# Means we're calling directly from OBJECT_CREATE
# and don't need to parse too much (not that there's a lot
# of data available.) I think because we're trying to figure
# out more info about files, maybe we should leave it NULL for now,
# and have it updated when an edge is created (and we know the
# file name/path)
OBJECTS = {
    'FILE_OBJECT': None,
    'IPC_OBJECT': PIPE(0),
    'REGISTRY_KEY_OBJECT': REG(0),
    'PACKET_SOCKET_OBJECT': SOCKET(0),
    'NET_FLOW_OBJECT': None, # Handled in parser file
    'SRC_SINK_OBJECT': SRC_SNK(0)
}

EXTRA_OBJECTS = {
    'TMP_FILE': TMP_FILE,
    'DLL_FILE': DLL_FILE
}

# Direction
SUB_TO_OB = 0
OB_TO_SUB = 1
BIDIRECTIONAL = 2
OB_TO_OB = 3 # Special case with pattern subj -> obj1 -> obj2

# Defaults if uuid is unknown at time of reading
# also specifies edge direction
EVENT_TYPES = {
    # We can pretty much guess that the object will be only one thing
    'EVENT_OPEN': (FILE, SUB_TO_OB),
    'EVENT_CLOSE': (FILE, SUB_TO_OB),
    'EVENT_READ': (FILE, OB_TO_SUB),                # Could also be socket or reg key
    'EVENT_MODIFY_FILE_ATTRIBUTES': (FILE, SUB_TO_OB),
    'EVENT_WRITE': (FILE, SUB_TO_OB),
    'EVENT_LSEEK': (FILE, SUB_TO_OB),
    'EVENT_MMAP': (FILE, BIDIRECTIONAL),
    'EVENT_MPROTECT': (FILE, SUB_TO_OB),
    'EVENT_SERVICEINSTALL': (FILE, SUB_TO_OB),
    'EVENT_CHECK_FILE_ATTRIBUTES': (FILE, OB_TO_SUB),
    'EVENT_FCNTL': (FILE, BIDIRECTIONAL),
    'EVENT_RENAME': (FILE, OB_TO_OB),
    'EVENT_LOADLIBRARY': (DLL_FILE, OB_TO_SUB),
    'EVENT_LINK': (FILE, OB_TO_OB),                 # Similar to rename, but linking
    'EVENT_UNLINK': (FILE, SUB_TO_OB),
    'EVENT_UPDATE': (FILE, OB_TO_OB),
    'EVENT_EXECUTE': (FILE, SUB_TO_OB),             # I've seen files and pipes, but mostly files
    'EVENT_TRUNCATE': (FILE, SUB_TO_OB),
    'EVENT_FINIT_MODULE': (FILE, SUB_TO_OB),

    'EVENT_SHM': (PIPE, OB_TO_OB),
    'EVENT_SIGNAL': (PIPE, SUB_TO_OB),
    'EVENT_WAIT': (PIPE, OB_TO_SUB),
    'EVENT_TEE': (PIPE, OB_TO_OB),
    'EVENT_SPLICE': (PIPE, OB_TO_OB),
    'EVENT_VMSPLICE': (PIPE, SUB_TO_OB),

    'EVENT_BIND': (FLOW, SUB_TO_OB),
    'EVENT_RECVMSG': (FLOW, OB_TO_SUB),
    'EVENT_RECVFROM': (FLOW, OB_TO_OB),
    'EVENT_SENDTO': (SOCKET, SUB_TO_OB),
    'EVENT_SENDMSG': (FLOW, SUB_TO_OB),
    'EVENT_CONNECT': (FLOW, SUB_TO_OB),             # I think?
    'EVENT_ACCEPT': (FLOW, OB_TO_SUB),              # Maybe socket.. not sure

    'EVENT_READ_SOCKET_PARAMS': (SOCKET, OB_TO_SUB),
    'EVENT_WRITE_SOCKET_PARAMS': (SOCKET, SUB_TO_OB),

    # Parser expects 'object' return types to be callable
    # so need to make these lambdas bc they aren't objects
    'EVENT_STARTSERVICE': (lambda x : PROCESS, SUB_TO_OB),     # Services are procs, right?
    'EVENT_FORK': (lambda x : PROCESS, SUB_TO_OB),             # (The new proc is the object?)
    'EVENT_CLONE': (lambda x : PROCESS, SUB_TO_OB),            # Similar to fork
    'EVENT_MODIFY_PROCESS': (lambda x : PROCESS, SUB_TO_OB),
    'EVENT_CREATE_THREAD': (lambda x : THREAD, SUB_TO_OB),

    # Ambiguous Edges; object could be many things
    'EVENT_CREATE_OBJECT': (UNK, SUB_TO_OB),
    'EVENT_LOGIN': (UNK, SUB_TO_OB),
    'EVENT_LOGOUT': (UNK, SUB_TO_OB),
    'EVENT_SIGNAL': (UNK, SUB_TO_OB),
    'EVENT_CHANGE_PRINCIPAL': (UNK, SUB_TO_OB), # They used the wrong spelling of "principle"
    'EVENT_FLOWS_TO': (UNK, OB_TO_OB),
    'EVENT_OTHER': (UNK, BIDIRECTIONAL)
}

events = list(EVENT_TYPES.keys())
events.sort() # Guarantee ordering
EVENT_VALS = {e:i for i,e in enumerate(events)}

FEATS=max(NODE_TYPES.keys())
RELS =max(EVENT_VALS.values())+1