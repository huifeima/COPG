"""Atomic writes, stable identities and deterministic tensor runtime setup."""
from __future__ import annotations
import contextlib, fcntl, hashlib, importlib.metadata, json, os, platform, shutil, sys, tempfile
from pathlib import Path

def require(condition,message):
    if not condition: raise RuntimeError(message)

def plain(obj):
    import dataclasses, numpy as np
    if dataclasses.is_dataclass(obj): return plain(dataclasses.asdict(obj))
    if isinstance(obj,dict): return {str(k):plain(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [plain(x) for x in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    return obj
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_torch(path,payload):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_copy(src,dst):
    src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+dst.name+'.',dir=dst.parent)
    try:
        with src.open('rb') as inp,os.fdopen(fd,'wb') as out:
            shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
        os.replace(tmp,dst)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def emit(event,**d):
    print(event,json.dumps(d,sort_keys=True,ensure_ascii=False,allow_nan=False),flush=True)


@contextlib.contextmanager
def lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Already running: {p}') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def versions():
    out={'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform()}
    for n in ('numpy','torch','gymnasium','stable-baselines3','cloudpickle'):
        try:out[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:out[n]='MISSING'
    return out


def configure_torch(seed,device):
    import random
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if hasattr(torch.backends,'mha'):torch.backends.mha.set_fastpath_enabled(False)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import random,numpy as np,torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(s):
    import random,numpy as np,torch
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


@contextlib.contextmanager
def preserve_rng():
    state=rng_state()
    try:yield
    finally:restore_rng(state)


def tensor_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

