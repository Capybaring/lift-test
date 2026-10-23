# Verbatim function extracted from official models/LIFT.py. See PROVENANCE.md.
def instance_norm(ts, dim):
    mu = ts.mean(dim, keepdims=True)
    ts = ts - mu
    std = ((ts ** 2).mean(dim, keepdims=True) + 1e-8) ** 0.5
    return ts / std, mu, std
