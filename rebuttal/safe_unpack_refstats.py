"""Unpack FD-Loss paper_ref_stats.pkl into .npz files, allowing only numpy reconstruction classes (no arbitrary code)."""
import pickle, sys, os, numpy as np
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
           ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
           ('numpy._core.multiarray', 'scalar'), ('collections', 'OrderedDict'), ('numpy._core.numeric', '_frombuffer'), ('numpy.core.numeric', '_frombuffer')}
class U(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in ALLOWED:
            return super().find_class(m.replace('numpy._core', 'numpy.core'), n)   # numpy-2 pickle on numpy 1.26
        raise pickle.UnpicklingError(f'blocked global {m}.{n}')
src, out = sys.argv[1], sys.argv[2]
b = U(open(src, 'rb')).load()
for name, arrays in b.items():
    np.savez(os.path.join(out, name), **arrays)
    print(name, {k: v.shape for k, v in arrays.items()})
