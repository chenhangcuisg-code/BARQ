"""Small CPU checks; does not download models or datasets."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from vq_quant import sinkhorn_em_hessian_weighted, get_assignments

torch.manual_seed(7)
x=torch.randn(2,32,3)
c=torch.randn(2,4,3)
out=sinkhorn_em_hessian_weighted(x,c,eps=0.1,sinkhorn_iters=100)
assert out.shape==c.shape and torch.isfinite(out).all()
assert torch.all(out>=x.min(1).values[:,None,:]-1e-5)
assert torch.all(out<=x.max(1).values[:,None,:]+1e-5)
indices=get_assignments(x,out)
expected=((x[:,:,None,:]-out[:,None,:,:])**2).sum(-1).argmin(-1)
assert torch.equal(indices,expected)

same=torch.full((2,16,3),2.5)
recovered=sinkhorn_em_hessian_weighted(same,c,eps=0.1,sinkhorn_iters=100)
assert torch.allclose(recovered,torch.full_like(c,2.5),atol=1e-5)
print('PASS: dimensions, finite centroids, convex bounds, hard encoding, identical-vector recovery')
