def curvature_loss(hessian, outside=None):
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()