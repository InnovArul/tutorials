import torch

def sinkhorn(Q, nmb_iters, gpus=0):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        print(Q.sum(dim=0))
        print(Q.sum(dim=1))
        
        K, B = Q.shape

        if gpus > 0:
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
        else:
            u = torch.zeros(K)
            r = torch.ones(K) / K
            c = torch.ones(B) / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)

            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

if __name__ == "__main__":
    x = torch.randn(6,6).exp()
    y = sinkhorn(x, nmb_iters=3)
    print(y)
    print(y.sum(dim=0))
    print(y.sum(dim=1))