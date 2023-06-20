class Similarity_loss(nn.Module):
    def __init__(self):
        super(Similarity_loss, self).__init__()
    def forward(self,input1,input2):
        cos_sim = F.cosine_similarity(input1, input2, dim=0)
        loss = torch.mean(torch.clamp(cos_sim, min=0.0))
        return loss
