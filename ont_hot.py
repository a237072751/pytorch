data, target, otherfea = next(iter(train_loader))
data = torch.autograd.Variable(data.cuda())
target = torch.autograd.Variable(target.cuda())
one_hot = torch.rand(target.size(0),nlabels).zero_()
one_hot.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
one_hot = torch.autograd.Variable(one_hot.cuda())
