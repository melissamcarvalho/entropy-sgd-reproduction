

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Args:
            val (float): Increment on the numerator
                         of the average value.
            n (int): Increment on the denominator
                     of the average value.
        """
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the number of correct values over a batch

    Args:
        output (th.Tensor): Tensor of dimension (batch, num_classes).
                            output[i, j] is the probability that the
                            i-th instance on the batch belongs to the
                            j-th class.
        target (th.Tensor): Tensor with the expected class for each
                            instance on the batch.
        topk (tuple): tuple with k top values to be selected.
                      Defaults to 1.

    Returns:
        correct_k (th.Tensor): total top_k correct values. Sum of top_1
                               + top_2 + ... + top_k. By default, gets
                               only the correct values considering the
                               maximum probability (top1).
    """

    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)  # dimension (batch, maxk)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)

    return correct_k


def check_models(model1, model2):
    """
    Compares two pytorch models

    Args:
        model1 (nn.Module): base model.
        model2 (nn.Module): model to be compared.

    Returns:
        (bool): True, if models have the same parameters.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum().item() > 0:
            return False
    return True
