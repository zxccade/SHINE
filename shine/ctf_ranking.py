import torch
import torch.nn.functional as F

def coarse_ranking(targets, outputs, margin1, margin2, q=8):
    """
    comments needs to be added
    """
    saliency_scores = outputs["saliency_scores"].clone()  # (N, T)
    saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, T)
    video_len = targets["video_length"].long()

    device = saliency_scores.device

    saliency_label = targets["saliency_all_labels"]

    nonzero_ids = torch.nonzero(saliency_label).tolist()

    groups = {}
    for row, col in nonzero_ids:
        if row in groups:
            groups[row].append(col)
        else:
            groups[row] = [col]

    relevant_ids = list(groups.values())

    all_clip_ids = [list(range(0, length)) for length in video_len]
    irrelevant_ids = [list(set(i) - set(j)) for i, j in zip(all_clip_ids, relevant_ids)]

    loss = torch.zeros(1, dtype=torch.float32).to(device).float()
    bs = saliency_scores.shape[0]

    for i in range(bs):

        saliency_score = saliency_scores[i][:video_len[i]]
        saliency_score_neg = saliency_scores_neg[i][:video_len[i]]

        inner_gt_saliency_score = torch.gather(saliency_score, 0, torch.tensor(relevant_ids[i], dtype=torch.long).to(device))
        outer_gt_saliency_score = torch.gather(saliency_score, 0, torch.tensor(irrelevant_ids[i], dtype=torch.long).to(device))
        neg_gt_saliency_score = torch.gather(saliency_score_neg, 0, torch.tensor(relevant_ids[i], dtype=torch.long).to(device))

        inner_high, _ = torch.topk(inner_gt_saliency_score, k=int(max(len(relevant_ids[i]) // q, 1)), largest=True)
        if len(outer_gt_saliency_score) == 0:
            outer_high = saliency_score[0]
        else:
            outer_high, _ = torch.topk(outer_gt_saliency_score, k=1, largest=True)
        neg_high, _ = torch.topk(neg_gt_saliency_score, k=int(max(len(relevant_ids[i]) // q, 1)), largest=True)


        inner_high_score = torch.mean(inner_high).view(1)
        outer_high_score = torch.mean(outer_high).view(1)
        neg_high_score = torch.mean(neg_high).view(1)

        intra_ranking = F.relu(margin1 + outer_high_score - inner_high_score)
        inter_ranking = F.relu(margin2 + neg_high_score - inner_high_score)

        loss += (intra_ranking + inter_ranking)

    loss /= bs

    return loss.squeeze()

def div_loss(target, logits, seq_len, epsilon=1e-9, gt=False):
    """
    target: ground-truth like [[0,0,0,1,1,1,1,0], ..., [0,0,1,1,1,0,0,0]] (bs, T)
    logits: predictions like [[0.1,0.2,0.4,0.8,0.9,0.7,0.8,0.3], ..., [0.1,0.1,0.6,0.7,0.9,0.2,0.1,0.3]] (bs, T)
    """
    losses = []  # List to collect individual losses
    if gt:
        logits = torch.sigmoid(logits)
    else:
        logits = torch.sigmoid(logits)
        target = torch.sigmoid(target)

    for i in range(target.shape[0]):
        pred = logits[i, :seq_len[i]].squeeze()
        label = target[i, :seq_len[i]].squeeze()
        loss = torch.mean(-label.detach() * torch.log(pred + epsilon))
        losses.append(loss)  # Unsqueeze to make each loss (1,) shape

    stacked_losses = torch.stack(losses)
    return stacked_losses


def kl_loss(target, logits, seq_len, temp=0.07, eps=1e-9):
    """
    :param temp: temperature coefficient, range (0.01, 1),
    the larger, the prob distribution will be smoother; the smaller, the prob distribution will be sharper;
    """
    losses = []
    for i in range(target.shape[0]):
        pred = logits[i, :seq_len[i]]
        label = target[i, :seq_len[i]]
        softmax_pred = F.softmax(pred / temp, dim=-1)
        softmax_label = F.softmax(label / temp, dim=-1)

        loss = softmax_pred * torch.log((softmax_pred + eps) / (softmax_label + eps))
        losses.append(loss.sum())

    stacked_losses = torch.stack(losses)
    return stacked_losses


def bi_margin_loss(predictions, margin1, margin2):
    positive_distance_to_margin1 = F.relu(margin1 - predictions)
    positive_distance_to_margin2 = F.relu(predictions - margin2)

    # This ensures we only consider the nearest margin violation
    combined_loss = torch.max(positive_distance_to_margin1, positive_distance_to_margin2)
    return combined_loss


def fine_ranking(meta, outputs, hn_num=3, fine_margin=[0.25, 0.25, 0.25, 0.25], rt="abd"):

    if len(fine_margin) != (hn_num+1):
        raise ValueError

    video_len = meta["video_length"].long()  # [12, 18, 32, 20, 29]
    gt_windows = meta["saliency_all_labels"]  # [0,0,0,1,1,1,1,0,0,0,0]

    pos_saliency_score = outputs["saliency_scores"]  # origin query  # (bs, T)
    par1_saliency_score = outputs["saliency_scores_hn1"]  # partial positive 1  # (bs, T)
    par2_saliency_score = outputs["saliency_scores_hn2"]  # partial positive 2
    par3_saliency_score = outputs["saliency_scores_hn3"]  # partial positive 3
    neg_saliency_score = outputs["saliency_scores_neg"]  # other query in the batch

    # KL divergence
    # div1 = kl_loss(gt_windows, pos_saliency_score, video_len)
    # div2 = kl_loss(pos_saliency_score, par1_saliency_score, video_len)
    # div3 = kl_loss(pos_saliency_score, par2_saliency_score, video_len)
    # div4 = kl_loss(pos_saliency_score, par3_saliency_score, video_len)
    # div5 = kl_loss(pos_saliency_score, neg_saliency_score, video_len)

    # Cross Entropy
    div1 = div_loss(gt_windows, pos_saliency_score, video_len, gt=True)
    div2 = div_loss(pos_saliency_score, par1_saliency_score, video_len)
    div3 = div_loss(pos_saliency_score, par2_saliency_score, video_len)
    div4 = div_loss(pos_saliency_score, par3_saliency_score, video_len)
    div5 = div_loss(pos_saliency_score, neg_saliency_score, video_len)

    if rt == "abd":
        # absolute distance (first-order)
        rank1 = torch.sum(F.relu(fine_margin[0] + div1 - div2)) / len(video_len)
        rank2 = torch.sum(F.relu(fine_margin[1] + div2 - div3)) / len(video_len)
        rank3 = torch.sum(F.relu(fine_margin[2] + div2 - div4)) / len(video_len)
        rank4 = torch.sum(F.relu(fine_margin[3] + div2 - div5)) / len(video_len)
    elif rt == "red":
        # relative distance (second-order)
        rank1 = torch.sum(F.relu(fine_margin[0] + div1 - div2)) / len(video_len)
        rank2 = torch.sum(F.relu(fine_margin[1] + div2 - div3)) / len(video_len)
        rank3 = torch.sum(F.relu(fine_margin[2] + div3 - div4)) / len(video_len)
        rank4 = torch.sum(F.relu(fine_margin[3] + div4 - div5)) / len(video_len)

    rank_loss = rank1 + rank2 + rank3 + rank4

    return rank_loss.squeeze()


if __name__ == "__main__":

    target = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    par1_saliency_score = torch.tensor([[0.1, 0.2, 0.4, 0.8, 0.9, 0.7, 0.8, 0.3], [0.1, 0.1, 0.6, 0.7, 0.9, 0.2, 0.1, 0.3]],
                          dtype=torch.float32)
    par2_saliency_score = torch.tensor(
        [[0.15, 0.24, 0.34, 0.78, 0.89, 0.73, 0.58, 0.23], [0.21, 0.16, 0.56, 0.73, 0.69, 0.25, 0.14, 0.33]],
        dtype=torch.float32)
    # print('hello world')
    seq_len = [5, 6]
    # div1 = div_loss(target, par1_saliency_score, seq_len, gt=True)
    # div2 = div_loss(par1_saliency_score, par2_saliency_score, seq_len)

    div1 = kl_loss(par1_saliency_score, target, seq_len)
    div2 = kl_loss(par1_saliency_score, par2_saliency_score, seq_len)

    rank2 = bi_margin_loss(div2, 0,0.25).mean()

    delta = F.relu(0.2 + div1 - div2)
    rank1 = torch.sum(delta) / len(seq_len)
    print(rank1)
