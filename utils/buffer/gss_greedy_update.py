import torch
import torch.nn.functional as F

from utils.buffer.buffer_utils import get_grad_vector, cosine_similarity
from utils.utils import maybe_cuda

class GSSGreedyUpdate(object):
    def __init__(self, params):
        super().__init__()
        # the number of gradient vectors to estimate new samples similarity, line 5 in alg.2
        self.mem_strength = params.gss_mem_strength
        self.gss_batch_size = params.gss_batch_size
        self.buffer_score = maybe_cuda(torch.FloatTensor(params.mem_size).fill_(0))

    def update(self, buffer, x, y, **kwargs):
        buffer.model.eval()

        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())

        place_left = buffer.buffer_img.size(0) - buffer.current_index
        if place_left <= 0:  # buffer is full
            batch_sim, mem_grads = self.get_batch_sim(buffer, grad_dims, x, y)
            if batch_sim < 0:
                buffer_score = self.buffer_score[:buffer.current_index].cpu()
                buffer_sim = (buffer_score - torch.min(buffer_score)) / \
                             ((torch.max(buffer_score) - torch.min(buffer_score)) + 0.01)
                # draw candidates for replacement from the buffer
                index = torch.multinomial(buffer_sim, x.size(0), replacement=False)
                # estimate the similarity of each sample in the recieved batch
                # to the randomly drawn samples from the buffer.
                batch_item_sim = self.get_each_batch_sample_sim(buffer, grad_dims, mem_grads, x, y)
                # normalize to [0,1]
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(1)
                # draw an event to decide on replacement decision
                outcome = torch.multinomial(torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1), 1,
                                            replacement=False)
                # replace samples with outcome =1
                added_indx = torch.arange(end=batch_item_sim.size(0))
                sub_index = outcome.squeeze(1).bool()
                buffer.buffer_img[index[sub_index]] = x[added_indx[sub_index]].clone()
                buffer.buffer_label[index[sub_index]] = y[added_indx[sub_index]].clone()
                self.buffer_score[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
        else:
            offset = min(place_left, x.size(0))
            x = x[:offset]
            y = y[:offset]
            # first buffer insertion
            if buffer.current_index == 0:
                batch_sample_memory_cos = torch.zeros(x.size(0)) + 0.1
            else:
                # draw random samples from buffer
                mem_grads = self.get_rand_mem_grads(buffer, grad_dims)
                # estimate a score for each added sample
                batch_sample_memory_cos = self.get_each_batch_sample_sim(buffer, grad_dims, mem_grads, x, y)
            buffer.buffer_img[buffer.current_index:buffer.current_index + offset].data.copy_(x)
            buffer.buffer_label[buffer.current_index:buffer.current_index + offset].data.copy_(y)
            self.buffer_score[buffer.current_index:buffer.current_index + offset] \
                .data.copy_(batch_sample_memory_cos)
            buffer.current_index += offset
        buffer.model.train()

    def get_batch_sim(self, buffer, grad_dims, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            batch_x: batch images
            batch_y: batch labels
        Returns: score of current batch, gradient from memory subsets
        """
        mem_grads = self.get_rand_mem_grads(buffer, grad_dims)
        buffer.model.zero_grad()
        loss = F.cross_entropy(buffer.model.forward(batch_x), batch_y)
        loss.backward()
        batch_grad = get_grad_vector(buffer.model.parameters, grad_dims).unsqueeze(0)
        batch_sim = max(cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, buffer, grad_dims):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
        Returns: gradient from memory subsets
        """
        gss_batch_size = min(self.gss_batch_size, buffer.current_index)
        num_mem_subs = min(self.mem_strength, buffer.current_index // gss_batch_size)
        mem_grads = maybe_cuda(torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32))
        shuffeled_inds = torch.randperm(buffer.current_index)
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                                i * gss_batch_size:i * gss_batch_size + gss_batch_size]
            batch_x = buffer.buffer_img[random_batch_inds]
            batch_y = buffer.buffer_label[random_batch_inds]
            buffer.model.zero_grad()
            loss = F.cross_entropy(buffer.model.forward(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(get_grad_vector(buffer.model.parameters, grad_dims))
        return mem_grads

    def get_each_batch_sample_sim(self, buffer, grad_dims, mem_grads, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = maybe_cuda(torch.zeros(batch_x.size(0)))
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            buffer.model.zero_grad()
            ptloss = F.cross_entropy(buffer.model.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            this_grad = get_grad_vector(buffer.model.parameters, grad_dims).unsqueeze(0)
            cosine_sim[i] = max(cosine_similarity(mem_grads, this_grad))
        return cosine_sim
