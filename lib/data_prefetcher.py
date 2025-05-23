import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_rgb, self.next_gt,self.next_edge, _,_ = next(self.loader)
        except StopIteration:
            self.next_rgb = None
            self.next_t = None
            self.next_gt = None
            self.next_edge = None
            return

        with torch.cuda.stream(self.stream):
            self.next_rgb = self.next_rgb.cuda(non_blocking=True).float()
            self.next_gt = self.next_gt.cuda(non_blocking=True).float()
            self.next_edge = self.next_edge.cuda(non_blocking=True).float()


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        rgb = self.next_rgb
        # t= self.next_t
        gt = self.next_gt
        edge = self.next_edge
        self.preload()
        return rgb, gt, edge