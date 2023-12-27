from torch.utils.data import dataloader

class InfiniteDataLoader(dataloader.DataLoader):
  """ Dataloader that reuses workers
  Uses same syntax as vanilla DataLoader
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
    self.iterator = super().__iter__()
  def __len__(self):
    return len(self.batch_sampler.sampler)
  def __iter__(self):
    for _ in range(len(self)):
      yield next(self.iterator)

class _RepeatSampler:
  """ Sampler that repeats forever
  Args:
      sampler (Sampler)
  """
  def __init__(self, sampler):
    self.sampler = sampler
  def __iter__(self):
    while True:
      yield from iter(self.sampler)