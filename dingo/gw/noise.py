import torch


# TODO:
#  * Support transformations
#    1. add noise to detector_projected wf sample
#    2. whiten
#  * Caveat: Don't use PSDs with float32!
#  * simplest case: fixed designed sensitivity PSD
#  * noise needs to know the domain
#  * more complex: database of PSDs for each detector
#    - randomly select a PSD for each detector
#  * Maybe create a PSD_DataSet class (open / non-open data), and transform
#  * window_function

# pytorch transforms:
# - https://pytorch.org/docs/stable/distributions.html?highlight=transform#module-torch.distributions.transforms
# - torch.distributions.transforms.Transform(cache_size=0)
# - torch.distributions.transforms.ComposeTransform(parts, cache_size=0)
