import argparse

import torch

from dingo.core.utils.backward_compatibility import torch_load_with_fallback
from dingo.gw.result import Result

"""
2. March 2023: Previously, we stored the input_dim of the embedding net as a tuple, 
e.g., (2, 3, 8033). This is created in dingo.core.nn.nsf.autocomplete_model_kwargs_nsf, 
where we previously set:

    model_kwargs["embedding_net_kwargs"]["input_dims"] = data_sample[1].shape
    
This is problematic when saving the metadata as string in hdf5, since python does not 
properly convert the tuple to a string. This results in corrupted output with dingo_ls,

          input_dims: !!python/tuple
          - 2
          - 3
          - 8033
          
and causes trouble when interfacing with pesummary. In commit 
53929ae0ce5eee0607510119194167cf9db003a4, we thus changed the input_dim to be stored as 
a list:

    model_kwargs["embedding_net_kwargs"]["input_dims"] = list(data_sample[1].shape)
    
Old models and result files (pre commit 53929ae0ce5eee0607510119194167cf9db003a4) need 
to be updated with the present script to account for this change (note: training will 
still work, updates to old models are only required when running inference and 
interfacing with pesummary). 
"""


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_file", type=str, default=None)
    return parser.parse_args()


def change_dict_elements_recursively(d, target_key, f, previous_keys=()):
    for k, el in d.items():
        if k == target_key:
            print(f"Changing {previous_keys + (k,)} from {el} to {f(el)}.")
            d[k] = f(el)
        elif type(el) == dict:
            change_dict_elements_recursively(el, target_key, f, previous_keys + (k,))


def main():
    args = parse_args()

    if args.checkpoint is not None:
        print(f"Updating model {args.checkpoint}.")
        d, _ = torch_load_with_fallback(args.checkpoint)

        d["model_kwargs"]["embedding_net_kwargs"]["input_dims"] = list(
            d["model_kwargs"]["embedding_net_kwargs"]["input_dims"]
        )

        torch.save(d, args.checkpoint)

    if args.results_file is not None:
        print(f"Updating results file {args.results_file}.")
        res = Result(file_name=args.results_file)
        change_dict_elements_recursively(res.settings, "input_dims", list)
        res.to_file(file_name=args.results_file)


if __name__ == "__main__":
    main()
