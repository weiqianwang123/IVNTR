"""Utilities to support using the code in gnn.py."""

from __future__ import division

import collections
import logging
import time
import wandb
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from predicators.structs import Array
from itertools import product

def discrete_sigmoid(logits, temperature=1.0, boundary=0.5):
    """
    Sample from the Gumbel-Sigmoid distribution and optionally discretize.

    Args:
    - logits: [batch_size, num_classes] unnormalized log-probs
    - temperature: non-negative scalar

    Returns:
    - [batch_size, num_classes] sample from the Gumbel-Sigmoid distribution.
      If hard=True, then the returned samples are binary.
    """
    y_soft = torch.sigmoid(logits / temperature)
    # Straight-through trick: y_hard is binary, but gradients are like soft sample
    y_hard = (y_soft > boundary).float()
    y_hard = (y_hard - y_soft).detach() + y_soft
    return y_hard
    
def discrete_graph(logits_graph, temperature=1.0, boundary=0.5):
    binary_graph = {
        'nodes': None,
        'edges': None,
        "n_node": logits_graph['n_node'],
        "n_edge": logits_graph['n_edge'],
        "globals": logits_graph['globals'],
        "senders": logits_graph['senders'],
        "receivers": logits_graph['receivers']
    }
    nodes_logits = logits_graph['nodes']
    nodes_sample = discrete_sigmoid(nodes_logits, temperature, boundary)
    binary_graph['nodes'] = nodes_sample
    edges_logits = logits_graph['edges']
    edges_sample = discrete_sigmoid(edges_logits, temperature, boundary)
    binary_graph['edges'] = edges_sample
    return binary_graph

def dicts_are_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    
    for key in dict1:
        if isinstance(dict1[key], dict):
            if not dicts_are_equal(dict1[key], dict2[key]):
                return False
        elif isinstance(dict1[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        elif isinstance(dict1[key], torch.Tensor):
            if not torch.equal(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True

def train_model(
    model: Any,
    dataloaders: Dict,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    global_criterion: Optional[Callable[[torch.Tensor, torch.Tensor],
                                        torch.Tensor]],
    num_epochs: int,
    do_validation: bool,
    device: Optional[torch.device] = None,
) -> OrderedDict[str, torch.Tensor]:
    """Optimize the model and save checkpoints."""
    since = time.perf_counter()

    # Note: best_seen_model_weights is measured on validation (not train) loss.
    best_seen_model_weights: OrderedDict[
        str, torch.Tensor] = collections.OrderedDict({})
    best_seen_model_train_loss = np.inf
    best_seen_running_validation_loss = np.inf
    if device is not None:
        model.to(device)

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            logging.info(f'Epoch {epoch}/{num_epochs - 1}')
            logging.info('-' * 10)
        # Each epoch has a training and validation phase
        if epoch % 10 == 0 and do_validation:
            phases = ['train', 'val']
        else:
            phases = ['train']

        running_loss = {'train': [], 'val': []}
        if not do_validation:
            del running_loss['val']

        for phase in phases:

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['graph_input']
                targets = data['graph_target']
                for key in targets.keys():
                    if targets[key] is not None:
                        targets[key] = targets[key].detach()
                if device is not None:
                    for key, val in inputs.items():
                        inputs[key] = val.to(device) if val is not None else val
                    for key, val in targets.items():
                        targets[key] = val.to(device) if val is not None else val
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs.copy())
                output = outputs[-1]

                loss = torch.tensor(0.0)
                if device is not None:
                    loss = loss.to(device)
                if criterion is not None:
                    loss += criterion(output['nodes'], targets['nodes'])

                if global_criterion is not None:
                    global_loss = global_criterion(output['globals'],
                                                   targets['globals'])
                    loss += global_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()  # type: ignore
                    optimizer.step()

                # statistics
                running_loss[phase].append(loss.item())

        if epoch % 10 == 0:
            logging.info(f"running_loss (Train): {np.mean(running_loss['train'])}")
            if len(running_loss['val']):
                logging.info(f"running_loss (Val): {np.mean(running_loss['val'])}")

            if do_validation and \
                    np.mean(running_loss['val']) < best_seen_running_validation_loss:
                best_seen_running_validation_loss = np.mean(running_loss['val'])
                best_seen_model_weights = model.state_dict()
                best_seen_model_train_loss = np.mean(running_loss['train'])
                logging.info(
                    "Found new best model with validation loss "
                    f"{best_seen_running_validation_loss} at epoch {epoch}")

    time_elapsed = time.perf_counter() - since
    num_min = time_elapsed // 60
    num_sec = time_elapsed % 60

    if not do_validation:
        train_loss = np.mean(running_loss['train'])
        logging.info(f"Training complete in {num_min:.0f}m {num_sec:.0f}s "
                     f"with train loss {train_loss:.5f}")
        return model.state_dict()
    logging.info(
        f"Training complete in {num_min:.0f}m {num_sec:.0f}s "
        f"with train loss {best_seen_model_train_loss:.5f} and validation "
        f"loss {best_seen_running_validation_loss:.5f}")

    assert best_seen_model_weights
    return best_seen_model_weights

def compute_normalizers(
    data: List[Dict],
    normalize_nodes: List[bool],
    normalize_edges: List[bool],
    normalize_globals: bool = True,
) -> Dict[str, Tuple[Array, Array]]:
    """Compute the normalizers of the given list of graphs.

    These can be passed into normalize_graph.
    """
    node_data_lst, edge_data_lst, global_data_lst = [], [], []
    for graph in data:
        node_data_lst.extend(graph["nodes"])
        edge_data_lst.extend(graph["edges"])
        global_data_lst.append(graph["globals"])
    node_data = np.array(node_data_lst)
    edge_data = np.array(edge_data_lst)
    global_data = np.array(global_data_lst)
    normalizers = {}
    if (~np.array(normalize_nodes)).any() and len(node_data):
        if not (~np.array(normalize_nodes)).all():
            logging.warning("Some nodes are recognized as rotation quanterion")
        indicator = ~np.array(normalize_nodes) # normalize non rot feats
        node_normalizers = _compute_normalizer_array(node_data, indicator)
        normalizers["nodes"] = node_normalizers
    if (~np.array(normalize_edges)).any() and len(edge_data):
        if not (~np.array(normalize_edges)).all():
            logging.warning("Some edges are recognized as rotation quanterion")
        indicator = ~np.array(normalize_edges) # normalize non rot feats
        edge_normalizers = _compute_normalizer_array(edge_data, indicator)
        normalizers["edges"] = edge_normalizers
    if normalize_globals and len(global_data):
        indicator = np.array([True] * global_data.shape[1])
        global_normalizers = _compute_normalizer_array(global_data, indicator)
        normalizers["globals"] = global_normalizers
    return normalizers

def _compute_normalizer_array(array_data: Array, indicator: Array) -> Tuple[Array, Array]:
    """Helper function for compute_normalizers()."""
    shift = np.min(array_data, axis=0)
    scale = np.max(array_data - shift, axis=0)
    shift[np.where(scale == 0)] = 0
    scale[np.where(scale == 0)] = 1

    # If the indicator is False, then the shift and scale should be 0 and 1
    shift = shift * indicator
    scale = scale * indicator + (1 - indicator)
    return shift, scale


def normalize_graph(graph: Dict,
                    normalizers: Dict[str, Tuple[Array, Array]],
                    invert: bool = False) -> Dict:
    """Normalize the given graph using the given normalizers, which were
    returned by compute_normalizers() on some other list of graphs."""
    assert set(normalizers.keys()).issubset(set(graph.keys()))
    if invert:
        transform = _invert_normalize_array
    else:
        transform = _normalize_array
    new_graph = {}
    for k in graph:
        if k in normalizers:
            new_graph[k] = transform(graph[k], normalizers[k])
        else:
            new_graph[k] = graph[k]
    return new_graph


def _normalize_array(array_data: Array, normalizer: Tuple[Array,
                                                          Array]) -> Array:
    """Helper function for normalize_graph()."""
    shift, scale = normalizer
    return (array_data - shift) / scale


def _invert_normalize_array(array_data: Array,
                            normalizer: Tuple[Array, Array]) -> Array:
    """Helper function for normalize_graph()."""
    shift, scale = normalizer
    return (array_data * scale) + shift


def get_single_model_prediction(model: Any,
                                single_input: Dict,
                                device: Optional[torch.device] = None) -> Dict:
    """Get a prediction from the given model on the given input."""
    if device is not None:
        model.to(device)
    model.train(False)
    model.eval()
    inputs = _create_super_graph([single_input], device=device)
    if device is not None:
        inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(inputs.copy())
    graphs = split_graphs(_convert_to_data(outputs[-1]))
    assert len(graphs) == 1
    graph = graphs[0]
    graph['nodes'] = graph['nodes'].numpy()
    graph['senders'] = graph['senders'].numpy()
    graph['receivers'] = graph['receivers'].numpy()
    graph['edges'] = graph['edges'].numpy()
    if graph['globals'] is not None:
        graph['globals'] = graph['globals'].numpy()
    graph['n_node'] = graph['n_node'].item()
    graph['n_edge'] = graph['n_edge'].item()
    return graph


def _compute_stacked_offsets(sizes: List[Array],
                             repeats: List[Any]) -> torch.Tensor:
    """Computes offsets to add to indices of stacked np arrays.

    When a set of np arrays are stacked, the indices of those from the second
    one on must be offset in order to be able to index into the stacked np
    array. This function computes those offsets.
    Args:
    sizes: A 1D sequence of np arrays of the sizes per graph.
    repeats: A 1D sequence of np arrays of the number of repeats per graph.
    Returns:
    The index offset per graph.
    """
    idxs = np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)
    return torch.LongTensor(idxs)


def _convert_to_data(graph: Dict) -> Dict:
    for key in graph.keys():
        if graph[key] is not None:
            graph[key] = graph[key].cpu().data
    return graph


def replace_graph(graph: Dict, graph_dict: Dict) -> Dict:
    """Return a copy of the given graph with some values replaced."""
    out_graph = graph.copy()
    for key in graph_dict.keys():
        out_graph[key] = graph_dict[key]
    return out_graph


def _unstack(array: Array) -> List[torch.Tensor]:
    num_splits = int(array.shape[0])
    return [
        torch.squeeze(x, dim=0)  # type: ignore
        for x in np.split(array, num_splits, axis=0)
    ]


def split_graphs(graph: Dict) -> List[Dict]:
    """Splits the stored data into a list of individual data dicts.

    Each list is a dictionary with fields NODES, EDGES, GLOBALS, RECEIVERS,
    SENDERS.
    Args:
      graph: A graph containing numpy arrays.
    Returns:
      A list of the graph data dictionaries. The GLOBALS field is a tensor of
      rank at least 1, as the RECEIVERS and SENDERS field (which have integer
      values). The NODES and EDGES fields have rank at least 2.
    """
    offset = _compute_stacked_offsets(graph['n_node'].view(-1),
                                      graph['n_edge'].view(-1))
    nodes_splits = np.cumsum(graph['n_node'][:-1])
    edges_splits = np.cumsum(graph['n_edge'][:-1])
    graph_of_lists: Dict[str, Any] = collections.defaultdict(lambda: [])
    if graph['nodes'] is not None:
        graph_of_lists['nodes'] = np.split(graph['nodes'], nodes_splits)
    if graph['edges'] is not None:
        graph_of_lists['edges'] = np.split(graph['edges'], edges_splits)
    if graph['receivers'] is not None:
        graph_of_lists['receivers'] = np.split(graph['receivers'] - offset,
                                               edges_splits)
        graph_of_lists['senders'] = np.split(graph['senders'] - offset,
                                             edges_splits)
    if graph['globals'] is not None:
        graph_of_lists['globals'] = _unstack(graph['globals'])

    n_graphs = graph['n_node'].shape[0]
    # Make all fields the same length.
    for k in ['nodes', 'edges', 'globals']:
        graph_of_lists[k] += [None] * (n_graphs - len(graph_of_lists[k]))
    graph_of_lists['n_node'] = graph['n_node']
    graph_of_lists['n_edge'] = graph['n_edge']

    result = []
    for index in range(n_graphs):
        result.append({
            field: graph_of_lists[field][index]
            for field in [
                'nodes', 'edges', 'receivers', 'senders', 'globals', 'n_node',
                'n_edge'
            ]
        })
    return result


def concat_graphs(input_graphs: List[Dict], dim: int) -> Dict:
    """Concatenate NODES, EDGES and GLOBALS dimensions along `dim`. If a field
    is `None`, the concatenation is just `None`.

    There is an underlying assumption that the RECEIVERS, SENDERS, N_NODE
    and N_EDGE fields of the graphs in `values` should all match, but this
    is not checked by this op. The graphs in `input_graphs` should have the
    same set of keys for which the corresponding fields are not `None`.
    Args:
      input_graphs: A list of at least two graphs containing `Tensor`s and
      satisfying the constraints outlined above.
      dim: A dim to concatenate on.
    Returns: An op that returns the concatenated graphs.
    """
    assert dim != 0
    assert len(input_graphs) > 1
    nodes_lst = [gr['nodes'] for gr in input_graphs if gr['nodes'] is not None]
    edges_lst = [gr['edges'] for gr in input_graphs if gr['edges'] is not None]
    globals_lst = [
        gr['globals'] for gr in input_graphs if gr['globals'] is not None 
    ]

    nodes = torch.cat(nodes_lst, dim) if len(nodes_lst) else None
    edges = torch.cat(edges_lst, dim) if len(edges_lst) else None
    globals_ = torch.cat(globals_lst, dim) if len(globals_lst) else None

    output = replace_graph(input_graphs[0], {
        'nodes': nodes,
        'edges': edges,
        'globals': globals_
    })
    return output


class GraphDictDataset(Dataset):
    """A Dataset that stores input and output graphs."""

    def __init__(self, graph_dicts_input: List[Dict],
                 graph_dicts_target: List[Dict]):
        self.graph_dicts_input = graph_dicts_input
        self.graph_dicts_target = graph_dicts_target

    def __len__(self) -> int:
        return len(self.graph_dicts_input)

    def __getitem__(self, idx: int) -> Dict:
        sample = {
            'graph_input': self.graph_dicts_input[idx],
            'graph_target': self.graph_dicts_target[idx]
        }
        return sample
    
class GraphTransC2DDataset(Dataset):
    """A Dataset that combine all the data pairs in two lists."""

    def __init__(self, data_list: List[Dict]
                 ):
        self.data_list = data_list
        logging.info(f"Created transition dataset with {len(data_list)} transition pairs")
        np.random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def _create_super_graph(batches: List[Dict],
                        device: Optional[torch.device] = None) -> Dict:
    nodes = batches[0]['nodes']
    edges = batches[0]['edges']
    if 'receivers' in batches[0]:
        receivers = batches[0]['receivers'][:, None]
        senders = batches[0]['senders'][:, None]
    globals_ = batches[0]['globals']
    if globals_ is not None and len(globals_.shape) < 2:
        globals_ = globals_[None, :]

    num_nodes = np.array(batches[0]['n_node'], ndmin=2)
    num_edges = np.array(batches[0]['n_edge'], ndmin=2)

    for b in batches[1:]:
        nodes = np.vstack((nodes, b['nodes']))
        if 'receivers' in batches[0]:
            receivers = np.vstack(
                (receivers, (b['receivers'] + np.sum(num_nodes))[:, None]))
            senders = np.vstack(
                (senders, (b['senders'] + np.sum(num_nodes))[:, None]))
        if globals_ is not None:
            globals_ = np.vstack((globals_, b['globals']))
        edges = np.vstack((edges, b['edges']))

        num_nodes = np.vstack((num_nodes, b['n_node']))
        num_edges = np.vstack((num_edges, b['n_edge']))
    if 'receivers' in batches[0]:
        super_graph = {
            'n_node':
            torch.from_numpy(num_nodes),
            'n_edge':
            torch.from_numpy(num_edges),
            'nodes':
            torch.from_numpy(nodes).float().requires_grad_(),
            'edges':
            torch.from_numpy(edges).float().requires_grad_(),
            'receivers':
            torch.LongTensor(list(map(int, receivers))),
            'senders':
            torch.LongTensor(list(map(int, senders))),
            'globals': (torch.from_numpy(globals_).float().requires_grad_()
                        if globals_ is not None else None),
        }
    else:
        super_graph = {
            'n_node':
            torch.from_numpy(num_nodes),
            'n_edge':
            torch.from_numpy(num_edges),
            'nodes':
            torch.from_numpy(nodes).float().requires_grad_(),
            'edges':
            torch.from_numpy(edges).float().requires_grad_(),
            'globals': (torch.from_numpy(globals_).float().requires_grad_()
                        if globals_ is not None else None),
        }
    # Convert Tensors to device
    if device is not None:
        for key, val in super_graph.items():
            super_graph[key] = val.to(device) if val is not None else val
    return super_graph

def _create_super_graph_action(batches: List[Dict],
                        action_info: List[Dict],
                        device: Optional[torch.device] = None) -> Tuple[Dict, Dict]:
    nodes = batches[0]['nodes']
    edges = batches[0]['edges']
    action_node_ids = action_info[0]['action_node_ids'][None, :]
    action_id = np.array([action_info[i]['action_id'] for i in range(len(action_info))])
    action_arity = np.array([action_info[i]['action_arity'] for i in range(len(action_info))])
    if 'receivers' in batches[0]:
        receivers = batches[0]['receivers'][:, None]
        senders = batches[0]['senders'][:, None]
    globals_ = batches[0]['globals']
    if globals_ is not None and len(globals_.shape) < 2:
        globals_ = globals_[None, :]

    num_nodes = np.array(batches[0]['n_node'], ndmin=2)
    num_edges = np.array(batches[0]['n_edge'], ndmin=2)

    for i in range(1, len(batches)):
        b = batches[i]
        a = action_info[i]
        nodes = np.vstack((nodes, b['nodes']))
        if 'receivers' in batches[0]:
            receivers = np.vstack(
                (receivers, (b['receivers'] + np.sum(num_nodes))[:, None]))
            senders = np.vstack(
                (senders, (b['senders'] + np.sum(num_nodes))[:, None]))
            action_node_ids = np.vstack(
                (action_node_ids, (a['action_node_ids'] + np.sum(num_nodes))))

        if globals_ is not None:
            globals_ = np.vstack((globals_, b['globals']))
        edges = np.vstack((edges, b['edges']))

        num_nodes = np.vstack((num_nodes, b['n_node']))
        num_edges = np.vstack((num_edges, b['n_edge']))
        
    if 'receivers' in batches[0]:
        super_graph = {
            'n_node':
            torch.from_numpy(num_nodes),
            'n_edge':
            torch.from_numpy(num_edges),
            'nodes':
            torch.from_numpy(nodes).float().requires_grad_(),
            'edges':
            torch.from_numpy(edges).float().requires_grad_(),
            'receivers':
            torch.LongTensor(list(map(int, receivers))),
            'senders':
            torch.LongTensor(list(map(int, senders))),
            'globals': (torch.from_numpy(globals_).float().requires_grad_()
                        if globals_ is not None else None),
        }
    else:
        super_graph = {
            'n_node':
            torch.from_numpy(num_nodes),
            'n_edge':
            torch.from_numpy(num_edges),
            'nodes':
            torch.from_numpy(nodes).float().requires_grad_(),
            'edges':
            torch.from_numpy(edges).float().requires_grad_(),
            'globals': (torch.from_numpy(globals_).float().requires_grad_()
                        if globals_ is not None else None),
        }
    # Convert Tensors to device
    new_action_info = {
        'action_node_ids': torch.LongTensor(action_node_ids),
        'action_id': torch.LongTensor(list(map(int, action_id))),
        'action_arity': torch.LongTensor(list(map(int, action_arity)))
    }
    if device is not None:
        for key, val in super_graph.items():
            super_graph[key] = val.to(device) if val is not None else val
        for key, val in new_action_info.items():
            new_action_info[key] = val.to(device) if val is not None else val
    return super_graph, new_action_info


def graph_batch_collate(batch: List[Dict],
                        device: Optional[torch.device] = None) -> Dict:
    """Collate the given batch of graphs.

    Assumes batch is a dictionary where each key contains a list of
    graphs.
    """
    return {
        key: _create_super_graph([d[key] for d in batch], device=device)
        for key in batch[0]
    }

def action_graph_batch_collate(batch: List[Dict],
                             device: Optional[torch.device] = None) -> Dict:
    """Collate the given batch of paired two graph_datas.
    create super graphs for each data pair.
    Specifically, the "input_graphs" in each "data1" in combined,
    the "target_graphs" in each "data1" in combined.
    the "input_graphs" in each "data2" in combined,
    the "target_graphs" in each "data2" in combined.
    The returned data dict still has "data1" and "data2" keys.

    Note, the tricky part is that the "action_node_ids" will also 
    change to the new super graph.
    """
    # assert len(batch) == 1, "Only support batch size 1 now"
    action_infos = [d['action_info'] for d in batch]
    new_batch = {}
    new_action_info = None
    for key in batch[0]:
        if key == 'action_info':
            continue
        new_batch[key], local_new_action = _create_super_graph_action([d[key] for d in batch], 
                                                             action_infos,
                                                             device=device)
        if new_action_info is not None:
            assert dicts_are_equal(new_action_info, local_new_action)
        else:
            new_action_info = local_new_action.copy()

    new_batch['action_info'] = new_action_info
    return new_batch

def get_graph_batch_collate_with_device(device: torch.device) -> Callable:
    """Return a graph_batch_collate function that is given a device."""
    return lambda batch: graph_batch_collate(batch, device=device)
