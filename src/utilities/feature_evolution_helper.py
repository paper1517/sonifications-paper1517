import pickle
import matplotlib.pyplot as plt
import torch
from src.utilities import interpretability_utils
import tqdm
from tqdm import notebook


def get_selected_maps():
    with open("/media/user/nvme/contrastive_experiments/select_feature_maps.pkl", "rb") as fd:
        selected_maps = pickle.load(fd)
    return selected_maps


def get_selected_inputs():
    with open("/media/user/nvme/contrastive_experiments/selected_inputs_evolution.pkl", "rb") as fd:
        selected_inputs = pickle.load(fd)
    return selected_inputs


def get_max_activating_inputs_for_best_ckpt(layer_index, dataset,
                                            inv_lbl_map, net,
                                            selected_maps,
                                            top_per_map=9, to_exclude=[]):
    assert layer_index >= 1 and layer_index <= 11
    features_of_interest = {}
    random_maps = None
    used_data_points = []
    cnt = 0
    skipped = 0
    # for batch in tqdm.notebook.tqdm_notebook(loader, position=1):
    for ix in tqdm.notebook.tqdm_notebook(range(len(dataset)), position=1):
        x, y = dataset[ix]
        # print(x.shape, y.shape)
        # x, _, y = batch
        x = x.unsqueeze(0)
        min_ = x.min()
        max_ = x.max()
        if min_ < -1 and max_ > 1:
            print("IN ANALYZIE RANDOM MAX, INPUT MIN, MAX:", x.min(), x.max())
        idxs = torch.where(y == 1)[0].tolist()
        skip_flag = False
        for idx in idxs:
            if idx in to_exclude:
                skip_flag = True
                break
        if skip_flag:
            skipped += 1
            continue
        lbls = ";".join([inv_lbl_map[lbl_idx] for lbl_idx in idxs])
        # data.append(x)
        # gts.append(lbls)
        x = x.cuda()
        output_features, switch_indices = interpretability_utils.infer_model(net, x)
        act_feats = output_features['act{}'.format(layer_index)]

        if random_maps is None:
            random_maps = selected_maps['act{}'.format(layer_index)]
            for jx in random_maps:
                features_of_interest[jx] = []
        for m in random_maps:
            features_of_interest[m].append(act_feats[0, m, :].detach().cpu().mean())
        cnt += 1
        used_data_points.append(ix)
    print("Skipped:", skipped)
    indices = {}
    for k, values in features_of_interest.items():
        mean_activations = torch.tensor(values)
        idxs = torch.argsort(mean_activations, descending=True)[:top_per_map]
        indices[k] = idxs.tolist()
    return indices, used_data_points


def analyze_random_maps(layer_index, dataset, inv_lbl_map,
                        net, deconv,
                        max_activation_inputs, used_data_points):
    assert layer_index >= 1 and layer_index <= 11
    outputs = {}
    for k in max_activation_inputs.keys():
        outputs[k] = []

    for k, idxs in max_activation_inputs.items():
        for idx in idxs:
            # print(idx)
            # inp = data[idx].cuda()
            inp, y = dataset[used_data_points[idx]]
            label_indicators = torch.where(y == 1)[0].tolist()
            gt = ";".join([inv_lbl_map[lbl_idx] for lbl_idx in label_indicators])
            inp = inp.unsqueeze(0).cuda()
            with torch.no_grad():
                pred, output_features, switch_indices = net(inp, True)
                vis = deconv.visualize_specific_map(inp, output_features, switch_indices, layer_index, k)
            outputs[k].append({
                # "data": inp.detach().cpu(),
                "data_idx": used_data_points[idx],
                "vis": interpretability_utils.process_vis(vis.squeeze(), inp.squeeze().cpu().numpy()),
                "gt": gt
            })
    return outputs


def get_ckpt_indices(is_contrastive=False):
    if is_contrastive:
        ckpt_idxs = [100, 80, 60, 40, 20, 1]
    else:
        ckpt_idxs = [1] + [i * 10 for i in range(1, 6)]
        ckpt_idxs = ckpt_idxs[::-1]
    return ckpt_idxs


def process_features_over_training(exp_dir, ckpt_idxs, layer_index, selected_inputs,
                                   val_set, inv_lbl_map, is_contrastive=False):
    # load model
    all_results = {}
    indices, used_data_points = selected_inputs[layer_index]['indices'], selected_inputs[layer_index]['used_data_points']
    print(indices)
    for ckpt_idx in notebook.tqdm(ckpt_idxs, position=0):
        model, net, deconv, hparams = interpretability_utils.model_helper(exp_dir, is_contrastive, epoch_index=ckpt_idx)
        results_ckpt = analyze_random_maps(layer_index, val_set, inv_lbl_map, net, deconv, indices, used_data_points)
        all_results[ckpt_idx] = results_ckpt
    return all_results


def spec_helper(spec_list, filter_idx, layer_idx, ckpt_idxs, save_path=None):
    plt.close()
    plt.clf()
    figs, axs = plt.subplots(ncols=6, figsize=(30, 5))
    figs.suptitle('Feature Evolution: Layer {:02d} | Filter: {:02d}'.format(layer_idx, filter_idx))
    for ix in range(len(spec_list)):
        axs[5 - ix].imshow(spec_list[ix])
        axs[5 - ix].set_title("Epoch: {:02d}".format(ckpt_idxs[ix]))
        axs[5 - ix].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_evo_spectrograms(layer_results, layer_idx, ckpt_idxs):
    specs = {}
    for epoch, data in layer_results.items():
        for key, value in data.items():
            # print(key, value, len(value))
            vis = value[0]['vis']
            spec = interpretability_utils.get_spec(vis)
            try:
                specs[key].append(spec)
            except KeyError as ex:
                specs[key] = [spec]

    for k, v in specs.items():
        spec_helper(v, k, layer_idx, ckpt_idxs)


def plot_evo_spectrograms_noplot(layer_results, layer_idx, ckpt_idxs):
    specs = {}
    for epoch, data in layer_results.items():
        for key, value in data.items():
            # print(key, value, len(value))
            vis = value[0]['vis']
            spec = interpretability_utils.get_spec(vis)
            # print(spec.shape)
            # spec = torch.from_numpy(spec).unsqueeze
            torch_spec = torch.cat([torch.from_numpy(spec).unsqueeze(0)]*3)
            try:
                specs[key].append(torch_spec)
            except KeyError as ex:
                specs[key] = [torch_spec]
    return specs


def tile_spectrograms(specs):
    all_specs = []
    for key, val in specs.items():
        print(key)
        key_specs = []
        for item in val[::-1]:
            all_specs.append(item)
        all_specs.extend(key_specs)
    return all_specs
