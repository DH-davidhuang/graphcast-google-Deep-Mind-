import argparse
#from model import MyModel
#from trainer import train
import xarray as xr
from google.cloud import storage
import xarray as xr
import dataclasses


from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import autoregressive
from graphcast import rollout
from plot import Plotter


# @title Imports

import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
from old_model import GraphCastModel


def load_dataset(bucket, file_name):
    # Construct the full path to the file in the GCS bucket
    blob = bucket.blob(f"dataset/{file_name}")
    # Use a temporary file to download the data
    with blob.open("rb") as f:
        ds = xarray.load_dataset(f).compute()
    return ds


def data_valid_for_model(file_name, model_config, task_config):
    parts = file_name.split('_')
    file_info = {part.split('-')[0]: part.split('-')[1] for part in parts}
    
    resolution_valid = float(file_info['res']) == model_config.resolution
    levels_valid = int(file_info['levels']) in task_config.pressure_levels
    source_valid = file_info['source'] in ['era5', 'fake'] or \
        ('total_precipitation_6hr' not in task_config.input_variables and file_info['source'] == 'hres')
    
    return resolution_valid and levels_valid and source_valid

def select(self, data: xarray.Dataset, variable: str, level: Optional[int] = None, max_steps: Optional[int] = None) -> xarray.Dataset:
        data = data[variable]
        if "batch" in data.dims:
            data = data.isel(batch=0)
        if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
            data = data.isel(time=range(0, max_steps))
        if level is not None and "level" in data.coords:
            data = data.sel(level=level)
        return data

def scale(self, data: xarray.Dataset, center: Optional[float] = None, robust: bool = False) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax), ("RdBu_r" if center is not None else "viridis"))




def main():
    print(jax.devices())
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    #data_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
    dataset_file_name = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc"
    example_batch = load_dataset(gcs_bucket, dataset_file_name)
    print(example_batch['2m_temperature'])

    plotter = Plotter()

    plot_size = 7
    

    data = {
        " ": plotter.scale(plotter.select(example_batch, '2m_temperature', 500, 3),
                robust=True),
    }
    fig_title = "2m_temperature"
    if "level" in example_batch['2m_temperature'].coords:
        fig_title += f" at {500} hPa"

    plotter.plot_data(data, fig_title, plot_size, True)
    num_time_steps = example_batch.sizes["time"] - 2

    # era5_data = xr.open_zarr(data_path)

    # #model_config = graphcast.ModelConfig(resolution=1.0, 
    #                                      mesh_size=5, 
    #                                      latent_size=512, 
    #                                      gnn_msg_steps=16, 
    #                                      hidden_layers=1, 
    #                                      radius_query_fraction_edge_length=0.6, 
    #                                      mesh2grid_edge_normalization_factor=0.6180338738074472)

    
    # task_config = graphcast.TaskConfig(input_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity', 'toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos', 'geopotential_at_surface', 'land_sea_mask'), 
    #            target_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity'), 
    #            forcing_variables=('toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos'), 
    #            pressure_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000), 
    #            input_duration='12h')

    #model = MyModel(model_config, task_config, params_file="GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz")

    # Load the data for training and evaluation
        # Extract training data


    model = GraphCastModel(
        # model_config=model_config, 
        # task_config=task_config, 
        gcs_bucket_name=gcs_bucket, 
        params_file='GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
    )
    params_file='GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
    # @title Extract training and eval data
    train_steps = 6
    eval_steps = 6
    with gcs_bucket.blob(f"params/{params_file}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}

    model_config = ckpt.model_config
    task_config = ckpt.task_config
  
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
        **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config))


    init_jitted = jax.jit(model.with_configs(model.run_forward.init))

    if model.params is None:
        model.params, model.state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)

    loss_fn_jitted = model.drop_state(model.with_params(jax.jit(model.with_configs(model.loss_fn.apply))))
    grads_fn_jitted = model.with_params(jax.jit(model.with_configs(model.grads_fn)))
    run_forward_jitted = model.drop_state(model.with_params(jax.jit(model.with_configs(
        model.run_forward.apply))))
        

    # @title Autoregressive rollout (loop in python)
    print(f"model resolution {model.model_config.resolution}")
    print(eval_inputs.sizes["lon"])

    assert model.model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
    "Model resolution doesn't match the data resolution. You likely want to "
    "re-filter the dataset list, and download the correct data.")

    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    # @title Build jitted functions, and possibly initialize random weights

    def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig):
        # Deeper one-step predictor.
        predictor = graphcast.GraphCast(model_config, task_config)

        # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
        # from/to float32 to/from BFloat16.
        predictor = casting.Bfloat16Cast(predictor)

        # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
        # BFloat16 happens after applying normalization to the inputs/targets.
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=model.diffs_stddev_by_level,
            mean_by_level=model.mean_by_level,
            stddev_by_level=model.stddev_by_level)

        # Wraps everything so the one-step model can produce trajectories.
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor


    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)


    @hk.transform_with_state
    def loss_fn(model_config, task_config, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))

    def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), model_config, task_config,
                i, t, f)
            return loss, (diagnostics, next_state)
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True)(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def with_configs(fn):
        return functools.partial(
            fn, model_config=model_config, task_config=task_config)

    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    init_jitted = jax.jit(with_configs(run_forward.init))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)

    loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
    grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)




    # Initialize jitted functions for the model
    model.init_jitted_functions()

    predictions = model.run_model(eval_inputs, eval_targets, eval_forcings)


    predictions = model.run_forward_jitted(
        model.params, 
        model.state, 
        jax.random.PRNGKey(0), 
        eval_inputs, 
        eval_targets, 
        eval_forcings
    )

    # @title Autoregressive rollout (loop in python)

    assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
    "Model resolution doesn't match the data resolution. You likely want to "
    "re-filter the dataset list, and download the correct data.")

    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    predictions = rollout.chunked_prediction(
        model.run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)

    # Create an instance of the Plotter class
    plotter = Plotter()

    # Define the parameters for plotting
    plot_pred_variable = "2m_temperature"
    plot_pred_level = 500
    plot_pred_robust = True
    plot_pred_max_steps = 1

    # Select the data for plotting
    targets_data = select(eval_targets, plot_pred_variable, plot_pred_level, plot_pred_max_steps)
    predictions_data = select(predictions, plot_pred_variable, plot_pred_level, plot_pred_max_steps)
    diff_data = targets_data - predictions_data

    # Create a dictionary with the data to be plotted
    data = {
        "Targets": scale(targets_data, robust=plot_pred_robust),
        "Predictions": scale(predictions_data, robust=plot_pred_robust),
        "Diff": scale(diff_data, robust=plot_pred_robust, center=0),
    }

    # Define plot parameters
    fig_title = plot_pred_variable
    if "level" in predictions[plot_pred_variable].coords:
        fig_title += f" at {plot_pred_level} hPa"
    plot_size = 5

    # Call the plot_data method from the Plotter class to generate the plot
    plotter.plot_data(data, fig_title, plot_size, plot_pred_robust)

    loss, diagnostics = model.loss_fn_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings)
    print("Loss:", float(loss))

    # @title Gradient computation (backprop through time)
    loss, diagnostics, next_state, grads = model.grads_fn_jitted(
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings)
    mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
    print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

    # @title Autoregressive rollout (keep the loop in JAX)
    print("Inputs:  ", train_inputs.dims.mapping)
    print("Targets: ", train_targets.dims.mapping)
    print("Forcings:", train_forcings.dims.mapping)

    predictions = model.run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets * np.nan,
        forcings=train_forcings)


    #data = {
    #    "Targets": scale(select(eval_targets, "2m_temperature", 500, 10), robust=True),
    #    "Predictions": scale(select(predictions, "2m_temperature", 500, 10), robust=True),
     #   "Diff": scale((select(eval_targets, "2m_temperature", 500, 10) -
     #                       select(predictions, "2m_temperature", 500, 10)),
    #                   robust=True, center=0),
    #}
    
    # Call the method with a filename
    #plotter.plot_data_save_to_file(data, "Temperature at 500 hPa", filename="temperature_predictions.png")
        

    # Train the model
    #model.train
    return

if __name__ == "__main__":
   # parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path', type=str, required=True)
    #parser.add_argument('--model_config', type=str, required=True)
    # Add more arguments as needed

    #args = parser.parse_args()
    #main(args)
    main()
