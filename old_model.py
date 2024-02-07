import xarray
import jax
import haiku as hk
from google.cloud import storage
from graphcast import normalization, autoregressive, casting
from graphcast import graphcast

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

class GraphCastModel:
    def __init__(self, gcs_bucket_name, params_file=None):
        self.model_config = None
        self.task_config = None
        self.gcs_bucket_name = gcs_bucket_name
        self.params_file = params_file
        self.gcs_client = storage.Client.create_anonymous_client()
        self.gcs_bucket = self.gcs_client.get_bucket(gcs_bucket_name)
        self.params = None
        self.state = {}
        self.load_normalization_data()
        self.load_model_params()

        # Initialize jitted functions
        #self.init_jitted_functions()

    def load_normalization_data(self):
        self.diffs_stddev_by_level = self.load_xarray_data("stats/diffs_stddev_by_level.nc")
        self.mean_by_level = self.load_xarray_data("stats/mean_by_level.nc")
        self.stddev_by_level = self.load_xarray_data("stats/stddev_by_level.nc")

    def load_xarray_data(self, file_path):
        blob = self.gcs_bucket.blob(file_path)
        with blob.open("rb") as f:
            return xarray.load_dataset(f).compute()

    def load_model_params(self):
        blob = self.gcs_bucket.blob(f"params/{self.params_file}")
        with blob.open("rb") as f:
            self.params = checkpoint.load(f, graphcast.CheckPoint)
        print(self.params.model_config)
        self.model_config = self.params.model_config
        self.task_config = self.params.task_config

    def construct_predictor(self):
        predictor = graphcast.GraphCast(self.model_config, self.task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, self.diffs_stddev_by_level, self.mean_by_level, self.stddev_by_level)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(self, inputs, targets_template, forcings):
        predictor = self.construct_predictor()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(self, inputs, targets, forcings):
        predictor = self.construct_predictor()
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

    def grads_fn(self, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = self.loss_fn.apply(params, state, jax.random.PRNGKey(0), self.model_config, self.task_config, i, t, f)
            return loss, (diagnostics, next_state)
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(self.params, self.state, inputs, targets, forcings)
        return grads, loss, next_state
    
    def drop_state(self, fn):
        return lambda **kw: fn(**kw)[0]

    def with_configs(self, fn):
        return functools.partial(fn, model_config=self.model_config, task_config=self.task_config)

    def with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)
    
    def init_jitted_functions(self):
        # Assuming construct_wrapped_graphcast, run_forward, and loss_fn are defined as before
        # Prepare partial functions with configurations
        run_forward_with_configs = functools.partial(self.run_forward, self.model_config, self.task_config)
        loss_fn_with_configs = functools.partial(self.loss_fn, self.model_config, self.task_config)

        # Jit the functions with configurations
        self.run_forward_jitted = jax.jit(run_forward_with_configs)
        self.loss_fn_jitted = jax.jit(loss_fn_with_configs)

        # Initialize params and state if not loaded
        if self.params is None:
            self.params, self.state = self.run_forward_jitted.init(jax.random.PRNGKey(0), jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)))

    def run_model(self, inputs, targets_template, forcings):
        # Example method to run the model, dropping state
        predictions = self.run_forward_jitted(self.params, self.state, inputs, targets_template, forcings)[0]
        return predictions