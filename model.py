import haiku as hk
import jax

from google.cloud import storage
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import autoregressive
from graphcast import rollout
from graphcast.modified_deep_typed_graph_net import ModifiedDeepTypedGraphNet
from google.cloud import storage
from graphcast import checkpoint, data_utils, graphcast, normalization, autoregressive, rollout
import xarray
import numpy as np

class MyModel:
    def __init__(self, model_config, task_config, params_file=None):
        self.model_config = model_config
        self.task_config = task_config
        self.params = None
        self.state = {}
        if params_file:
            self.load_model(params_file)
        

    def load_model(self, params_file):
        gcs_client = storage.Client.create_anonymous_client()
        gcs_bucket = gcs_client.get_bucket("dm_graphcast")
        # Load normalization data and model checkpoint
        self.diffs_stddev_by_level = self._load_xarray_data(gcs_bucket, "stats/diffs_stddev_by_level.nc")
        self.mean_by_level = self._load_xarray_data(gcs_bucket, "stats/mean_by_level.nc")
        self.stddev_by_level = self._load_xarray_data(gcs_bucket, "stats/stddev_by_level.nc")
        self._load_checkpoint(gcs_bucket, f"params/{params_file}")

    def _load_xarray_data(self, bucket, path):
        with bucket.blob(path).open("rb") as f:
            return xarray.load_dataset(f).compute()

    def _load_checkpoint(self, bucket, path):
        with bucket.blob(path).open("rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        self.params = self.update_unchanged_layers(ckpt.params)

    def update_unchanged_layers(self, loaded_params):
        updated_params = self.params.copy() if self.params else {}
        for key, value in loaded_params.items():
            updated_params.setdefault(key, value)
        return updated_params

    def construct_predictor(self):
        modified_predictor = graphcast.GraphCast(self.model_config, self.task_config)
        modified_predictor = normalization.InputsAndResiduals(modified_predictor, 
                                                             self.diffs_stddev_by_level, 
                                                             self.mean_by_level, 
                                                             self.stddev_by_level)
        return autoregressive.Predictor(modified_predictor, gradient_checkpointing=True)

    def gaussian_nll_loss(y_true, y_pred_mean, y_pred_std):
        precision = 1 / (y_pred_std ** 2)
        return np.sum(precision * (y_true - y_pred_mean) ** 2 + np.log(y_pred_std ** 2))

    def train(self, train_inputs, train_targets, train_forcings, learning_rate=0.01):
            # Transform the predictor function with Haiku and state
            @hk.transform_with_state
            def run_model(inputs, targets_template, forcings):
                predictor = self.construct_predictor()
                return predictor(inputs, targets_template=targets_template, forcings=forcings)

            # Function to compute loss and diagnostics
            def loss_fn(params, state, inputs, targets, forcings):
                (predictions, next_state) = run_model.apply(params, state, None, inputs, targets, forcings)
                loss = self.gaussian_nll_loss(targets, predictions[0], predictions[1])  # Assuming targets, mean, and std are in the correct format
                return loss, next_state

            # Function to compute gradients
            def grads_fn(params, state, inputs, targets, forcings):
                (loss, next_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, inputs, targets, forcings)
                return grads, loss, next_state

            # Initialize parameters and state if not already done
            if self.params is None:
                self.params, self.state = run_model.init(jax.random.PRNGKey(0), train_inputs, train_targets, train_forcings)

            # Compute gradients
            grads, loss, next_state = grads_fn(self.params, self.state, train_inputs, train_targets, train_forcings)

            # Update parameters using SGD
            self.params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)

            # Update state
            self.state = next_state

            # Return the loss for monitoring
            return loss
    
    def evaluate(self, eval_inputs, eval_targets, eval_forcings):
            # Implement evaluation logic here
            pass

    
    
    def predict(self, inputs, targets_template, forcings):
        # Get predictions from the modified network
        predictions = self.predictor(inputs, targets_template, forcings)
        mean_predictions, std_predictions = predictions[..., :predictions.shape[-1] // 2], predictions[..., predictions.shape[-1] // 2:]
        return mean_predictions, std_predictions


