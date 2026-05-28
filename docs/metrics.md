We support automatically collecting metrics such as
1. High level system metrics such as MFU, average loss, max loss and words per second along with some
2. Memory metrics to measure max VRAM consumption and the number of OOMs
3. Timing metrics to measure data loading bottlenecks

Those metrics can then be visualized in either a TensorBoard or WandDB dashboard

## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a config_registry function or from CLI).

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.

## Weights and Biases

Weights and Biases will automatically send metrics to a remote server if you login with `wandb login`

So all you need to do is make sure that `metrics.enable_wandb` is enabled

For an example you can inspect the Llama 3 [config_registry.py](../torchtitan/models/llama3/config_registry.py)

Note that if both W&B and Tensorboard are enabled then we will prioritize W&B.

## ReporterV2

ReporterV2 can be enabled with `metrics.enable_reporterv2`. TorchTitan sends
metrics only; checkpoint upload is intentionally left to TorchTitan's DCP
checkpoint manager.

```
MODULE=worldmodel CONFIG=worldmodel_debugmodel ./run_train.sh \
  --metrics.enable_reporterv2 \
  --metrics.reporterv2_training_id my-run-id \
  --metrics.reporterv2_host http://data-gen.comma.life:3080/reporterv2
```

If `metrics.reporterv2_training_id` is empty, TorchTitan uses
`REPORTERV2_TRAINING_ID` or generates a UUID.
