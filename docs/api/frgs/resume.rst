.. code-block:: text

    usage: frgs resume [-h] [RESUME OPTIONS] PATH

    Resume reconstructing a 3D Gaussian Splat radiance field from a checkpoint file or run directory.
    When given a run directory, this command loads its latest completed checkpoint and appends new
    output to the same directory. The dataset used to create the checkpoint must be at the same path
    as when the checkpoint was created.

    Example usage:

        # Resume reconstruction from a checkpoint and save the final model to out_resumed.ply
        frgs resume checkpoint.pt -o out_resumed.ply

        # Resume the latest completed checkpoint in an existing run and append new output to that run
        frgs resume frgs_logs/my_run

    ╭─ positional arguments ───────────────────────────────────────────────────────────────────────╮
    │ PATH                                                                                         │
    │     Path to a checkpoint file or a run directory containing checkpoints. A run directory    │
    │     resumes its latest completed checkpoint and receives the resumed job's output.           │
    │     (required)                                                                               │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help                                                                                   │
    │     show this help message and exit                                                          │
    │ -n {None}|STR, --run-name {None}|STR                                                         │
    │     Name of the new output run when resuming from a checkpoint file. When resuming a run     │
    │     directory, its existing name and location are used instead. (default: None)              │
    │ -uv FLOAT, --update-viz-every FLOAT                                                          │
    │     How frequently (in epochs) to update the viewer during reconstruction. An epoch is one   │
    │     full pass through the dataset. If -1, do not visualize. (default: -1.0)                  │
    │ -p INT, --viewer-port INT                                                                    │
    │     The port to expose the viewer server on if update_viz_every > 0. (default: 8080)         │
    │ -ip STR, --viewer-ip-address STR                                                             │
    │     The IP address to expose the viewer server on if update_viz_every > 0. (default:         │
    │     127.0.0.1)                                                                               │
    │ -d STR|DEVICE, --device STR|DEVICE                                                           │
    │     Which device to use for reconstruction. Must be a cuda device. You can pass in a         │
    │     specific device index via cuda:N where N is the device index, or "cuda" to use the       │
    │     default cuda device. CPU is not supported. Default is "cuda". (default: cuda)            │
    │ -v, --verbose, --no-verbose                                                                  │
    │     If set, show verbose debug messages. (default: False)                                    │
    │ -o {None}|PATH, --out-path {None}|PATH                                                       │
    │     Optional path to save the final model. Path must end in .ply, .usdc, or .usdz. If        │
    │     omitted, the final model is not exported. (default: None)                                │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ io options ─────────────────────────────────────────────────────────────────────────────────╮
    │ Configure saving and logging metrics, images, and checkpoints.                               │
    │ ──────────────────────────────────────────────────────────────────────────────────────────── │
    │ --io.save-images, --io.no-save-images                                                        │
    │     Whether to save images to disk. If ``False``, images will not be saved to disk.          │
    │                                                                                              │
    │                                                                                              │
    │     Default is ``False``. (default: False)                                                   │
    │ --io.save-checkpoints, --io.no-save-checkpoints                                              │
    │     Whether to save checkpoints to disk. If ``False``, checkpoints will not be saved to      │
    │     disk.                                                                                    │
    │                                                                                              │
    │                                                                                              │
    │     Default is ``True``. (default: True)                                                     │
    │ --io.save-plys, --io.no-save-plys                                                            │
    │     Whether to save PLY files to disk. If ``False``, PLY files will not be saved to disk.    │
    │                                                                                              │
    │                                                                                              │
    │     Default is ``True``. (default: True)                                                     │
    │ --io.save-metrics, --io.no-save-metrics                                                      │
    │     Whether to save metrics to a CSV file. If ``False``, metrics will not be saved to a CSV  │
    │     file.                                                                                    │
    │                                                                                              │
    │                                                                                              │
    │     Default is ``True``. (default: True)                                                     │
    │ --io.metrics-file-buffer-size INT                                                            │
    │     How much buffering (in bytes) to use for metrics file logging. Larger values can improve │
    │     performance when logging many metrics.                                                   │
    │                                                                                              │
    │                                                                                              │
    │     Default is 8 MiB. (default: 8388608)                                                     │
    │ --io.use-tensorboard, --io.no-use-tensorboard                                                │
    │     Whether to use TensorBoard for logging metrics and images. If ``True``, metrics and      │
    │     images will be logged to TensorBoard.                                                    │
    │                                                                                              │
    │                                                                                              │
    │     Default is ``False``. (default: False)                                                   │
    │ --io.save-images-to-tensorboard, --io.no-save-images-to-tensorboard                          │
    │     Whether to also save images to TensorBoard if :obj:`use_tensorboard` is ``True``. If     │
    │     ``True``, images will be saved to TensorBoard.                                           │
    │                                                                                              │
    │                                                                                              │
    │     Default is ``False``. (default: False)                                                   │
    │ --io.log-path {None}|PATH                                                                    │
    │     Path to save logs, checkpoints, and other output to. Defaults to `frgs_logs` in the      │
    │     current working directory. (default: frgs_logs)                                          │
    │ --io.log-every INT                                                                           │
    │     How frequently to log metrics during reconstruction. (default: 10)                       │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
