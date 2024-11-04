# NeMo-Run entrypoint

`main.py` acts as a simple entrypoint to pretraining geneformer via use of configs and factories (`factories.py`). The command below will  execute the equivalent of what we have under scripts/pretrain.py

This module is a work in progress.

``` bash
python sub-packages/bionemo-geneformer/src/bionemo/geneformer/run/main.py \
	geneformer_config=basic_geneformer_config_recipe \
	data_config=small_data_config \
	parallel_config=simple_parallel_recipe \
	training_config=default_trainer_config \
	optim_config=default_adam_optimizer_with_cosine_annealing_recipe \
	experiment_config=experiment_config_recipe \
	resume_if_exists=False \
	data_config.seq_length=128 \
	parallel_config.num_devices=1 \
	data_config.micro_batch_size=2 \
	training_config.precision=bf16-mixed
```

## Concepts and things to keep in mind

Plain Function - A function that does literally anything and produces something else. In somecases, we have functions that take configs and produce an object. In these scenarios we are often composing an object with pieces of various configs.
Factory - A method that constructs a config and is decorated with run.cli.factory. These act as configs presentable to the command line.
Recipe - A specific factory with a distinct purpose. E.g. BERT XL vs BERT Small
Config - A fiddle dataclass presentable and mutatable via nemo run. These are also serialized and used for restoring previous configuations.
Entrypoint - A method that takes a mixture of plain arguments and configs. These are exposed to the command line. The body of the function represents the execution occuring in the program.
