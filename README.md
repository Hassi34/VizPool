
# OneFlow

It is a Package having the Implementation of ANN with callbacks

## 🔗 Project Link
Check out the Pypi Package [here](https://pypi.org/project/OneFlow-Hassi34/)
## Run Locally

Create two files in your working directory:
* config.yaml
* training.py

## config.yaml
```yaml
params:
  epochs : 3
  batch_size : 32
  num_classes : 10
  input_shape : [28, 28]
  loss_function : sparse_categorical_crossentropy
  metrics : accuracy
  optimizer : SGD
  validation_datasize : 5000
  es_patience : 5

artifacts:
  artifacts_dir : artifacts
  model_dir : model
  plots_dir : plots
  model_name : model.h5
  plot_name : results_plot.png
  model_ckpt_dir : ModelCheckpoints
  callbacked_model_name : model_ckpt.h5

logs:
  logs_dir : logs_dir
  general_logs : general_logs
  tensorboard_root_log_dir : tensorboard_logs


```

## training.py
```python 
from OneFlow.utils.common import read_config
from OneFlow.utils.data_mgmt import get_data
from OneFlow.utils.model import StepFlow
import argparse, os 

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    #This "get_data" function is loading the mnist dataset, bring your own and divide into categories to perform the custom training
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    sp = StepFlow(config, X_train, y_train, X_valid, y_valid)
    sp.create_model()
    sp.fit_model()
    sp.save_final_model()
    sp.save_plot()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path = parsed_args.config)

```
Then run the following commands on the termial 
```bash
pip install OneFlow-Hassi34
python training.py
```
##### On completion of training, run the following command on termial and observe the metrics on tensorboard 

```bash
tensorboard --logdir=logs_dir/tensorboard_logs/

```


