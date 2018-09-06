# Motion prediction with Hierarchical Motion Recurrent Network

## Introduction
This work concerns motion prediction of articulate objects such as human, fish and mice. Given a sequence of historical skeletal joints locations, we model the dynamics of the trajectory as kinematic chains of SE(3) group actions, parametrized by se(3) Lie algebra parameters. A sequence to sequence model employing our novel Hierarchical Motion Recurrent (HMR) Network as the decoder is employed to learn the temporal context of input pose sequences so as to predict future motion. 

Instead of adopting the conventional Euclidean L2 loss function for the 3D coordinates, we propose a geodesic regression loss layer on the SE(3) manifold which provides the following advantages.

- The SE(3) representation respects the anatomical and kinematic constraints of the skeletal model, i.e. bone lengths and physical degrees of freedom at the joints.
- Spatial relations underlying the joints are fully captured.
- Subtleties of temporal dynamics are better modelled in the manifold space than Euclidean space due to the absence of redundancy and constraints in the Lie algebra parameterization.


## Train and Predict
The main file is found in [motion_prediction.py](./src/motion_prediction.py).
<br/>
To train and predict on default setting, execute the following with python 3.
```
python motion_prediction.py
```

|FLAGS|  Default value|  Possible values| Remarks|
| ---        | ---       | ---   | ---|
| dataset | `--dataset Human` | Human, Fish, Mouse| |
| datatype   | `--datatype lie` | lie, xyz |  |
| action    | `--action all` | all, actions listed [below](#action_list) | |
| training      | `--training=1` | 0, 1 | |
| visualize      | `--visualize=1` | 0, 1 ||
| longterm      | `--longterm=0` | 0, 1 | Super long-term prediction exceeding 60s. <br/> dataset: Human <br/> action: walking, eating or smoking. |


To train and predict for different settings, simply set different value for the flags. An example of long term prediction for walking on the Human dataset is given below.
```
python motion_prediction.py --action walking --longterm=1
```

#### <a id="action_list"></a>Possible actions for Human 3.6m
```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

The configuration file is found in [training_config.py](./src/training_config.py). There are choices of different LSTM architectures as well as different loss functions that can be chosen in the configuration.


## Checkpoint and Output
checkpoints are saved in:
```
./checkpoint/dataset[Human, Fish, Mouse]/datatype[lie, xyz]_model(_recurrent-steps_context-window_hidden-size)_loss/action/inputWindow_outputWindow
```
outputs are saved in:
```
./output/dataset[Human, Fish, Mouse]/datatype[lie, xyz]_model_(_recurrent-steps_context-window_hidden-size)_loss/action/inputWindow_outputWindow
```
*[ ] denotes possible arguments and ( ) is specific for our HMR model
