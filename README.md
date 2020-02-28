# CSCI-1850-Midterm-Project

## Model

### Current

We are currently using an ensemble of fully-convolutional models with about 8 layers each. The convolutional blocks are based off those used by Codevilla et al (2018).

### History

In order from original experiments to current model:

1. DeepChrome, but adapted to produce a single real value
2. (1) with added convolutional and linear layers (1-6 and 1-3, respectively)
3. Codevilla et al (2018)'s vision module, simply adapted to this task but changing the output to a single real value
4. (3) with different numbers of convolutional and linear layers (1-6 and 1-3, respectively) and kernel sizes, strides, etc [Note: this covers dozens of experiments of different (combinations of) specific values]
5. Current model


## References

Codevilla, F., Miiller, M., López, A., Koltun, V., & Dosovitskiy, A. (2018, May). End-to-end driving via conditional imitation learning. In 2018 IEEE International Conference on Robotics and Automation (ICRA) (pp. 1-9). IEEE.
