# Fashion MNIST

## Dataset:

+ Dataset clones from [here](https://github.com/zalandoresearch/fashion-mnist)

+ It was provided by [Zalando](https://jobs.zalando.com/tech/)

## Model:

+ I implemented by python (with tensorflow framework) and used model from tensorflow tutorial.

+ Architecture:

```
Image 28x28x1
     |
     v
  [5x5x32]
     |
max pool 2x2
     |
     v
  [5x5x64]
     |
max pool 2x2
     |
     v
 [FC 1024]
     |
dropout 0.50
     |
     v
  [FC 10]
     |
     v
  softmax
```

## Experiment and Result:

+ You can train this model:

```bash
python FashionMnist.py
```

+ The accuracy I achieved on test set is 92.39% 

## Author

+ Thanh-Dat Truong

+ Email: thanhdat01234@gmail.com
