# Python深度学习



## 文件结构

|文件夹名   |说明                         |
|:--        |:--                          |
|ch02       |第2章使用的源代码            |
|...        |...                          |
|ch08       |第8章使用的源代码            |
|ch09       |第9章使用的源代码            |
|common     |共同使用的源代码             |
|dataset    |数据集用的源代码             |


源代码的解释请参考本书。

## 必要条件
执行源代码需要按照以下软件。

* Python 3.x
* NumPy
* Matplotlib
* keras

※Python的版本为Python 3。

## 执行方法

前进到各章节的文件夹，执行Python命令。

```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

## 声明

本源代码归属官方网站[官方源码](https://github.com/fchollet/deep-learning-with-python-notebooks)。
此项目仅为对官方代码的整理，且仅用个人学习，如有侵权，请告知，谢谢。

## 注意事项
### 1. 第二章

下载mnist数据集时，若无法连接，且报如下错误,则手动下载，然后修改程序：

> Exception: URL fetch failure on https://s3.amazonaws.com/img-datasets/mnist.npz : None -- [Errno 104] Connection reset by peer
 
手动下载网站：

https://storage.googleapis.com/cvdf-datasets/mnist/

程序修改：

```python
# 内置load_data() 多次加载数据都是失败 于是下载数据后 自定义方法
def load_data(path="MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
```
