## 下载数据

在终端中输入

```shell
python ./data.py
```

数据会保存在 `./data_file` 文件夹中




## 训练
一键训练（同时训练5个模型）
```shell
./start_train.sh
```

或手动在终端中输入：

```shell
mkdir nohup
nohup python -u ./trainer.py -m 'a2c' -tts 200000 >./nohup/A2C.log 2>&1 &
```

**TIPS**：

* 运行日志保存在 `./nohup` 文件夹中
* 运行完成后的模型保存在 `./train_file` 文件夹中


## 回测
一键回测
```shell
./start_trade.sh
```
或单个测试，在终端中输入

```shell
python -u ./trader.py -m 'a2c'
```

**TIPS**：

* 若未找到 `.model` 文件，可以在 [release](https://github.com/sunnyswag/RL_in_Stock/releases/) 中进行下载
* 回测数据保存在 `./trade_file` 文件夹中

