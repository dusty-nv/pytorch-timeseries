# pytorch-timeseries

Time-series forecasting and prediction on tabular data using PyTorch.  Supports Jetson Nano, TX1/TX2, AGX Xavier, and Xavier NX.

#### Starting the Container

``` bash
$ git clone https://github.com/dusty-nv/pytorch-timeseries
$ cd pytorch-timeseries
$ docker/run.sh
$ cd pytorch-timeseries
```

#### Weather Forecasting

``` bash
$ python3 train.py --data data/weather.csv --inputs temperature --outputs temperature --horizon 1

train RMSE: [1.7638184641368702]
train R2:   [0.9768890819461414]
val RMSE:   [1.788160646675811]
val R2:     [0.9815134518452414]
```

![Weather Forecasting](data/weather.jpg)

#### Solar Power Prediction

``` bash
$ python3 train.py --data data/solar_power.csv --inputs AMBIENT_TEMPERATURE,IRRADIATION --outputs DC_POWER,AC_POWER

train RMSE: [6912.877912624019, 660.7508599953275]
train R2:   [0.9940444439752174, 0.9942998898150646]
val RMSE:   [4932.04556598682, 482.82950183757634]
val R2:     [0.9963377448275983, 0.9963286618121212]
```

![Solar Power Prediction](data/solar_power.jpg)
