# Hyperparameters Tuning!



## Learning rates trying

### 10-2, 10-4

configs [{'learning_rate': 0.0013283378516632724, 'reg': 9.418257919247722e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>}, {'learning_rate': 0.00018061357439124492, 'reg': 9.445375342257156e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>}, {'learning_rate': 0.00702433368465948, 'reg': 5.2857048289304723e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>}] Evaluating Config #1 [of 3]: {'learning_rate': 0.0013283378516632724, 'reg': 9.418257919247722e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>} (Epoch 1 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 2.289031; val loss: 2.170512 (Epoch 2 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.922918; val loss: 2.189783 (Epoch 3 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.647401; val loss: 2.444251 (Epoch 4 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.344891; val loss: 2.905024 (Epoch 5 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.170809; val loss: 2.388317 (Epoch 6 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 0.898617; val loss: 2.867217 Stopping early at epoch 5! 

Evaluating Config #2 [of 3]: {'learning_rate': 0.00018061357439124492, 'reg': 9.445375342257156e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>} (Epoch 1 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 2.234415; val loss: 2.048425 (Epoch 2 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.986454; val loss: 1.894749 (Epoch 3 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.796895; val loss: 1.817914 (Epoch 4 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.671746; val loss: 1.825843 (Epoch 5 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.566297; val loss: 1.786321 (Epoch 6 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.450470; val loss: 1.797149 (Epoch 7 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.347156; val loss: 1.778806 (Epoch 8 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.246980; val loss: 1.819089 (Epoch 9 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.132164; val loss: 1.799144 (Epoch 10 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 1.020089; val loss: 1.803812 (Epoch 11 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 0.907348; val loss: 1.843339 (Epoch 12 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 0.808279; val loss: 1.944877 Stopping early at epoch 11! 

Evaluating Config #3 [of 3]: {'learning_rate': 0.00702433368465948, 'reg': 5.2857048289304723e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>} (Epoch 1 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 7.917622; val loss: 12.563828 (Epoch 2 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 11.583004; val loss: 17.698546 (Epoch 3 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 12.813666; val loss: 10.317717 (Epoch 4 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 12.181763; val loss: 12.814377 (Epoch 5 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 11.531178; val loss: 15.005241 (Epoch 6 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 10.645777; val loss: 18.902742 (Epoch 7 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 11.508750; val loss: 25.885581 (Epoch 8 [/](https://file+.vscode-resource.vscode-cdn.net/) 20) train loss: 9.759771; val loss: 24.979896 Stopping early at epoch 7! 

Search done. Best Val Loss = 1.778805979042064 Best Config: {'learning_rate': 0.00018061357439124492, 'reg': 9.445375342257156e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>}



### 10^-4, 10^-6

configs [{'learning_rate': 6.166700442253193e-05, 'reg': 8.302268328367997e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>}, {'learning_rate': 6.502919062138012e-05, 'reg': 6.475731545228225e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>}, {'learning_rate': 5.153936424620212e-06, 'reg': 7.72561681873255e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>}]

Evaluating Config #1 [of 3]:
 {'learning_rate': 6.166700442253193e-05, 'reg': 8.302268328367997e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>}
(Epoch 1 / 20) train loss: 2.279985; val loss: 2.204761
(Epoch 2 / 20) train loss: 2.163193; val loss: 2.076356
(Epoch 3 / 20) train loss: 2.052343; val loss: 1.999710
(Epoch 4 / 20) train loss: 1.969785; val loss: 1.954029
(Epoch 5 / 20) train loss: 1.894684; val loss: 1.923583
(Epoch 6 / 20) train loss: 1.829562; val loss: 1.892199
(Epoch 7 / 20) train loss: 1.774319; val loss: 1.873372
(Epoch 8 / 20) train loss: 1.720315; val loss: 1.855222
(Epoch 9 / 20) train loss: 1.675299; val loss: 1.833402
(Epoch 10 / 20) train loss: 1.632378; val loss: 1.842839
(Epoch 11 / 20) train loss: 1.585267; val loss: 1.834816
(Epoch 12 / 20) train loss: 1.545779; val loss: 1.828434
(Epoch 13 / 20) train loss: 1.505556; val loss: 1.835312
(Epoch 14 / 20) train loss: 1.463031; val loss: 1.828312
(Epoch 15 / 20) train loss: 1.421283; val loss: 1.840282
(Epoch 16 / 20) train loss: 1.387303; val loss: 1.835332
(Epoch 17 / 20) train loss: 1.346345; val loss: 1.845792
(Epoch 18 / 20) train loss: 1.304005; val loss: 1.815029
(Epoch 19 / 20) train loss: 1.266442; val loss: 1.834683
(Epoch 20 / 20) train loss: 1.226893; val loss: 1.818096

Evaluating Config #2 [of 3]:
 {'learning_rate': 6.502919062138012e-05, 'reg': 6.475731545228225e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>}
(Epoch 1 / 20) train loss: 2.277220; val loss: 2.196688
(Epoch 2 / 20) train loss: 2.160504; val loss: 2.073082
(Epoch 3 / 20) train loss: 2.049960; val loss: 2.005851
(Epoch 4 / 20) train loss: 1.961018; val loss: 1.954504
(Epoch 5 / 20) train loss: 1.882837; val loss: 1.908850
(Epoch 6 / 20) train loss: 1.813688; val loss: 1.887495
(Epoch 7 / 20) train loss: 1.753885; val loss: 1.859745
(Epoch 8 / 20) train loss: 1.702408; val loss: 1.848008
(Epoch 9 / 20) train loss: 1.650685; val loss: 1.845288
(Epoch 10 / 20) train loss: 1.604532; val loss: 1.834866
(Epoch 11 / 20) train loss: 1.560985; val loss: 1.825337
(Epoch 12 / 20) train loss: 1.518519; val loss: 1.814473
(Epoch 13 / 20) train loss: 1.477606; val loss: 1.827602
(Epoch 14 / 20) train loss: 1.436268; val loss: 1.838694
(Epoch 15 / 20) train loss: 1.396283; val loss: 1.810033
(Epoch 16 / 20) train loss: 1.355389; val loss: 1.801817
(Epoch 17 / 20) train loss: 1.312883; val loss: 1.811383
(Epoch 18 / 20) train loss: 1.277045; val loss: 1.807370
(Epoch 19 / 20) train loss: 1.237213; val loss: 1.806071
(Epoch 20 / 20) train loss: 1.194048; val loss: 1.826513

Evaluating Config #3 [of 3]:
 {'learning_rate': 5.153936424620212e-06, 'reg': 7.72561681873255e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.Relu'>}
(Epoch 1 / 20) train loss: 2.302186; val loss: 2.301390
(Epoch 2 / 20) train loss: 2.300417; val loss: 2.299081
(Epoch 3 / 20) train loss: 2.297187; val loss: 2.294684
(Epoch 4 / 20) train loss: 2.291746; val loss: 2.287709
(Epoch 5 / 20) train loss: 2.283806; val loss: 2.277652
(Epoch 6 / 20) train loss: 2.273303; val loss: 2.265805
(Epoch 7 / 20) train loss: 2.261009; val loss: 2.252200
(Epoch 8 / 20) train loss: 2.247541; val loss: 2.237347
(Epoch 9 / 20) train loss: 2.233374; val loss: 2.222465
(Epoch 10 / 20) train loss: 2.219046; val loss: 2.207714
(Epoch 11 / 20) train loss: 2.205073; val loss: 2.192754
(Epoch 12 / 20) train loss: 2.191362; val loss: 2.178989
(Epoch 13 / 20) train loss: 2.178422; val loss: 2.166119
(Epoch 14 / 20) train loss: 2.165974; val loss: 2.153242
(Epoch 15 / 20) train loss: 2.154095; val loss: 2.141864
(Epoch 16 / 20) train loss: 2.142768; val loss: 2.131310
(Epoch 17 / 20) train loss: 2.131717; val loss: 2.120602
(Epoch 18 / 20) train loss: 2.121482; val loss: 2.111219
(Epoch 19 / 20) train loss: 2.111393; val loss: 2.101972
(Epoch 20 / 20) train loss: 2.101852; val loss: 2.093769

Search done. Best Val Loss = 1.8018168001945982
Best Config: {'learning_rate': 6.502919062138012e-05, 'reg': 6.475731545228225e-05, 'loss_func': <class 'exercise_code.networks.loss.CrossEntropyFromLogits'>, 'activation': <class 'exercise_code.networks.layer.LeakyRelu'>}

### 1e-3, 1e-4

configs [{'learning_rate': 0.001, 'reg': 0.0001}, {'learning_rate': 0.0008, 'reg': 0.0001}, {'learning_rate': 0.0006, 'reg': 0.0001}, {'learning_rate': 0.0004, 'reg': 0.0001}, {'learning_rate': 0.0003, 'reg': 0.0001}, {'learning_rate': 0.0001, 'reg': 0.0001}]

Evaluating Config #1 [of 6]:
 {'learning_rate': 0.001, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.258635; val loss: 2.254781
(Epoch 2 / 10) train loss: 2.144519; val loss: 2.077584
(Epoch 3 / 10) train loss: 2.029698; val loss: 2.038831
(Epoch 4 / 10) train loss: 1.927143; val loss: 2.021878
(Epoch 5 / 10) train loss: 1.823135; val loss: 1.985960
(Epoch 6 / 10) train loss: 1.745445; val loss: 1.980391
(Epoch 7 / 10) train loss: 1.659959; val loss: 1.938142
(Epoch 8 / 10) train loss: 1.577686; val loss: 1.927447
(Epoch 9 / 10) train loss: 1.490765; val loss: 2.016346
(Epoch 10 / 10) train loss: 1.409498; val loss: 1.943844

Evaluating Config #2 [of 6]:
 {'learning_rate': 0.0008, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.262940; val loss: 2.132110
(Epoch 2 / 10) train loss: 2.096939; val loss: 2.034092
(Epoch 3 / 10) train loss: 1.966678; val loss: 1.981437
(Epoch 4 / 10) train loss: 1.886873; val loss: 1.989534
(Epoch 5 / 10) train loss: 1.789326; val loss: 1.985921
(Epoch 6 / 10) train loss: 1.687549; val loss: 1.983241
(Epoch 7 / 10) train loss: 1.607593; val loss: 1.965545
(Epoch 8 / 10) train loss: 1.516750; val loss: 1.988777
(Epoch 9 / 10) train loss: 1.426844; val loss: 1.941927
(Epoch 10 / 10) train loss: 1.346204; val loss: 2.014797

Evaluating Config #3 [of 6]:
 {'learning_rate': 0.0006, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.270131; val loss: 2.144347
(Epoch 2 / 10) train loss: 2.116169; val loss: 2.055218
(Epoch 3 / 10) train loss: 2.013198; val loss: 1.973268
(Epoch 4 / 10) train loss: 1.905779; val loss: 1.942416
(Epoch 5 / 10) train loss: 1.822876; val loss: 1.968616
(Epoch 6 / 10) train loss: 1.733921; val loss: 1.938160
(Epoch 7 / 10) train loss: 1.646635; val loss: 1.931265
(Epoch 8 / 10) train loss: 1.551812; val loss: 1.936816
(Epoch 9 / 10) train loss: 1.462517; val loss: 1.953984
(Epoch 10 / 10) train loss: 1.389366; val loss: 1.995370

Evaluating Config #4 [of 6]:
 {'learning_rate': 0.0004, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.277196; val loss: 2.194002
(Epoch 2 / 10) train loss: 2.141943; val loss: 2.063945
(Epoch 3 / 10) train loss: 2.034024; val loss: 2.011552
(Epoch 4 / 10) train loss: 1.948817; val loss: 1.974781
(Epoch 5 / 10) train loss: 1.868124; val loss: 1.956682
(Epoch 6 / 10) train loss: 1.799162; val loss: 1.927152
(Epoch 7 / 10) train loss: 1.725254; val loss: 1.936445
(Epoch 8 / 10) train loss: 1.649275; val loss: 1.922822
(Epoch 9 / 10) train loss: 1.563670; val loss: 1.934443
(Epoch 10 / 10) train loss: 1.494239; val loss: 1.952743

Evaluating Config #5 [of 6]:
 {'learning_rate': 0.0003, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.280508; val loss: 2.235598
(Epoch 2 / 10) train loss: 2.207284; val loss: 2.197080
(Epoch 3 / 10) train loss: 2.145805; val loss: 2.154782
(Epoch 4 / 10) train loss: 2.086334; val loss: 2.118965
(Epoch 5 / 10) train loss: 2.017979; val loss: 2.080611
(Epoch 6 / 10) train loss: 1.952534; val loss: 2.036312
(Epoch 7 / 10) train loss: 1.878319; val loss: 2.004292
(Epoch 8 / 10) train loss: 1.804593; val loss: 2.006998
(Epoch 9 / 10) train loss: 1.732285; val loss: 1.975125
(Epoch 10 / 10) train loss: 1.658124; val loss: 1.945138

Evaluating Config #6 [of 6]:
 {'learning_rate': 0.0001, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.298974; val loss: 2.287991
(Epoch 2 / 10) train loss: 2.271514; val loss: 2.251652
(Epoch 3 / 10) train loss: 2.231249; val loss: 2.207744
(Epoch 4 / 10) train loss: 2.185946; val loss: 2.169235
(Epoch 5 / 10) train loss: 2.141694; val loss: 2.136016
(Epoch 6 / 10) train loss: 2.100937; val loss: 2.107592
(Epoch 7 / 10) train loss: 2.066464; val loss: 2.085820
(Epoch 8 / 10) train loss: 2.032077; val loss: 2.068340
(Epoch 9 / 10) train loss: 1.999963; val loss: 2.051930
(Epoch 10 / 10) train loss: 1.971963; val loss: 2.039167

Search done. Best Val Loss = 1.922822381004574
Best Config: {'learning_rate': 0.0004, 'reg': 0.0001}







### 1e-2, 1e-3 kharaa kek



### 1e-3, 1e-4 in details



configs [{'learning_rate': 0.001, 'reg': 0.0001}, {'learning_rate': 0.0009000000000000001, 'reg': 0.0001}, {'learning_rate': 0.0008, 'reg': 0.0001}, {'learning_rate': 0.0007, 'reg': 0.0001}, {'learning_rate': 0.0006, 'reg': 0.0001}, {'learning_rate': 0.0005, 'reg': 0.0001}, {'learning_rate': 0.0004, 'reg': 0.0001}, {'learning_rate': 0.0003, 'reg': 0.0001}, {'learning_rate': 0.0002, 'reg': 0.0001}, {'learning_rate': 0.0001, 'reg': 0.0001}, {'learning_rate': 0.0001, 'reg': 0.0001}]

Evaluating Config #1 [of 11]:
 {'learning_rate': 0.001, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.252424; val loss: 2.159874
(Epoch 2 / 10) train loss: 2.084063; val loss: 2.072884
(Epoch 3 / 10) train loss: 1.975869; val loss: 1.981688
(Epoch 4 / 10) train loss: 1.889912; val loss: 1.954843
(Epoch 5 / 10) train loss: 1.794012; val loss: 1.922796
(Epoch 6 / 10) train loss: 1.712632; val loss: 1.940923
(Epoch 7 / 10) train loss: 1.629459; val loss: 1.947068
(Epoch 8 / 10) train loss: 1.537327; val loss: 1.978289
(Epoch 9 / 10) train loss: 1.445518; val loss: 1.992559
(Epoch 10 / 10) train loss: 1.379684; val loss: 1.965534
Stopping early at epoch 9!

Evaluating Config #2 [of 11]:
 {'learning_rate': 0.0009000000000000001, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.241644; val loss: 2.143327
(Epoch 2 / 10) train loss: 2.062935; val loss: 2.074538
(Epoch 3 / 10) train loss: 1.964138; val loss: 2.008995
(Epoch 4 / 10) train loss: 1.873745; val loss: 1.963742
(Epoch 5 / 10) train loss: 1.787736; val loss: 1.973575
(Epoch 6 / 10) train loss: 1.715676; val loss: 1.967429
(Epoch 7 / 10) train loss: 1.629076; val loss: 1.978742
(Epoch 8 / 10) train loss: 1.545400; val loss: 1.980582
(Epoch 9 / 10) train loss: 1.479200; val loss: 1.957707
(Epoch 10 / 10) train loss: 1.384285; val loss: 1.996956

Evaluating Config #3 [of 11]:
 {'learning_rate': 0.0008, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.263759; val loss: 2.121140
(Epoch 2 / 10) train loss: 2.105570; val loss: 2.030326
(Epoch 3 / 10) train loss: 1.990765; val loss: 1.977666
(Epoch 4 / 10) train loss: 1.885602; val loss: 1.967401
(Epoch 5 / 10) train loss: 1.802320; val loss: 1.958637
(Epoch 6 / 10) train loss: 1.698844; val loss: 1.967931
(Epoch 7 / 10) train loss: 1.617930; val loss: 1.931127
(Epoch 8 / 10) train loss: 1.544705; val loss: 1.935426
(Epoch 9 / 10) train loss: 1.453458; val loss: 1.895817
(Epoch 10 / 10) train loss: 1.362025; val loss: 1.983038

Evaluating Config #4 [of 11]:
 {'learning_rate': 0.0007, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.252990; val loss: 2.137689
(Epoch 2 / 10) train loss: 2.071107; val loss: 2.016659
(Epoch 3 / 10) train loss: 1.953178; val loss: 2.043582
(Epoch 4 / 10) train loss: 1.869620; val loss: 1.954357
(Epoch 5 / 10) train loss: 1.793744; val loss: 1.959127
(Epoch 6 / 10) train loss: 1.703779; val loss: 1.959081
(Epoch 7 / 10) train loss: 1.624914; val loss: 1.981410
(Epoch 8 / 10) train loss: 1.535584; val loss: 1.968123
(Epoch 9 / 10) train loss: 1.439212; val loss: 1.991639
Stopping early at epoch 8!

Evaluating Config #5 [of 11]:
 {'learning_rate': 0.0006, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.265024; val loss: 2.171059
(Epoch 2 / 10) train loss: 2.116724; val loss: 2.097522
(Epoch 3 / 10) train loss: 2.024171; val loss: 2.019287
(Epoch 4 / 10) train loss: 1.930364; val loss: 2.026942
(Epoch 5 / 10) train loss: 1.851495; val loss: 1.988538
(Epoch 6 / 10) train loss: 1.775815; val loss: 1.991565
(Epoch 7 / 10) train loss: 1.706474; val loss: 1.997764
(Epoch 8 / 10) train loss: 1.633308; val loss: 1.985891
(Epoch 9 / 10) train loss: 1.542255; val loss: 1.987957
(Epoch 10 / 10) train loss: 1.463392; val loss: 1.974197

Evaluating Config #6 [of 11]:
 {'learning_rate': 0.0005, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.268854; val loss: 2.168680
(Epoch 2 / 10) train loss: 2.137661; val loss: 2.084071
(Epoch 3 / 10) train loss: 2.037131; val loss: 2.024747
(Epoch 4 / 10) train loss: 1.953076; val loss: 2.023979
(Epoch 5 / 10) train loss: 1.874642; val loss: 2.013461
(Epoch 6 / 10) train loss: 1.795402; val loss: 1.999612
(Epoch 7 / 10) train loss: 1.719040; val loss: 2.004757
(Epoch 8 / 10) train loss: 1.645159; val loss: 2.013525
(Epoch 9 / 10) train loss: 1.563996; val loss: 1.979130
(Epoch 10 / 10) train loss: 1.479158; val loss: 2.006402

Evaluating Config #7 [of 11]:
 {'learning_rate': 0.0004, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.271507; val loss: 2.236853
(Epoch 2 / 10) train loss: 2.206996; val loss: 2.195564
(Epoch 3 / 10) train loss: 2.146111; val loss: 2.143741
(Epoch 4 / 10) train loss: 2.079194; val loss: 2.120099
(Epoch 5 / 10) train loss: 2.017364; val loss: 2.076020
(Epoch 6 / 10) train loss: 1.941315; val loss: 2.035041
(Epoch 7 / 10) train loss: 1.872219; val loss: 2.008252
(Epoch 8 / 10) train loss: 1.791865; val loss: 2.027495
(Epoch 9 / 10) train loss: 1.724785; val loss: 1.966943
(Epoch 10 / 10) train loss: 1.651292; val loss: 1.981091

Evaluating Config #8 [of 11]:
 {'learning_rate': 0.0003, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.286429; val loss: 2.229019
(Epoch 2 / 10) train loss: 2.187090; val loss: 2.137069
(Epoch 3 / 10) train loss: 2.094486; val loss: 2.096258
(Epoch 4 / 10) train loss: 2.020581; val loss: 2.056543
(Epoch 5 / 10) train loss: 1.959376; val loss: 2.028114
(Epoch 6 / 10) train loss: 1.898603; val loss: 2.005448
(Epoch 7 / 10) train loss: 1.837019; val loss: 2.001304
(Epoch 8 / 10) train loss: 1.781285; val loss: 1.994672
(Epoch 9 / 10) train loss: 1.723208; val loss: 1.993724
(Epoch 10 / 10) train loss: 1.660821; val loss: 1.999667

Evaluating Config #9 [of 11]:
 {'learning_rate': 0.0002, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.293557; val loss: 2.255392
(Epoch 2 / 10) train loss: 2.233101; val loss: 2.183924
(Epoch 3 / 10) train loss: 2.166539; val loss: 2.142283
(Epoch 4 / 10) train loss: 2.101985; val loss: 2.099726
(Epoch 5 / 10) train loss: 2.045526; val loss: 2.065386
(Epoch 6 / 10) train loss: 1.991997; val loss: 2.041536
(Epoch 7 / 10) train loss: 1.940433; val loss: 2.033338
(Epoch 8 / 10) train loss: 1.895176; val loss: 2.007955
(Epoch 9 / 10) train loss: 1.843173; val loss: 1.998486
(Epoch 10 / 10) train loss: 1.800087; val loss: 2.002299

Evaluating Config #10 [of 11]:
 {'learning_rate': 0.0001, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.299248; val loss: 2.284842
(Epoch 2 / 10) train loss: 2.268406; val loss: 2.246669
(Epoch 3 / 10) train loss: 2.226272; val loss: 2.203249
(Epoch 4 / 10) train loss: 2.183438; val loss: 2.169925
(Epoch 5 / 10) train loss: 2.142792; val loss: 2.135563
(Epoch 6 / 10) train loss: 2.104335; val loss: 2.115475
(Epoch 7 / 10) train loss: 2.070720; val loss: 2.089107
(Epoch 8 / 10) train loss: 2.038222; val loss: 2.068673
(Epoch 9 / 10) train loss: 2.006537; val loss: 2.047602
(Epoch 10 / 10) train loss: 1.973389; val loss: 2.032825

Evaluating Config #11 [of 11]:
 {'learning_rate': 0.0001, 'reg': 0.0001}
(Epoch 1 / 10) train loss: 2.298748; val loss: 2.286573
(Epoch 2 / 10) train loss: 2.270357; val loss: 2.248200
(Epoch 3 / 10) train loss: 2.227048; val loss: 2.204737
(Epoch 4 / 10) train loss: 2.178580; val loss: 2.160186
(Epoch 5 / 10) train loss: 2.136311; val loss: 2.132115
(Epoch 6 / 10) train loss: 2.094274; val loss: 2.099207
(Epoch 7 / 10) train loss: 2.059832; val loss: 2.083422
(Epoch 8 / 10) train loss: 2.026464; val loss: 2.064770
(Epoch 9 / 10) train loss: 1.995155; val loss: 2.049450
(Epoch 10 / 10) train loss: 1.964129; val loss: 2.033320

Search done. Best Val Loss = 1.8958172550748407
Best Config: {'learning_rate': 0.0008, 'reg': 0.0001}





## focus more on 0.6*10e-3 and 10e-3, 10e-4 in general

---------------



## Regularization ! (1e-4, 1e-5)



----

## 

## Activation function ?

**Relu was the best**

----

