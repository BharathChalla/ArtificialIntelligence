<style>
{
  box-sizing: border-box;
}

.column {
  float: left;
  width: 33.33%;
  padding: 5px;
  background-color:white;
}

.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

# ArtificialIntelligence

## 101 - Machine Learning Basics

### Boston Housing Dataset

#### Heatmap

<p>
  <img src="./docs/images/MLBasics/1_BostonDatasetHeatmap.png" alt="Boston Housing Dataset" width="400" height="300" hspace="20"/>
</p>

### Linear Regression

|   LinearRegression   | DatasetType |  RMSE   |    R2    |
|:--------------------:|:-----------:|:-------:|:--------:|
| sklearn.linear_model |    Train    | 5.29127 | 0.649317 |
| sklearn.linear_model |    Test     | 6.04104 | 0.616202 |
|     Implemented      |    Train    | 5.29128 | 0.649316 |
|     Implemented      |    Test     | 6.03568 | 0.616883 |

### Titanic Dataset

#### Heatmap of each feature

<p style="background-color:white; display: inline-block;">
  <img src="./docs/images/MLBasics/3_TitanicDatasetHeatmap.png" alt="Logistic Regression" width="480" height="300" hspace="40"/>
</p>

#### P-Class Boxplot for Age Missing Data

<p>
  <img src="./docs/images/MLBasics/4_TitanicDatasetPClassAge.png" alt="Logistic Regression" width="400" height="300" hspace="20"/>
</p>

### Logistic Regression

|  LogisticRegression  | DatasetType | Accuracy | Precision | Recall | F1-Score |
|:--------------------:|:-----------:|:--------:|:---------:|:------:|:--------:|
| sklearn.linear_model |    Train    |  0.802   |   0.771   | 0.677  |  0.721   |
| sklearn.linear_model |    Test     |  0.809   |   0.863   | 0.620  |  0.721   |
|     Implemented      |    Train    |  0.812   |   0.773   | 0.710  |  0.740   |
|     Implemented      |    Test     |  0.815   |   0.839   | 0.662  |  0.740   |

## 102 - Neural Networks and Deep Learning

### CNN - Vertical Filter for Edge Detection

<p style="background-color:white; display: inline-block;">
  <img  src="./docs/images/NN/1.VerticalFilter.png" alt="Vertical Edge Detection" width="600" height="300" hspace="20"/>
</p>

### CNN - Horizontal Filter for Edge Detection

<p style="background-color:white; display: inline-block;">
  <img src="./docs/images/NN/2.HorizontalFilter.png" alt="Horizontal Edge Detection" width="600" height="300" hspace="20"/>
</p>

### simpleCNN - Fashion MNIST Dataset

<div class="row">
  <div class="column">
  <img src="./docs/images/NN/3.1LossVsEpoch.png" alt="Loss Vs Epoch" width="400" height="300">
  </div>
  <div class="column">
  <img src="./docs/images/NN/3.2.Predictions.png" alt="Predictions" width="400" height="300">
  </div>
</div>

### LeNet-5 - Fashion MNIST Dataset

<div class="row">
  <div class="column">
  <img src="./docs/images/NN/4.1LossVsEpoch.png" alt="Loss Vs Epoch" width="400" height="300">
  </div>
  <div class="column">
  <img src="./docs/images/NN/4.2.Predictions.png" alt="Predictions" width="400" height="300">
  </div>
</div>

## 103 - Cliff Walking

### Informed Search

|     |                                  Trajectory                                  |                                  Path                                  |                           Order Changed Trajectory                           | Order Changed Path                                                     |
|-----|:----------------------------------------------------------------------------:|:----------------------------------------------------------------------:|:----------------------------------------------------------------------------:|------------------------------------------------------------------------|
| DFS | <img width="1604" src="./docs/images/UnInformedSearch/1_1DFSTrajectory.png"> | <img width="1604" src="./docs/images/UnInformedSearch/1_1DFSPath.png"> | <img width="1604" src="./docs/images/UnInformedSearch/1_2DFSTrajectory.png"> | <img width="1604" src="./docs/images/UnInformedSearch/1_2DFSPath.png"> |
| BFS | <img width="1604" src="./docs/images/UnInformedSearch/2_1BFSTrajectory.png"> | <img width="1604" src="./docs/images/UnInformedSearch/2_1BFSPath.png"> | <img width="1604" src="./docs/images/UnInformedSearch/2_2BFSTrajectory.png"> | <img width="1604" src="./docs/images/UnInformedSearch/2_2BFSPath.png"> |
| UCS | <img width="1604" src="./docs/images/UnInformedSearch/3_1UCSTrajectory.png"> | <img width="1604" src="./docs/images/UnInformedSearch/3_1UCSPath.png"> | <img width="1604" src="./docs/images/UnInformedSearch/3_2UCSTrajectory.png"> | <img width="1604" src="./docs/images/UnInformedSearch/3_2UCSPath.png"> |

### Uninformed Search

|                  |                                       Trajectory                                       |                                       Path                                       |                                Order Changed Trajectory                                | Order Changed Path                                                               |
|------------------|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------|
| Greedy Manhattan | <img width="1604" src="./docs/images/InformedSearch/4_1GreedyManhattanTrajectory.png"> | <img width="1604" src="./docs/images/InformedSearch/4_1GreedyManhattanPath.png"> | <img width="1604" src="./docs/images/InformedSearch/4_2GreedyManhattanTrajectory.png"> | <img width="1604" src="./docs/images/InformedSearch/4_2GreedyManhattanPath.png"> |
| Greedy Euclidean | <img width="1604" src="./docs/images/InformedSearch/5_1GreedyEuclideanTrajectory.png"> | <img width="1604" src="./docs/images/InformedSearch/5_1GreedyEuclideanPath.png"> | <img width="1604" src="./docs/images/InformedSearch/5_2GreedyEuclideanTrajectory.png"> | <img width="1604" src="./docs/images/InformedSearch/5_2GreedyEuclideanPath.png"> |
| A* Manhattan     | <img width="1604" src="./docs/images/InformedSearch/6_1AStarManhattanTrajectory.png">  | <img width="1604" src="./docs/images/InformedSearch/6_1AStarManhattanPath.png">  | <img width="1604" src="./docs/images/InformedSearch/6_2AStarManhattanTrajectory.png">  | <img width="1604" src="./docs/images/InformedSearch/6_2AStarManhattanPath.png">  |
| A* Euclidean     | <img width="1604" src="./docs/images/InformedSearch/7_1AStarEuclideanTrajectory.png">  | <img width="1604" src="./docs/images/InformedSearch/7_1AStarEuclideanPath.png">  | <img width="1604" src="./docs/images/InformedSearch/7_2AStarEuclideanTrajectory.png">  | <img width="1604" src="./docs/images/InformedSearch/7_2AStarEuclideanPath.png">  |

## 104 - Minimax Search

### <center>Without Alpha-Beta Pruning</center>
<p align="center">
  <img src="./docs/images/MinMaxSearch/MinimaxSearchMethodDepth6.gif" alt="Mini-max Search"/>
</p>

### <center>With Alpha-Beta Pruning</center>

<p align="center">
  <img src="./docs/images/MinMaxSearch/MinimaxSearchMethodAlphaBetaPruningDepth6.gif" alt="Alpha-Beta Pruning"/>
</p>

## 105 - Reinforcement Learning

### 105.1 - Deep Q Learning - Value Iteration

```text
Initial State:
'🙂'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'😎'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'🙂'

Move #: 0; Taking action: Right
'🙂'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'😎'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'🙂'

Move #: 1; Taking action: Right
'🙂'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'😎'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'🙂'

Move #: 2; Taking action: Down
'🙂'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'😎'
'()'	'()'	'()'	'()'	'🙂'

Move #: 3; Taking action: Down
'🙂'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'()'
'()'	'()'	'()'	'()'	'😎'

Game won! Reward: 50
```

### 105.2 - Actor Critic

## 106 - Twitter Sentiment Analysis Using RNN

### Twitter Dataset - Tweet Length Distribution

<p style="background-color:white; display: inline-block;">
  <img src="./docs/images/TwitterSentimentAnalysis/TwitterDataDistribution.png" alt="Tweet Length Distribution" width="400" height="300" hspace="20"/>
</p>

### LSTM

```text
SentimentLSTM(
  (embedding): Embedding(850173, 400)
  (lstm): LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=256, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=1, bias=True)
  (sig): Sigmoid()
)
```

```text
Twitter 1,600,000 Data Accuracies: 
Train loss: 0.288
Train accuracy: 0.880

Valid loss: 0.509
Valid accuracy: 0.763

Test loss: 0.537
Test accuracy: 0.752
```

### GRU

```text
SentimentGRU(
  (embedding): Embedding(850173, 400)
  (gru): GRU(400, 256, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=256, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=1, bias=True)
  (sig): Sigmoid()
)
```

```text
Twitter 1,600,000 Data Accuracies: 
Train loss: 0.367
Train accuracy: 0.838

Valid loss: 0.522
Valid accuracy: 0.737

Test loss: 0.539
Test accuracy: 0.727
```

# Projects

## Image Classification Deep Convolutional Neural Networks

## Reinforcement Learning for Autonomous Driving (In - Progress)
