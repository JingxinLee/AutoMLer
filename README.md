# Non-Rule-Based Automated Machine Learning with Large Language Model

Agent:

1.数据预处理DataProcessor 

预处理Preprocess（Normalize,FeatureEngineering,Augumentator,DataLoader)

生成代码：RAG

2.构建模型Model：ModelConstructor（HuggingGPT）

数据,计算资源,用户条件模型选择

ModelSelector，Optimizer，Scheduler

3.Trainer/Evaluator

代码，根据输入生成

Evaluation Metric、LossFunction

4.HPO

HyperParameter 有哪些

HyperParameter Search Space

Search Method

NAS 

5.Logger
