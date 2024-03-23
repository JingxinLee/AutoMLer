import openml
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
import glob
import pandas as pd
import numpy as np
from keras.utils import normalize, to_categorical
from sklearn.model_selection import train_test_split

_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")


# model = "gpt-3.5-turbo-1106" or "gpt-4-1106-preview"
def get_completion(prompt, model="gpt-3.5-turbo-0125"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content


# transformers version v4.36.1
task_choices = [
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForMaskGeneration",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
    "AutoModelForMultipleChoice",
    "AutoModelForNextSentencePrediction",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForTextEncoding",
    "AutoModelForDepthEstimation",
    "AutoModelForlmageClassification",
    "AutoModelForVideoClassification",
    "AutoModelForMaskedImageModeling",
    "AutoModelForObjectDetection",
    "AutoModelForlmageSegmentation",
    "AutoModelForImageTolmage",
    "AutoModelForSemanticSegmentation",
    "AutoModelForlnstanceSegmentation",
    "AutoModelForUniversalSegmentation",
    "AutoModelForZeroShotlmageClassification",
    "AutoModelForZeroShotObjectDetection",
    "AutoModelForAudioClassification",
    "AutoModelForAudioFrameClassification",
    "AutoModelForCTC",
    "AutoModelForSpeechSeq2Seq",
    "AutoModelForAudioXVector",
    "AutoModelForTextToSpectrogram",
    "AutoModelForTextToWaveform",
    "AutoModelForTableQuestionAnswering",
    "AutoModelForDocumentQuestionAnswering",
    "AutoModelForVisualQuestionAnswering",
    "AutoModelForVision2Seq",
]


def get_markdown_files(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        print("给定的路径不存在。")
        return []

    # 构建搜索模式以匹配所有 .md 文件
    search_pattern = os.path.join(path, "*.md")

    # 使用 glob.glob() 查找所有匹配的文件路径
    markdown_files = glob.glob(search_pattern)

    return markdown_files


def select_model_from_mdfiles(task_description, markdown_files_path):
    # 加载预训练的模型
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 计算任务描述的向量
    task_vector = model.encode(task_description, convert_to_tensor=True)

    # 假设 markdown_files 是一个包含 Markdown 文件路径的列表
    markdown_files = get_markdown_files(markdown_files_path)

    # 初始化最高相似度分数和相应的文件内容
    highest_similarity = -1
    most_relevant_content = ""

    # 遍历所有 Markdown 文件
    for file_path in markdown_files:
        # 读取 Markdown 文件内容
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()

        # 计算 Markdown 文件内容的向量
        markdown_vector = model.encode(markdown_content, convert_to_tensor=True)

        # 计算余弦相似度
        similarity = util.pytorch_cos_sim(task_vector, markdown_vector)

        # 更新最高相似度分数和内容
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_relevant_content = markdown_content

    return most_relevant_content

def load_data_from_openml(dataset_name):
    pass

def local_dataset_inference(train_filepath, test_filepath, label_column):
    if test_filepath is None:
        try:
            X = pd.read_csv(train_filepath)
            Y = X[label_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )
        except:
            print("The data file path is invalid or something wrong with the data.")
            return
    else:
        try:
            X_train = pd.read_csv(train_filepath)
            y_train = X_train[label_column]
            X_test = pd.read_csv(test_filepath)
            y_test = X_test[label_column]
        except:
            print("The data file path is invalid or something wrong with the data.")
            return
    # Normalize the data
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)
    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_head = X_train.head()
    y_head = y_train.head()
    # print("X_head\n", X_head)
    # print("y_head\n", y_head)

    # TASK INFERENCE
    taskInference_prompt = f"""
        Your task is to infer the task based on the training data, X_head is the features and y_head is the labels. 
        Both the X_head and y_head are enclosed in the triple backticks.
        you should follow these steps when you infer the task:
        1. Load the X_head and y_head data.
        2. Infer the task based on the X_head and y_head data.
        3. Return the task.
 
        At the end, please return the task.
        
        X_head: ```{X_head}```
        y_head: ```{y_head}```
        
    """
    taskInference_response = get_completion(taskInference_prompt)
    print("taskInference_response:\n ", taskInference_response)
    print("*" * 100)

    # MODEL SELECT
    markdown_file_contents = select_model_from_mdfiles(
        taskInference_response,
        "/root/paper/mypaper_code/model_constructor/data/MarkdownFiles",
    )
    print("markdown_file_contents:\n ", markdown_file_contents)
    print("*" * 100)

    model_select_prompt = f"""
    Your task is to identify the most suitable model for the following task, The task is enclosed in triple backticks.
    Additionally, consider the models described in the Markdown files provided. If the most suitable model is found within the Markdown files, return its specific name, such as 'microsoft/resnet-50' or 'microsoft/resnet-18'.
    Do not give me explanation information. Only output a list of the models after you selected and compared. eg. ['microsoft/resnet-50', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'].
    If there are no models suitable, output an empty list [].
    
    task: ```{taskInference_response}```
    markdown_files: ```{markdown_file_contents}```
    """

    model_select_response = get_completion(model_select_prompt)
    print("model_select_response:\n ", model_select_response)
    model_selected_list = literal_eval(model_select_response)
    most_suitable_model = model_selected_list[0]
    print("most_suitable_model:\n ", most_suitable_model)
    print("*" * 100)

    # TRAINER which use Hugging Face Model and Trainer
    # 针对Tabular Data：BreastCancer,Californiahousing,KnotTheory,Titanic
    hf_model_trainer_prompt = f"""
        Your task is to generate training code snippet for the TASK with the MODEL give you. The TASK and MODEL is enclosed in triple backticks.
        You should follow these steps when you write the code snippet:
        1. Import the necessary libraries and modules,such as pandas,numpy,keras,sklearn,datasets,transformers etc.
        2. The TRAIN_FILEPATH and TEST_PATH and LABEL is enclosed in the triple backticks. You should follow these steps:
            2.1 If TEST_PATH is not given, just Use 'pandas.read_csv' function to load the TRAIN_FILEPATH and name it the X_train. If TEST_PATH is given, also use 'pandas.read_csv' function to load the TEST_FILEPATH then name it X_test.
            2.2 from keras.utils import normalize, to_categorical. Use normalize function to normalize the X_train and X_test. Use to_categorical function to one-hot encode the y_train and y_test.
            2.3 Split the data into training and testing sets if necessary.
            2.4 Get the head of the X_train and y_train.
        3. Initialize the MODEL. Be sure to use the most suitable model based on the MODEL. If the task related to text, use Tokenizer to tokenize the text. 
            If the task related to image classfication, do not use tokenizer but use AutoModelForImageClassification to initialize the model. If the task is a Machine Learning task, then use machine learning methods to deal.
        4. Define optimizers. The optimizers(`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple containing the optimizer and the scheduler to use. 
           First define opimizer, then define scheduler. Choose the most suitable opimizer and scheduler for the model and task. 
            Example code: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 
                          scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + decay * epoch))
                          opimizers = (opimizer, scheduler)
        5. Train the model on the train dataset. You must provide the optimizers which you defined before in Trainer.
            Example code: trainer = Trainer(..., opimizers=opimizers) 
        6. Make predictions on the testing set.
        7. Evaluate the model.
        

        MODEL: ```{most_suitable_model}```
        TASK: ```{taskInference_response}```
        TRAIN_FILEPATH:```{train_filepath}```
        TEST_FILEPATH:```{test_filepath}```
        LABEL:```{label_column}```
        
    """
    # If not provided, a default optimizer and scheduler will be created using the model's configuration.
    hf_model_trainer_response = get_completion(
        hf_model_trainer_prompt, model="gpt-4-0125-preview"
    )
    print("hf_model_trainer_response", hf_model_trainer_response)
    print("*" * 100)

    try:
        with open(
            f"./generated_scripts/{most_suitable_model.split('/')[1]}_hf.py", "w"
        ) as f:
            f.write(hf_model_trainer_response)
    except:
        with open(f"./generated_scripts/{most_suitable_model}_hf.py", "w") as f:
            f.write(hf_model_trainer_response)


def tf_dataset_inference(dataset_name):
    # Dataset for Mnist, FashionMnist,

    # Normalize the data
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)
    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_head = X_train.head()
    y_head = y_train.head()
    # print("X_head\n", X_head)
    # print("y_head\n", y_head)

    # TASK INFERENCE
    taskInference_prompt = f"""
        Your task is to infer the task based on the training data, X_head is the features and y_head is the labels. 
        Both the X_head and y_head are enclosed in the triple backticks.
        you should follow these steps when you infer the task:
        1. Load the X_head and y_head data.
        2. Infer the task based on the X_head and y_head data.
        3. Return the task.
 
        At the end, please return the task.
        
        X_head: ```{X_head}```
        y_head: ```{y_head}```
        
    """
    taskInference_response = get_completion(taskInference_prompt)
    print("taskInference_response:\n ", taskInference_response)
    print("*" * 100)

    # MODEL SELECT
    markdown_file_contents = select_model_from_mdfiles(
        taskInference_response,
        "/root/paper/mypaper_code/model_constructor/data/MarkdownFiles",
    )
    print("markdown_file_contents:\n ", markdown_file_contents)
    print("*" * 100)

    model_select_prompt = f"""
    Your task is to identify the most suitable model for the following task, The task is enclosed in triple backticks.
    Additionally, consider the models described in the Markdown files provided. If the most suitable model is found within the Markdown files, return its specific name, such as 'microsoft/resnet-50' or 'microsoft/resnet-18'.
    Do not give me explanation information. Only output a list of the models after you selected and compared. eg. ['microsoft/resnet-50', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'].
    If there are no models suitable, output an empty list [].
    
    task: ```{taskInference_response}```
    markdown_files: ```{markdown_file_contents}```
    """

    model_select_response = get_completion(model_select_prompt)
    print("model_select_response:\n ", model_select_response)
    model_selected_list = literal_eval(model_select_response)
    most_suitable_model = model_selected_list[0]
    print("most_suitable_model:\n ", most_suitable_model)
    print("*" * 100)

    # TRAINER which use Hugging Face Model and Trainer
    # 针对Tabular Data：BreastCancer,Californiahousing,KnotTheory,Titanic
    hf_model_trainer_prompt = f"""
        Your task is to generate training code snippet for the TASK with the MODEL give you. The TASK and MODEL is enclosed in triple backticks.
        You should follow these steps when you write the code snippet:
        1. Import the necessary libraries and modules,such as pandas,numpy,keras,sklearn,datasets,transformers etc.
        2. from keras.datasets import the dataset you want to use, such as mnist, fashion_mnist. 
            For example: from keras.datasets import mnist (x_train, y_train), (x_test, y_test) = mnist.load_data()
        3. Preprocess the data such as reshape and create Dataframe if necessary.
            Example code: train_data = pd.DataFrame(x_train.reshape(-1, 28*28), index=range(len(x_train)))   train_data["label"] = y_train
            test_data = pd.DataFrame(x_test.reshape(-1, 28*28), index=range(len(x_test)))   test_data["label"] = y_test
        3. Initialize the MODEL. Be sure to use the most suitable model based on the MODEL. If the task related to text, use Tokenizer to tokenize the text. 
            If the task related to image classfication, do not use tokenizer but use AutoModelForImageClassification to initialize the model. If the task is a Machine Learning task, then use machine learning methods to deal.
        4. Define optimizers. The optimizers(`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple containing the optimizer and the scheduler to use. 
           First define opimizer, then define scheduler. Choose the most suitable opimizer and scheduler for the model and task. 
            Example code: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 
                          scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + decay * epoch))
                          opimizers = (opimizer, scheduler)
        5. Train the model on the train dataset. You must provide the optimizers which you defined before in Trainer.
            Example code: trainer = Trainer(..., opimizers=opimizers) 
        6. Make predictions on the testing set.
        7. Evaluate the model.
        

        MODEL: ```{most_suitable_model}```
        TASK: ```{taskInference_response}```
        DATASET_NAME: ```{dataset_name}```

    """
    # If not provided, a default optimizer and scheduler will be created using the model's configuration.
    hf_model_trainer_response = get_completion(
        hf_model_trainer_prompt, model="gpt-4-0125-preview"
    )
    print("hf_model_trainer_response", hf_model_trainer_response)
    print("*" * 100)

    try:
        with open(
            f"./generated_scripts/{most_suitable_model.split('/')[1]}_hf.py", "w"
        ) as f:
            f.write(hf_model_trainer_response)
    except:
        with open(f"./generated_scripts/{most_suitable_model}_hf.py", "w") as f:
            f.write(hf_model_trainer_response)


def openml_task_inference(dataset_name):
    dataset = openml.datasets.get_dataset(dataset_name)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    X_head = X.head()
    y_head = y.head()
    # print("X_head\n", X_head)
    # print("y_head\n", y_head)
    # TASK INFERENCE
    taskInference_prompt = f"""
        Your task is to infer the task based on the training data, X_head is the features and y_head is the labels. 
        Both the X_head and y_head are enclosed in the triple backticks.
        you should follow these steps when you infer the task:
        1. Load the X_head and y_head data.
        2. Infer the task based on the X_head and y_head data.
        3. Return the task.
 
        At the end, please return the task.
        
        X_head: ```{X_head}```
        y_head: ```{y_head}```
        
    """
    taskInference_response = get_completion(taskInference_prompt)
    print("taskInference_response:\n ", taskInference_response)
    print("*" * 100)

    # MODEL SELECT
    # model_select_prompt = f"""
    # Your task is to identify the most suitable model for the following task, The task is enclosed in triple backticks.
    # DO not give me explanation information. Only Output a list of the models after you selected and compared. eg. ['microsoft/restnet-50', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'].
    # If there are no models suitable, output an empty list [].

    # task: ```{taskInference_response}```
    # """
    #  After selecting the most appropriate model based on your knowledge, compare it with the models in `{models}`. If the model you selected is not in `{models}`, explain why it is more suitable than the ones listed.
    #  Output your findings in JSON format with the following keys: chosen_model, compared_models, query, reason for choice.
    markdown_file_contents = select_model_from_mdfiles(
        taskInference_response,
        "/root/paper/mypaper_code/model_constructor/data/MarkdownFiles",
    )
    print("markdown_file_contents:\n ", markdown_file_contents)
    print("*" * 100)

    model_select_prompt = f"""
    Your task is to identify the most suitable model for the following task, The task is enclosed in triple backticks.
    Additionally, consider the models described in the Markdown files provided. If the most suitable model is found within the Markdown files, return its specific name, such as 'microsoft/resnet-50' or 'microsoft/resnet-18'.
    Do not give me explanation information. Only output a list of the models after you selected and compared. eg. ['microsoft/resnet-50', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'].
    If there are no models suitable, output an empty list [].
    
    task: ```{taskInference_response}```
    markdown_files: ```{markdown_file_contents}```
    """

    model_select_response = get_completion(model_select_prompt)
    print("model_select_response:\n ", model_select_response)
    model_selected_list = literal_eval(model_select_response)
    most_suitable_model = model_selected_list[0]
    print("most_suitable_model:\n ", most_suitable_model)
    print("*" * 100)

    # TRAINER which use Hugging Face Model and Trainer
    # 针对OpenML Dataset CIFAR_10
    # 切成2段，分段执行
    # 第7个单独拎出来
    # Prompt按照markdown语法来写 加粗 few shot learning写法：举例子 openml dimension check——》transform
    #  7 general assert检查dimension  --》 Warning
    hf_model_trainer_prompt = f"""
        Your task is to generate training code snippet for the TASK with the MODEL give you. The TASK and MODEL is enclosed in triple backticks.
        You should follow these steps when you write the code snippet:
        1. Import the necessary libraries and modules,such as openml, sklearn, datasets, transformers etc.
        2. Use 'openml.datasets.get_dataset' function to load the DATASET_NAME I gave you. The DATASET_NAME is enclosed in the triple backticks. for example,get_dataset('CIFAR_10')
        At the end, please return the Trainer code snippet with some usage code snippet.
        3. Use 'get_data' function to get the data from the dataset. dataset_format="dataframe",target=dataset.default_target_attribute. Use X and y to store the data.
        4. Split the data into training and testing sets.
        5. Convert the data to a CSV file for easy reading by the Hugging Face datasets library. You should follow these steps:
            5.1 Create a pandas DataFrame train_df use the X_train, append a column to the DataFrame with the column name 'target' and the values of y_train.
            5.2 Create a pandas DataFrame test_df use the X_test, append a column to the DataFrame with the column name 'target' and the values of y_test.
            5.3 Use to_csv function to generate the csv file.
            5.4 Load the csv file with the load_dataset function. Note the data_files in the load_dataset function should be a dict, the key is the split, and the value is the csv file path.
        6. Initialize the MODEL. Be sure to use the most suitable model based on the MODEL. If the task related to text, use Tokenizer to tokenize the text. 
            If the task related to image classfication, do not use tokenizer but use AutoModelForImageClassification to initialize the model. 
            Use ignore_mismatched_sizes=True when you use AutoModelForImageClassification to initialize the model.
        7. If use openml dataset such as cifar10, create a preprocess function to preprocess the data. 
            7.1 Use Augly Aument the data if necessary. For example, you can use aug_np_wrapper to augment the images. 
                Example code: from augly.image import aug_np_wrapper, overlay_emoji  augmented_images = [aug_np_wrapper(np.array(img, dtype=np.uint8).reshape((3, 1024)), overlay_emoji,opacity=0.5, y_pos=0.45) for img in zip(*(examples["a" + str(i)] for i in range(3072)))]
            7.1 Transform all the features number such as a0 a1... a3071 to 3 dimension (3,x,x), the first dimension is 3, the second and third dimension is x which use sqrt(number of features/3) to calculate.
                Example code: images = [torch.Tensor(list(img)).view(3, 32, 32) for img in zip(*(examples['a'+str(i)] for i in range(3072)))]
            7.2 Then save them to a parameter that the model requires. For example you can save it to examples['pixel_values']. 
            7.3 Then save the target numbers to lables such as examples['labels'].  
            If use other dataset, you can skip this step.
        8. Define optimizers. The optimizers(`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple containing the optimizer and the scheduler to use. 
           First define opimizer, then define scheduler. Choose the most suitable opimizer and scheduler for the model and task. 
            Example code: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 
                          scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + decay * epoch))
                          opimizers = (opimizer, scheduler)
        8. Train the model on the train dataset. You must provide the optimizers which you defined before in Trainer.
            Example code: trainer = Trainer(..., opimizers=opimizers) 
        9. Make predictions on the testing set.
        10. Evaluate the model.
        

        MODEL: ```{most_suitable_model}```
        TASK: ```{taskInference_response}```
        DATASET_NAME: ```{dataset_name}```
        
    """
    # If not provided, a default optimizer and scheduler will be created using the model's configuration.
    hf_model_trainer_response = get_completion(
        hf_model_trainer_prompt, model="gpt-4-0125-preview"
    )
    print("hf_model_trainer_response", hf_model_trainer_response)
    print("*" * 100)

    try:
        with open(
            f"./generated_scripts/{most_suitable_model.split('/')[1]}_hf2.py", "w"
        ) as f:
            f.write(hf_model_trainer_response)
    except:
        with open(f"./generated_scripts/{most_suitable_model}_hf2.py", "w") as f:
            f.write(hf_model_trainer_response)

    # # TRAINER which not use Hugging Face Model and Trainer
    # trainer_prompt = f"""
    #     Your task is to generate training code snippet for the task with the model give you. The task and model is enclosed in triple backticks.
    #     You should follow these steps when you write the code snippet:
    #     1. Import the necessary libraries and modules,such as openml, sklearn, datasets, transformers, Trainer etc.
    #     2. Use 'openml.datasets.get_dataset' function to load the dataset using the dataset name I gave you. The dataset name is enclosed in the triple backticks.
    #     At the end, please return the Trainer code snippet with some usage code snippet.
    #     3. Use 'get_data' function to get the data from the dataset. dataset_format="dataframe",target=dataset.default_target_attribute.
    #     4. Split the data into training and testing sets.
    #     5. Initialize the model.
    #     6. Train the model.
    #     7. Make predictions on the testing set.
    #     8. Evaluate the model.

    #     model: ```{most_suitable_model}```
    #     task: ```{taskInference_response}```
    #     dataset: ```{dataset_name}```

    # """
    # trainer_response = get_completion(trainer_prompt, model="gpt-4-1106-preview")
    # print(trainer_response)
    # with open(f'./generated_scripts/{most_suitable_model}.py', 'w') as f:
    #     f.write(trainer_response)


if __name__ == "__main__":
    # openml_task_inference('CIFAR_10')

    openml_task_inference("CIFAR_10")
