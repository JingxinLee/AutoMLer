# generated snippet Boston 
# import openml
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # 2. Use 'openml.datasets.get_dataset' function to load the dataset
# dataset = openml.datasets.get_dataset('boston')

# # 3. Use 'get_data' function to get the data from the dataset
# X, y, _, _ = dataset.get_data(dataset_format="dataframe", 
#                               target=dataset.default_target_attribute
#                               )

# # 4. Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Initialize the model
# model = LinearRegression()

# # 6. Train the model
# model.fit(X_train, y_train)

# # X_test=X_test.astype(float)
# # 7. Make predictions on the testing set
# y_pred = model.predict(X_test)

# # 8. Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)





# generated snippet Diabetes
# 1. Import the necessary libraries and modules
# import openml
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # 2. Use 'openml.datasets.get_dataset' function to load the dataset
# diabetes_dataset = openml.datasets.get_dataset('diabetes')

# # 3. Use 'get_data' function to get the data from the dataset
# X, y, _, _ = diabetes_dataset.get_data(dataset_format="dataframe", target=diabetes_dataset.default_target_attribute)

# # 4. Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Initialize the model
# model = RandomForestClassifier()

# # 6. Train the model
# model.fit(X_train, y_train)

# # 7. Make predictions on the testing set
# y_pred = model.predict(X_test)

# # 8. Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")


# Generated snippet CIFAR_10


# # 1. Import the necessary libraries and modules
# import openml
# from sklearn.model_selection import train_test_split
# from transformers import Trainer
# from transformers import AutoModelForImageClassification
# from transformers import TrainingArguments

# # 2. Use 'openml.datasets.get_dataset' function to load the dataset
# dataset = openml.datasets.get_dataset('CIFAR_10')

# # 3. Use 'get_data' function to get the data from the dataset
# X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# # 4. Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Initialize the model
# model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

# # 6. Train the model
# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )
# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
# )
# trainer.train()

# # 7. Make predictions on the testing set
# predictions = trainer.predict(X_test)

# # 8. Evaluate the model
# results = trainer.evaluate()


# # 1. Import the necessary libraries and modules                                                                                        
# import openml                                                                                                                           
# from sklearn.model_selection import train_test_split                                                                                   
# from transformers import Trainer, TrainingArguments                                                                                   
# from datasets import load_dataset                                                                                                     
# from transformers import AutoModelForSequenceClassification, AutoTokenizer                                                             
                                                                                                                                        
# # 2. Use 'openml.datasets.get_dataset' function to load the dataset                                                               
# dataset = openml.datasets.get_dataset(6)                                                                                               
                                                                                                                                        
# # 3. Use 'get_data' function to get the data from the dataset                                                                         
# data = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)                                            
# X, y, _, _ = data                                                                                                                       
                                                                                                                                     
# # 4. Split the data into training and testing sets                                                                                    
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                                            
                                                                                                                                      
# # 5. Transform the data into the format that huggingface model can understand                                                        
# dataset = load_dataset('cifar10')                                                                                                    
                                                                                                                                    
# # 5. Initialize the model                                                                                                             
# model_name = "microsoft/resnet-50"  # Replace with the actual model name                                                             
# model = AutoModelForSequenceClassification.from_pretrained(model_name)                                                                  
# tokenizer = AutoTokenizer.from_pretrained(model_name)                                                                                  
                                                                                                                                 
# # 6. Train the model on the train dataset                                                                                              
# training_args = TrainingArguments(                                                                                                      
#     output_dir='./results',          # output directory                                                                                 
#     num_train_epochs=3,              # total number of training epochs                                                                  
#     per_device_train_batch_size=16,  # batch size per device during training                                                            
#     per_device_eval_batch_size=64,   # batch size for evaluation                                                                        
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler                                               
#     weight_decay=0.01,               # strength of weight decay                                                                         
#     logging_dir='./logs',            # directory for storing logs                                                                       
# )                                                                                                                                       
                                                                                                                                        
# trainer = Trainer(                                                                                                                      
#     model=model,                         # the instantiated 🤗 Transformers model to be trained                                         
#     args=training_args,                  # training arguments, defined above                                                            
#     train_dataset=dataset['train'],      # training dataset                                                                             
#     eval_dataset=dataset['test']         # evaluation dataset                                                                           
# )                                                                                                                                       
                                                                                                                                        
# trainer.train()                                                                                                                         
                                                                                                                                        
# # 7. Make predictions on the testing set                                                                                                
# predictions = trainer.predict(dataset['test'])                                                                                          
                                                                                                                                        
# # 8. Evaluate the model                                                                                                                 
# results = trainer.evaluate()




# # 1. Import necessary libraries and modules
# import openml
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import Trainer, TrainingArguments
# from datasets import load_dataset
# from transformers import AutoModelForImageClassification, AutoTokenizer

# # 2. Load the dataset
# dataset = openml.datasets.get_dataset("CIFAR_10")

# # 3. Get the data from the dataset
# X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# # 4. Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Convert the data to a CSV file
# train_df = pd.DataFrame(X_train, columns=dataset.features)
# train_df['target'] = y_train
# train_df.to_csv('train_dataset.csv', index=False)

# test_df = pd.DataFrame(X_test, columns=dataset.features)
# test_df['target'] = y_test
# test_df.to_csv('test_dataset.csv', index=False)

# # 5.4 Load the csv file with the load_dataset function
# train_dataset = load_dataset('csv', data_files={'train': 'train_dataset.csv'})
# test_dataset = load_dataset('csv', data_files={'test': 'test_dataset.csv'})

# # 5. Initialize the MODEL
# model_name = "microsoft/resnet-50"  # Example model
# model = AutoModelForImageClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 6. Train the model on the train dataset
# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )

# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
# )

# trainer.train()

# # 7. Make predictions on the testing set
# predictions = trainer.predict(test_dataset)

# # 8. Evaluate the model
# results = trainer.evaluate()


# 1. Import the necessary libraries and modules
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoTokenizer

# 2. Use 'openml.datasets.get_dataset' function to load the CIFAR_10 dataset
dataset = openml.datasets.get_dataset('CIFAR_10')

# 3. Use 'get_data' function to get the data from the dataset
X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Convert the data to a CSV file for easy reading by the Hugging Face datasets library
# 5.1 Create a pandas DataFrame train_df

# train_df = pd.DataFrame(X_train)
# train_df['target'] = y_train
# # 5.2 Create a pandas DataFrame test_df
# test_df = pd.DataFrame(X_test)
# test_df['target'] = y_test
# # 5.3 use to_csv function to generate the csv file
# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)
# 5.4 Load the csv file with the load_dataset function
train_dataset = load_dataset('csv', data_files={'train': 'train.csv'}, split="train")
test_dataset = load_dataset('csv', data_files={'test': 'test.csv'}, split="test")

# 6. Initialize the MODEL
model_name = "microsoft/resnet-50"  # Example model
model = AutoModelForImageClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 7. Train the model on the train dataset
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    remove_unused_columns=False
)
trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset  # evaluation dataset
)
trainer.train()

# 8. Make predictions on the testing set
predictions = trainer.predict(test_dataset)


# 9. Evaluate the model
results = trainer.evaluate()