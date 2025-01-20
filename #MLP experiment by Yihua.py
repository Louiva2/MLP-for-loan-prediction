#MLP experiment by Yihua
import torch
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#include the training data and test data
train_data=pd.read_csv('/home/public/MLP and transformer experiments by Yihua/data/train.csv')
test_data=pd.read_csv('/home/public/MLP and transformer experiments by Yihua/data/test.csv')


# Identify categorical columns
onehot_columns = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file'] 
label_columns = ['loan_grade']

# One-Hot Encoding for categorical columns
encoder = OneHotEncoder(sparse_output=False)
encoded_train_data = encoder.fit_transform(train_data[onehot_columns])
encoded_test_data = encoder.transform(test_data[onehot_columns])

# Label Encoding for label columns
label_encoder = LabelEncoder()
encoded_train_data_label = train_data[label_columns].apply(label_encoder.fit_transform)
encoded_test_data_label = test_data[label_columns].apply(label_encoder.transform)

# Drop original categorical columns and concatenate encoded columns
train_data = train_data.drop(onehot_columns+label_columns , axis=1)
test_data = test_data.drop(onehot_columns+label_columns, axis=1)
train_data = np.concatenate([train_data.values, encoded_train_data,encoded_train_data_label], axis=1)
test_data = np.concatenate([test_data.values, encoded_test_data,encoded_test_data_label], axis=1)

#the last column is the target variable, and the firsr column is the id
x_train = train_data[:, 1:-1]
y_train = train_data[:, -1]
x_test = test_data[:, 1:]
#y_test = test_data[:, -1]

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_train=y_train.clamp(1e-7,1-1e-7)
x_test = torch.tensor(x_test, dtype=torch.float32)
#y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)



# #batch the data(use it when no K-FOLD)
# train_data=TensorDataset(x_train,y_train)
# train_data=DataLoader(train_data,batch_size=64,shuffle=True)
# # x_test=DataLoader(x_test,batch_size=64,shuffle=True)

#input normalization
scaler = StandardScaler()
x_train = torch.tensor(scaler.fit_transform(x_train),dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test),dtype=torch.float32)

# write the multi-layer perceptron model with shortcut connections for a model with 20 inputs and 1 output
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(20, 64)
        self.bn = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = torch.relu(self.fc2(x))+x
        x = self.bn(x)
        x = torch.relu(self.fc2(x))+x
        x = self.bn(x)
        x = torch.relu(self.fc2(x))+x
        x = self.fc4(x)
        x=torch.sigmoid(x)
        # x=x.clamp(1e-7,1-1e-7)
        return x
model=MLP()
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)


#try K fold to reduce overfitting
scores = []
kf = KFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in kf.split(x_train):
    # 划分训练集和测试集
    training_x, testing_x = x_train[train_index], x_train[test_index]
    training_y, testing_y = y_train[train_index], y_train[test_index]
    train_dataset=TensorDataset(training_x,training_y)
    train_dataset=DataLoader(train_dataset,batch_size=64,shuffle=True)
    
    num_epochs=100
    for epochs in range(num_epochs):
        model.train()
        start_time=time.time()
        for batch_x,batch_y in train_dataset:
            optimizer.zero_grad()
            y_predict=model(batch_x)
            loss=loss_fn(y_predict,batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step(loss) 
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epochs}, Loss: {loss.item()}, Time: {epoch_time:.4f} seconds")           
    # 测试模型
    model.eval()
    with torch.no_grad():
        outputs = model(testing_x)
        predicted = (outputs > 0.5).int()
        testing_y = (testing_y>0.5).int()
        accuracy = accuracy_score(testing_y, predicted)
        scores.append(accuracy)
        # 计算平均准确率
        print(f'Cross-Validation Accuracy Scores: {scores}')
        print(f'Mean Accuracy: {np.mean(scores)}')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model.pth')

# Evaluate the model after training
model.eval()
with torch.no_grad():
    y_val_pred = model(x_test)

y_val_pred = y_val_pred.numpy().squeeze()
test_ids = pd.read_csv('/home/public/MLP and transformer experiments by Yihua/data/test.csv')['id']
combined = pd.DataFrame({'id': test_ids, 'target': y_val_pred})
combined.to_csv('/home/public/MLP and transformer experiments by Yihua/data/submission.csv', index=False)










# # train the model(without KFOLD)
# for i in range(200):
#     start_time = time.time()

#     model.train()




#     for x_batch, y_batch in train_data:
#         y_pred = model(x_batch)
#         loss = loss_fn(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    

#     # y_pred = model(x_train)
#     # loss = loss_fn(y_pred, y_train)
#     # optimizer.zero_grad()
#     # loss.backward()
#     # optimizer.step()
#     scheduler.step(loss)
#     end_time = time.time()
#     epoch_time = end_time - start_time
#     print(f"Epoch {i}, Loss: {loss.item()}, Time: {epoch_time:.4f} seconds")

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict()
# }, 'model.pth')





# # Evaluate the model after training
# model.eval()
# with torch.no_grad():
#     y_val_pred = model(x_test)

# y_val_pred = y_val_pred.numpy().squeeze()
# test_ids = pd.read_csv('/home/public/MLP and transformer experiments by Yihua/data/test.csv')['id']
# combined = pd.DataFrame({'id': test_ids, 'target': y_val_pred})
# combined.to_csv('/home/public/MLP and transformer experiments by Yihua/data/submission.csv', index=False)


