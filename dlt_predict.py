import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 设置文件路径
BASE_DIR = os.path。dirname(os.path。abspath(__file__))
sys.path。append(BASE_DIR)
CSV_FILE_PATH = os.path。join(BASE_DIR, 'dlt_lottery_data.csv')

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self)。__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)。到(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)。到(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 处理数据
def process_data(seq_length):
    data = pd.read_csv(CSV_FILE_PATH)
    red_data = data.iloc[:, 1:6]。values  # 红色球数据
    blue_data = data.iloc[:, 6:8]。values  # 蓝色球数据

    red_data = torch.tensor(red_data, dtype=torch.float32)
    blue_data = torch.tensor(blue_data, dtype=torch.float32)

    # 标准化
    red_mean = red_data.mean(dim=0)
    red_std = red_data.std(dim=0)
    red_data = (red_data - red_mean) / red_std

    blue_mean = blue_data.mean(dim=0)
    blue_std = blue_data.std(dim=0)
    blue_data = (blue_data - blue_mean) / blue_std

    red_train, red_target, blue_train, blue_target = []， []， []， []

    # 构建序列数据
    for i in range(len(red_data) - seq_length):
        red_train.append(red_data[i:i + seq_length])
        red_target.append(red_data[i + seq_length])
    red_train = torch.stack(red_train)
    red_target = torch.stack(red_target)

    for i in range(len(blue_data) - seq_length):
        blue_train.append(blue_data[i:i + seq_length])
        blue_target.append(blue_data[i + seq_length])
    blue_train = torch.stack(blue_train)
    blue_target = torch.stack(blue_target)

    return red_train, red_target, blue_train, blue_target, red_mean, red_std, blue_mean, blue_std

def train_model(input_size, hidden_size, output_size, num_layers, train_data, target_data, num_epochs=100):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

def predict(model, data):
    model.eval()
    with torch.no_grad():
        test_input = data.unsqueeze(0)
        prediction = model(test_input)
    return prediction

def scale_to_range_tensor(values, old_min, old_max, new_min, new_max):
    if old_max == old_min:
        return torch.full_like(values, new_min)

    values = torch.clamp(values, old_min, old_max)
    scaled_values = new_min + (new_max - new_min) * (values - old_min) / (old_max - old_min)
    return torch.round(scaled_values)

def ensure_unique(values, min_val, max_val, num_values):
    unique_values = set()
    while len(unique_values) < num_values:
        value = random.randint(min_val, max_val)
        if value not in unique_values:
            unique_values.add(value)
    return torch.tensor(list(unique_values), dtype=torch.int32)。unsqueeze(0)

if __name__ == '__main__':
    seq_length = 10
    hidden_size = 32
    num_layers = 2
    num_epochs = 200

    # 处理数据
    red_train, red_target, blue_train, blue_target, red_mean, red_std, blue_mean, blue_std = process_data(seq_length)

    # 训练红球模型
    red_model = train_model(red_train.size(-1), hidden_size, red_train.size(-1), num_layers, red_train, red_target, num_epochs)
    
    # 训练蓝球模型
    blue_model = train_model(blue_train.size(-1), hidden_size, blue_train.size(-1), num_layers, blue_train, blue_target, num_epochs)

    # 预测并输出多组结果
    def generate_multiple_predictions(model, data, mean, std, scale_min, scale_max, num_sets, num_unique):  
        predictions = []
        for _ in range(num_sets):
            prediction = predict(model, data)
            prediction = prediction * std + mean  # 反标准化
            prediction = scale_to_range_tensor(prediction, prediction.min()。item(), prediction.max()。item(), scale_min, scale_max)  
            prediction = ensure_unique(prediction, scale_min, scale_max, num_unique)
            predictions.append(prediction.int()。tolist()[0])  # 保存预测结果
        return predictions

    # 生成 5 组红色和蓝色球
    red_predictions_sets = generate_multiple_predictions(red_model, red_train[-1], red_mean, red_std, 1， 35， 5， 5)
    blue_predictions_sets = generate_multiple_predictions(blue_model, blue_train[-1], blue_mean, blue_std, 1， 12， 5， 2)

    # 输出结果
    print("Predictions (Red Balls and Blue Balls):")
    for i in range(5):
        print(f"Set {i + 1}: Red: {red_predictions_sets[i]}, Blue: {blue_predictions_sets[i]}")
