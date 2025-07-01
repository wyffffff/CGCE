import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

MODEL_NAME = "unixcoder-base-nine"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_LENGTH = 1024
TEMPERATURE = 0.1
MODEL_SAVE_DIR = 'saved_models_CL'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model_bigvul.pth')

LABEL_DESCRIPTIONS = {
    0: "This code does not contain defects.",
    1: "This code contains defects."
}

class CodeDefectDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.labels = dataframe.iloc[:, 0].values
        self.codes = dataframe.iloc[:, 1].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = int(self.labels[idx])

        code_encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'code_input_ids': code_encoding['input_ids'].squeeze(0),
            'code_attention_mask': code_encoding['attention_mask'].squeeze(0),
            'label_id': label
        }

class CodeTextCLIPModel(nn.Module):
    def __init__(self, model_name, num_classes, temperature):
        super(CodeTextCLIPModel, self).__init__()
        self.code_encoder = RobertaModel.from_pretrained(model_name)
        self.text_encoder = RobertaModel.from_pretrained(model_name)
        self.temperature = temperature

        label_descriptions = [LABEL_DESCRIPTIONS[i] for i in range(num_classes)]
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        label_encoding = tokenizer(
            label_descriptions,
            add_special_tokens=True,
            max_length=MAX_LENGTH // 4,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            label_outputs = self.text_encoder(
                input_ids=label_encoding['input_ids'],
                attention_mask=label_encoding['attention_mask']
            )
            label_embeddings = label_outputs.last_hidden_state[:, 0, :]
            label_embeddings = nn.functional.normalize(label_embeddings, dim=1)

        self.label_embeddings = nn.Parameter(label_embeddings)

    def forward(self, code_input_ids, code_attention_mask):
        code_outputs = self.code_encoder(input_ids=code_input_ids, attention_mask=code_attention_mask)
        code_embeddings = code_outputs.last_hidden_state[:, 0, :]
        code_embeddings = nn.functional.normalize(code_embeddings, dim=1)
        return code_embeddings

class CGCE(nn.Module):
    def __init__(self, temperature=0.07):
        super(CGCE, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, code_embeddings, label_embeddings, labels):
        logits = torch.matmul(code_embeddings, label_embeddings.t()) / self.temperature
        loss = self.cross_entropy(logits, labels)
        return loss

def load_data(csv_path, tokenizer, max_length):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件 '{csv_path}' 不存在。请检查路径。")

    dataframe = pd.read_csv(csv_path)
    dataset = CodeDefectDataset(dataframe, tokenizer, max_length)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"训练 第{epoch+1}/{num_epochs}轮", leave=False)

        for batch in pbar:
            code_input_ids = batch['code_input_ids'].to(device)
            code_attention_mask = batch['code_attention_mask'].to(device)
            labels = batch['label_id'].to(device)

            code_embeddings = model(code_input_ids, code_attention_mask)
            label_embeddings = model.label_embeddings
            loss = criterion(code_embeddings, label_embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({'损失': f"{avg_loss:.4f}"})

        print(f'第{epoch+1}/{num_epochs}轮, 损失: {avg_loss:.4f}')
        accuracy = evaluate_model(model, val_dataloader, device, mode='validation')
        print(f'第{epoch+1}/{num_epochs}轮, 验证集准确率: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'最佳模型已保存，验证集准确率: {best_accuracy:.4f}')

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(device)
        print(f'最佳模型已加载，验证集准确率: {best_accuracy:.4f}')
    else:
        print("未找到最佳模型。")

def evaluate_model(model, dataloader, device, mode='test'):
    model.eval()
    y_true = []
    y_pred = []
    num_classes = len(LABEL_DESCRIPTIONS)

    with torch.no_grad():
        label_embeddings = model.label_embeddings
        label_embeddings = nn.functional.normalize(label_embeddings, dim=1)

        pbar = tqdm(dataloader, desc=f"评估 {mode.capitalize()} 集", leave=False)
        for batch in pbar:
            code_input_ids = batch['code_input_ids'].to(device)
            code_attention_mask = batch['code_attention_mask'].to(device)
            labels = batch['label_id'].cpu().numpy()

            code_embeddings = model(code_input_ids, code_attention_mask)
            logits = torch.matmul(code_embeddings, label_embeddings.t()) / model.temperature
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(predicted_labels)

    accuracy = accuracy_score(y_true, y_pred)

    if mode == 'test':
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f'测试集准确率: {accuracy:.4f}')
        print(f'精确率: {precision:.4f}')
        print(f'召回率: {recall:.4f}')
        print(f'F1 分数: {f1:.4f}')

        target_names = [LABEL_DESCRIPTIONS[i] for i in sorted(LABEL_DESCRIPTIONS.keys())]
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        print("分类报告:\n", report)
    else:
        return accuracy

def main():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    csv_path = 'labelcode.csv'
    train_dataloader, val_dataloader, test_dataloader = load_data(csv_path, tokenizer, MAX_LENGTH)

    num_classes = len(LABEL_DESCRIPTIONS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CodeTextCLIPModel(MODEL_NAME, num_classes, TEMPERATURE).to(device)

    criterion = CGCE(temperature=TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, NUM_EPOCHS)
    evaluate_model(model, test_dataloader, device, mode='test')

if __name__ == "__main__":
    main()
