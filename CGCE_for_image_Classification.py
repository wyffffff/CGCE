import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTFeatureExtractor
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from PIL import ImageFile  # 添加此行
from torch.cuda import amp  # 添加此行

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 添加此行

# =======================
# 超参数定义
# =======================
# 随机种子
SEED = 42

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据相关参数
IMAGENET_DIR = 'imagenet'  # 替换为您的 ImageNet 数据集根目录
TRAIN_DIR = os.path.join(IMAGENET_DIR, 'train')
VAL_DIR = os.path.join(IMAGENET_DIR, 'val')
BATCH_SIZE = 256
NUM_WORKERS = 8


# 数据增强和预处理
# ImageNet 的数据增强通常包括更复杂的增强策略
def get_transforms(feature_extractor):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    return transform_train, transform_val


# 模型参数
MODEL_NAME = "vit-base-patch16-224-in21k"  # 确保使用正确的模型名称
NUM_CLASSES = 1000  # ImageNet 有1000个类别

PRETRAINED = True

# 优化器参数
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2

# 学习率调度器参数
NUM_EPOCHS = 30
COSINE_T_MAX = NUM_EPOCHS  # 一个完整的余弦周期
COSINE_ETA_MIN = 1e-6

# 混合精度训练
USE_AMP = False


# =======================
# 函数和类定义
# =======================

# 定义设置随机种子的函数
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义 CGCE
class CGCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(CGCE, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, code_embeddings, label_embeddings, labels):
        # 计算相似度矩阵
        logits = torch.matmul(code_embeddings, label_embeddings.t()) / self.temperature
        loss = self.cross_entropy(logits, labels)
        return loss


# 定义可学习的标签嵌入
class LabelEmbeddings(nn.Module):
    def __init__(self, num_classes, embedding_dim, initial_embeddings=None):
        super(LabelEmbeddings, self).__init__()
        self.label_embeddings = nn.Embedding(num_classes, embedding_dim)
        if initial_embeddings is not None:
            # 确保 initial_embeddings 的形状为 [num_classes, embedding_dim]
            assert initial_embeddings.shape == (num_classes, embedding_dim), \
                f"Expected initial_embeddings shape {(num_classes, embedding_dim)}, but got {initial_embeddings.shape}"
            self.label_embeddings.weight = nn.Parameter(initial_embeddings)
        else:
            # 默认初始化
            nn.init.normal_(self.label_embeddings.weight, mean=0.0, std=0.01)

    def forward(self):
        return self.label_embeddings.weight  # 返回形状为 [num_classes, embedding_dim]


# =======================
# 主函数
# =======================
def main():
    # 设置随机种子
    set_seed(SEED)

    # 打印设备信息
    print(f'Using device: {DEVICE}')

    # 1. 加载预训练模型和特征提取器
    model = ViTModel.from_pretrained(MODEL_NAME)
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # 2. 定义数据预处理
    transform_train, transform_val = get_transforms(feature_extractor)

    # 3. 加载 ImageNet 数据集
    if not os.path.isdir(IMAGENET_DIR):
        raise FileNotFoundError(f"ImageNet目录在 {IMAGENET_DIR} 未找到")

    # 使用 ImageFolder 加载训练集和验证集
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform_val)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 4. 计算每个类别的平均嵌入作为标签嵌入（用于初始化）
    embedding_dim = model.config.hidden_size  # ViT 的隐藏层大小

    # 初始化类别嵌入和计数
    class_embeddings = [torch.zeros(embedding_dim).to(DEVICE) for _ in range(NUM_CLASSES)]
    class_counts = [0 for _ in range(NUM_CLASSES)]

    model.eval()
    with torch.no_grad():
        loop = tqdm(train_loader, desc="计算类别平均嵌入")
        for inputs, labels in loop:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的嵌入
            outputs = outputs / outputs.norm(dim=1, keepdim=True)  # 归一化

            for i in range(inputs.size(0)):
                label = labels[i].item()
                class_embeddings[label] += outputs[i]
                class_counts[label] += 1

    # 计算平均嵌入
    class_prototypes = []
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            class_embeddings[i] /= class_counts[i]
        else:
            print(f"警告：类别 {i} 没有样本。")
            # 为没有样本的类别随机初始化嵌入
            class_embeddings[i] = torch.randn(embedding_dim).to(DEVICE) * 0.01
        class_prototypes.append(class_embeddings[i])

    class_prototypes = torch.stack(class_prototypes)  # 形状为 [num_classes, embedding_dim]

    # 5. 定义损失函数
    criterion = CGCE().to(DEVICE)

    # 6. 定义可学习的标签嵌入
    label_emb = LabelEmbeddings(num_classes=NUM_CLASSES, embedding_dim=embedding_dim,
                                initial_embeddings=class_prototypes).to(DEVICE)

    # 7. 定义优化器
    # 包含主模型参数和标签嵌入参数
    optimizer = optim.AdamW(list(model.parameters()) + list(label_emb.parameters()), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # 8. 定义学习率调度器为余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN)

    # 9. 初始化GradScaler用于混合精度训练
    scaler = amp.GradScaler() if USE_AMP else None

    # 10. 定义训练和评估函数
    def train_epoch(model, label_emb, dataloader, optimizer, criterion, device, scaler=None):
        model.train()
        label_emb.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(dataloader, desc="训练中", leave=False)
        for batch in loop:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with amp.autocast():
                    outputs = model(inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的嵌入
                    outputs = outputs / outputs.norm(dim=1, keepdim=True)  # 归一化

                    label_embeddings = label_emb()  # [num_classes, embedding_dim]
                    label_embeddings = label_embeddings / label_embeddings.norm(dim=1, keepdim=True)  # 归一化

                    loss = criterion(outputs, label_embeddings, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 不使用混合精度
                outputs = model(inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的嵌入
                outputs = outputs / outputs.norm(dim=1, keepdim=True)  # 归一化

                label_embeddings = label_emb()  # [num_classes, embedding_dim]
                label_embeddings = label_embeddings / label_embeddings.norm(dim=1, keepdim=True)  # 归一化

                loss = criterion(outputs, label_embeddings, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 计算预测
            with torch.no_grad():
                logits = torch.matmul(outputs, label_embeddings.t()) / criterion.temperature
                predicted = logits.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            loop.set_postfix(loss=loss.item(), accuracy=correct / total)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate_epoch(model, label_emb, dataloader, criterion, device):
        model.eval()
        label_emb.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(dataloader, desc="评估中", leave=False)
            for batch in loop:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs).last_hidden_state[:, 0, :]
                outputs = outputs / outputs.norm(dim=1, keepdim=True)  # 归一化

                label_embeddings = label_emb()  # [num_classes, embedding_dim]
                label_embeddings = label_embeddings / label_embeddings.norm(dim=1, keepdim=True)  # 归一化

                loss = criterion(outputs, label_embeddings, labels)

                running_loss += loss.item() * inputs.size(0)

                logits = torch.matmul(outputs, label_embeddings.t()) / criterion.temperature
                predicted = logits.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item(), accuracy=correct / total)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # 11. 训练模型
    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(model, label_emb, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_acc = evaluate_epoch(model, label_emb, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print("-" * 30)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'label_emb_state_dict': label_emb.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc,
            }
            if USE_AMP and scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()  # 保存scaler的状态
            torch.save(checkpoint, 'best_vit_model.pth')
            print(f"最佳模型已保存，验证准确率: {best_val_accuracy:.4f}")

    print(f"训练完成。最佳验证准确率: {best_val_accuracy:.4f}")

    # 12. 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='训练损失')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('损失曲线')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='训练准确率')
    plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('准确率曲线')

    plt.show()

    # 13. 保存模型
    save_path = "./vit_imagenet_supervised_contrastive.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_emb_state_dict': label_emb.state_dict()
    }, save_path)
    feature_extractor.save_pretrained("./vit_imagenet_model")

    print(f"模型已保存至 {save_path} 和 ./vit_imagenet_model")

    # 14. 加载并预测新图像
    def predict_image(image_path, model, transform, device, label_emb, class_idx_to_label, criterion):
        """
        对单张图像进行预测。

        参数:
        - image_path (str): 图像文件路径。
        - model (nn.Module): 训练好的模型。
        - transform (transforms.Compose): 图像预处理。
        - device (torch.device): 设备。
        - label_emb (nn.Module): 可学习的标签嵌入模块。
        - class_idx_to_label (dict): 类别索引到标签名称的映射。
        - criterion (nn.Module): 损失函数（用于 temperature）。

        返回:
        - predicted_label (str): 预测的类别名称。
        """
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        model.eval()
        label_emb.eval()
        with torch.no_grad():
            label_embeddings = label_emb()  # 获取最新的标签嵌入
            label_embeddings = label_embeddings / label_embeddings.norm(dim=1, keepdim=True)  # 归一化

            embedding = model(image).last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=1, keepdim=True)  # 归一化
            logits = torch.matmul(embedding, label_embeddings.t()) / criterion.temperature
            predicted_class = logits.argmax(dim=1).item()

        return class_idx_to_label[predicted_class]

    # 创建类别索引到标签名称的映射
    class_idx_to_label = {v: k for k, v in train_dataset.class_to_idx.items()}

    # 示例预测（根据需要自行启用）
    # image_path = "path_to_your_image.jpg"
    # if os.path.exists(image_path):
    #     predicted_label = predict_image(image_path, model, transform_val, DEVICE, label_emb, class_idx_to_label, criterion)
    #     print(f"预测类别: {predicted_label}")
    # else:
    #     print(f"图像路径不存在: {image_path}")


if __name__ == '__main__':
    main()
