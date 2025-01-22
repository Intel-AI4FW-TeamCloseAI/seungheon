import torch
import torchvision.models as models
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn

# -------------------------------------
# 0) device 설정
# -------------------------------------
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------
# 1) 모델 선언 및 마지막 레이어 교체
# -------------------------------------
model = models.efficientnet_b0(pretrained=True)  # ImageNet 사전학습
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # binary class (Fake, Real)

# -------------------------------------
# 2) 데이터 로더 설정
# -------------------------------------
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

base_dir = os.getcwd()  # 현재 작업 디렉토리
train_dir = os.path.join(base_dir, 'Train')
val_dir   = os.path.join(base_dir, 'Validation')
test_dir  = os.path.join(base_dir, 'Test')

batch = 16
img_resized = (128, 128)

train_transform = transforms.Compose([
    transforms.Resize(img_resized),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(img_resized),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_dir, transform=test_transform)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=test_transform)

# num_workers=0 == 메인 프로세스에서만 데이터를 로드하도록 함, CPU 코어 수 x (0.5 to 1.0) 로 설정 권장
# pin_memory=True == 데이터를 GPU로 전송할 때 복사를 사용
train_loader  = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=8, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)

# 클래스 인덱스 확인
print("Train Dataset class to index mapping:")
print(train_dataset.class_to_idx)

# -------------------------------------
# 3) 학습 설정 (Loss, Optimizer 등)
# -------------------------------------
import torch.optim as optim

model = model.to(device)  # 모델을 device(GPU/CPU)에 올림
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50

# -------------------------------------
# 4) EarlyStopping 클래스 정의
# -------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='best_model.pth'):
        """
        Args:
            patience (int): 개선이 없을 때 기다릴 에포크 수
            verbose (bool): 상세 로그 출력 여부
            delta (float): 개선의 최소 변화량
            path (str): 최적 모델을 저장할 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss  # 검증 손실이 낮을수록 좋으므로 음수로 변환

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """검증 손실이 감소하면 모델을 저장"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# -------------------------------------
# 5) Early Stopping 인스턴스 생성
# -------------------------------------
early_stopping = EarlyStopping(patience=5, verbose=True, path='best_model.pth')

# -------------------------------------
# 6) Training Loop with Early Stopping
# -------------------------------------
import matplotlib.pyplot as plt
from tqdm import tqdm

train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

total_start_time = time.time()

for epoch in range(num_epochs):
    start_time = time.time()  # 에포크 시작 시간 기록
    
    # -----------------
    # Train Loop
    # -----------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    
    train_acc_history.append(epoch_acc)
    train_loss_history.append(epoch_loss)

    # -----------------
    # Validation Loop
    # -----------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            v_loss = criterion(val_outputs, val_labels)
            val_loss += v_loss.item() * val_images.size(0)
            _, v_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(v_preds == val_labels).item()
            val_total += val_labels.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc  = val_correct / val_total
    
    val_acc_history.append(val_epoch_acc)
    val_loss_history.append(val_epoch_loss)

    # -----------------
    # Early Stopping 체크
    # -----------------
    early_stopping(val_epoch_loss, model)

    end_time = time.time()  # 에포크 종료 시간 기록
    epoch_duration = end_time - start_time  # 에포크 소요 시간 계산

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f} | "
          f"Time: {epoch_duration:.2f} sec\n")

    if early_stopping.early_stop:
        print("Early stopping triggered. Training stopped.")
        break

total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"Total Training Time: {total_duration/60:.2f} minutes")

# 모델 학습과정 그래프로 나타내기
epochs = range(1, len(train_acc_history) + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(epochs, train_acc_history, 'bo-', label='Training Accuracy')
ax[0].plot(epochs, val_acc_history, 'ro-', label='Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

ax[1].plot(epochs, train_loss_history, 'b-o', label='Training Loss')
ax[1].plot(epochs, val_loss_history, 'r-o', label='Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')

plt.show()

# -------------------------------------
# 7) Model Save
# -------------------------------------
# best_model.pth 에 이미 최적 모델이 저장되었으므로, 추가로 현재 모델 상태를 저장할 수 있습니다.
torch.save(model.state_dict(), "final_model.pth")

# -------------------------------------
# 8) Test
# -------------------------------------
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)  
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

avg_loss = test_loss / total
accuracy = correct / total

print(f"[Test] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
