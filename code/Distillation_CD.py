
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

# Set random seed to ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameter settings
num_samples = 100000    # Number of data rows
num_features = 10       # Number of features
batch_size = 512        # Mini-batch size
epochs_teacher = 5      # Number of training epochs for the teacher model
epochs_student = 10     # Number of training epochs for the student model
learning_rate = 0.01    # Learning rate
T = 2.0                 # Temperature parameter (used for probability smoothing)
alpha = 0.5             # Weight of hard labels and soft labels in traditional distillation (range 0~1)
# For step-by-step distillation, beta is the weight of "reasoning process loss"
beta = 0.3           

csv_file = "distdata.csv"

# Read data from CSV
df = pd.read_csv(csv_file)

# Extract features and labels
X_df = df.drop(columns=["label"]).values.astype(np.float32)
y_df = df["label"].values.astype(np.float32)

# Convert to torch tensors
X = torch.tensor(X_df)
y = torch.tensor(y_df)

# Construct DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------
# 2. Define a Simple Logistic Regression Model
# ---------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # Using a fully connected layer to simulate logistic regression
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Return raw scores (logits), apply sigmoid later to get probabilities
        return self.linear(x)

# ---------------------------
# 3. Train Teacher Model
# ---------------------------
def train_teacher(model, dataloader, epochs, lr):
    model.train()
    criterion = nn.BCEWithLogitsLoss()  # Built-in binary cross-entropy loss (with internal sigmoid)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)  # shape: (batch_size)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Teacher Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

# ---------------------------
# 4. Generate Teacher Model's Soft Labels (Traditional Distillation)
# ---------------------------
def generate_teacher_outputs(model, dataloader, temperature):
    model.eval()
    teacher_logits_all = []
    teacher_probs_all = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            logits = model(inputs).squeeze(1)  # Raw logits
            teacher_logits_all.append(logits)
            # Soft labels: Apply temperature T for smoothing (divide by T first, then apply sigmoid)
            soft_probs = torch.sigmoid(logits / temperature)
            teacher_probs_all.append(soft_probs)
    teacher_logits = torch.cat(teacher_logits_all)
    teacher_probs = torch.cat(teacher_probs_all)
    return teacher_logits, teacher_probs

# ---------------------------
# 5. Traditional Distillation: Train Student Model
# ---------------------------
def train_student_traditional(teacher_probs_all, model_teacher, model_student, dataloader, epochs, lr, temperature, alpha):
    model_student.train()
    optimizer = optim.SGD(model_student.parameters(), lr=lr)
    # Define hard label loss (standard binary cross-entropy)
    hard_criterion = nn.BCEWithLogitsLoss()
    
    # Define KL divergence function for binary classification
    def kd_loss(student_logits, teacher_probs):
        # Compute student probabilities with temperature scaling
        student_probs = torch.sigmoid(student_logits / temperature)
        # Compute KL divergence: p_teacher * log(p_teacher / p_student) + (1 - p_teacher) * log((1 - p_teacher) / (1 - p_student))
        # To prevent log(0), add a small value eps
        eps = 1e-7
        student_probs = torch.clamp(student_probs, eps, 1 - eps)
        teacher_probs = torch.clamp(teacher_probs, eps, 1 - eps)
        kl = teacher_probs * torch.log(teacher_probs / student_probs) + (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
        return torch.mean(kl)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            student_logits = model_student(inputs).squeeze(1)
            # Compute hard label loss: student model directly predicts (without temperature scaling)
            loss_hard = hard_criterion(student_logits, targets)
            # Compute distillation loss: use soft labels from teacher with temperature T
            # Assume the data order is consistent for simplicity
            # Note: In practice, additional alignment may be required; here we directly use precomputed teacher outputs
            # Extract corresponding soft teacher probabilities using the order of dataloader
            # Simulate extracting soft teacher labels using batch index and batch size
            start_idx = batch_idx * batch_size
            end_idx = start_idx + inputs.size(0)
            teacher_soft = teacher_probs_all[start_idx:end_idx].to(inputs.device)
            
            loss_kd = kd_loss(student_logits, teacher_soft)
            
            # Total loss: Weighted sum of hard label loss and distillation loss
            loss = alpha * loss_hard + (1 - alpha) * (temperature**2) * loss_kd
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Print intermediate results for each batch (for educational purposes)
            if batch_idx % 200 == 0:
                print(f"Traditional Distillation: Epoch {epoch+1} Batch {batch_idx}, "
                      f"Loss_hard: {loss_hard.item():.4f}, Loss_KD: {loss_kd.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Traditional Distillation: Epoch [{epoch+1}/{epochs}], Total Loss: {avg_loss:.4f}")
    
    return model_student

# ---------------------------
# 6. Step-by-Step Distillation: Train Student Model (Learn both Answers and Reasoning Process)
# ---------------------------
def train_student_step_by_step(teacher_logits_all, teacher_probs_all, model_teacher, model_student, dataloader, epochs, lr, temperature, alpha, beta):
    model_student.train()
    optimizer = optim.SGD(model_student.parameters(), lr=lr)
    # Define hard label loss (Standard Binary Cross-Entropy)
    hard_criterion = nn.BCEWithLogitsLoss()
    # Define reasoning process loss (Mean Squared Error, mimicking teacher’s linear scores)
    mse_loss = nn.MSELoss()
    
    def kd_loss(student_logits, teacher_probs):
        student_probs = torch.sigmoid(student_logits / temperature)
        eps = 1e-7
        student_probs = torch.clamp(student_probs, eps, 1 - eps)
        teacher_probs = torch.clamp(teacher_probs, eps, 1 - eps)
        kl = teacher_probs * torch.log(teacher_probs / student_probs) + (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
        return torch.mean(kl)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            student_logits = model_student(inputs).squeeze(1)
            # Compute hard label loss
            loss_hard = hard_criterion(student_logits, targets)
            # Compute distillation loss (soft label matching)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + inputs.size(0)
            teacher_soft = teacher_probs_all[start_idx:end_idx].to(inputs.device)
            loss_kd = kd_loss(student_logits, teacher_soft)
            # Compute reasoning process loss (MSE, match teacher and student linear scores)
            teacher_logits_batch = teacher_logits_all[start_idx:end_idx].to(inputs.device)
            loss_rationale = mse_loss(student_logits, teacher_logits_batch)
            
            # Total loss: Combine hard label loss, distillation loss, and reasoning process loss
            loss = alpha * loss_hard + beta * loss_rationale + (1 - alpha - beta) * (temperature**2) * loss_kd
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Print intermediate results for each batch (for educational purposes)
            if batch_idx % 200 == 0:
                print(f"Step-by-Step Distillation: Epoch {epoch+1} Batch {batch_idx}, "
                      f"Loss_hard: {loss_hard.item():.4f}, Loss_rationale: {loss_rationale.item():.4f}, Loss_KD: {loss_kd.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Step-by-Step Distillation: Epoch [{epoch+1}/{epochs}], Total Loss: {avg_loss:.4f}")
    
    return model_student

# ---------------------------
# 7. Main Function: Execute Teacher Training, Generate Teacher Outputs, and Train Two Student Models Separately
# ---------------------------
if __name__ == "__main__":
    # Train Teacher Model
    print("=== Training Teacher Model ===")
    teacher_model = LogisticRegressionModel(num_features)
    teacher_model = train_teacher(teacher_model, data_loader, epochs_teacher, learning_rate)
    
    # Generate Teacher Model's Intermediate Outputs (logits) and Soft Labels (smoothed with temperature T)
    teacher_logits_all, teacher_probs_all = generate_teacher_outputs(teacher_model, data_loader, T)
    print("Teacher model generated soft label samples:", teacher_probs_all[:5])
    
    # Copy soft labels into numpy arrays for indexing (keep data order consistent with DataLoader)
    teacher_probs_all = teacher_probs_all.cpu()
    teacher_logits_all = teacher_logits_all.cpu()
    
    # Train Student Model via Traditional Distillation
    print("\n=== Training Student Model via Traditional Distillation ===")
    student_model_traditional = LogisticRegressionModel(num_features)
    student_model_traditional = train_student_traditional(teacher_probs_all, teacher_model, student_model_traditional,
                                                          data_loader, epochs_student, learning_rate, T, alpha)
    
    # Train Student Model via Step-by-Step Distillation (Learn both Predictions and Reasoning Process)
    print("\n=== Training Student Model via Step-by-Step Distillation ===")
    student_model_step_by_step = LogisticRegressionModel(num_features)
    student_model_step_by_step = train_student_step_by_step(teacher_logits_all, teacher_probs_all, teacher_model,
                                                            student_model_step_by_step, data_loader, epochs_student, 
                                                            learning_rate, T, alpha, beta)
    
    # Test Model Predictions on Training Data (Educational Example)
    student_model_traditional.eval()
    student_model_step_by_step.eval()
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(data_loader))
        # Traditional Distillation Student Model Predictions
        logits_trad = student_model_traditional(sample_inputs).squeeze(1)
        preds_trad = torch.sigmoid(logits_trad)
        # Step-by-Step Distillation Student Model Predictions
        logits_step = student_model_step_by_step(sample_inputs).squeeze(1)
        preds_step = torch.sigmoid(logits_step)
        print("\nTraditional Distillation Student Model Predictions (First 10 Samples):")
        for i in range(10):
            print(f"Sample {i+1}: True Label: {sample_targets[i].item():.0f}, Student Prediction Probability: {preds_trad[i].item():.4f}")
        print("\nStep-by-Step Distillation Student Model Predictions (First 10 Samples):")
        for i in range(10):
            print(f"Sample {i+1}: True Label: {sample_targets[i].item():.0f}, Student Prediction Probability: {preds_step[i].item():.4f}")

'''
=== Training Teacher Model ===
Teacher Epoch [1/5], Loss: 0.6479
Teacher Epoch [2/5], Loss: 0.4656
Teacher Epoch [3/5], Loss: 0.3797
Teacher Epoch [4/5], Loss: 0.3307
Teacher Epoch [5/5], Loss: 0.2986
Teacher model generated soft label samples: tensor([0.7161, 0.5196, 0.5431, 0.3294, 0.7069])

=== Training Student Model via Traditional Distillation ===
Traditional Distillation: Epoch 1 Batch 0, Loss_hard: 0.7564, Loss_KD: 0.0817
Traditional Distillation: Epoch [1/10], Total Loss: 0.4871
Traditional Distillation: Epoch 2 Batch 0, Loss_hard: 0.5959, Loss_KD: 0.0743
Traditional Distillation: Epoch [2/10], Total Loss: 0.4253
Traditional Distillation: Epoch 3 Batch 0, Loss_hard: 0.5078, Loss_KD: 0.0750
Traditional Distillation: Epoch [3/10], Total Loss: 0.4000
Traditional Distillation: Epoch 4 Batch 0, Loss_hard: 0.4701, Loss_KD: 0.0767
Traditional Distillation: Epoch [4/10], Total Loss: 0.3901
Traditional Distillation: Epoch 5 Batch 0, Loss_hard: 0.4498, Loss_KD: 0.0786
Traditional Distillation: Epoch [5/10], Total Loss: 0.3850
Traditional Distillation: Epoch 6 Batch 0, Loss_hard: 0.4284, Loss_KD: 0.0803
Traditional Distillation: Epoch [6/10], Total Loss: 0.3832
Traditional Distillation: Epoch 7 Batch 0, Loss_hard: 0.4272, Loss_KD: 0.0866
Traditional Distillation: Epoch [7/10], Total Loss: 0.3812
Traditional Distillation: Epoch 8 Batch 0, Loss_hard: 0.4241, Loss_KD: 0.0796
Traditional Distillation: Epoch [8/10], Total Loss: 0.3817
Traditional Distillation: Epoch 9 Batch 0, Loss_hard: 0.4189, Loss_KD: 0.0800
Traditional Distillation: Epoch [9/10], Total Loss: 0.3815
Traditional Distillation: Epoch 10 Batch 0, Loss_hard: 0.4107, Loss_KD: 0.0850
Traditional Distillation: Epoch [10/10], Total Loss: 0.3806

=== Training Student Model via Step-by-Step Distillation ===
Step-by-Step Distillation: Epoch 1 Batch 0, Loss_hard: 0.7596, Loss_rationale: 4.3996, Loss_KD: 0.1113
Step-by-Step Distillation: Epoch [1/10], Total Loss: 1.3587
Step-by-Step Distillation: Epoch 2 Batch 0, Loss_hard: 0.5808, Loss_rationale: 2.8527, Loss_KD: 0.0691
Step-by-Step Distillation: Epoch [2/10], Total Loss: 1.1989
Step-by-Step Distillation: Epoch 3 Batch 0, Loss_hard: 0.5547, Loss_rationale: 2.7272, Loss_KD: 0.0666
Step-by-Step Distillation: Epoch [3/10], Total Loss: 1.1897
Step-by-Step Distillation: Epoch 4 Batch 0, Loss_hard: 0.5157, Loss_rationale: 2.7567, Loss_KD: 0.0678
Step-by-Step Distillation: Epoch [4/10], Total Loss: 1.1887
Step-by-Step Distillation: Epoch 5 Batch 0, Loss_hard: 0.5255, Loss_rationale: 2.7923, Loss_KD: 0.0689
Step-by-Step Distillation: Epoch [5/10], Total Loss: 1.1903
Step-by-Step Distillation: Epoch 6 Batch 0, Loss_hard: 0.5547, Loss_rationale: 2.7913, Loss_KD: 0.0686
Step-by-Step Distillation: Epoch [6/10], Total Loss: 1.1894
Step-by-Step Distillation: Epoch 7 Batch 0, Loss_hard: 0.5565, Loss_rationale: 2.8356, Loss_KD: 0.0696
Step-by-Step Distillation: Epoch [7/10], Total Loss: 1.1901
Step-by-Step Distillation: Epoch 8 Batch 0, Loss_hard: 0.5503, Loss_rationale: 2.7623, Loss_KD: 0.0679
Step-by-Step Distillation: Epoch [8/10], Total Loss: 1.1894
Step-by-Step Distillation: Epoch 9 Batch 0, Loss_hard: 0.5216, Loss_rationale: 2.7686, Loss_KD: 0.0680
Step-by-Step Distillation: Epoch [9/10], Total Loss: 1.1906
Step-by-Step Distillation: Epoch 10 Batch 0, Loss_hard: 0.5560, Loss_rationale: 2.7936, Loss_KD: 0.0687
Step-by-Step Distillation: Epoch [10/10], Total Loss: 1.1900

Traditional Distillation Student Model Predictions (First 10 Samples):
Sample 1: True Label: 1, Student Prediction Probability: 0.4223
Sample 2: True Label: 1, Student Prediction Probability: 0.6394
Sample 3: True Label: 0, Student Prediction Probability: 0.1353
Sample 4: True Label: 0, Student Prediction Probability: 0.2591
Sample 5: True Label: 0, Student Prediction Probability: 0.3090
Sample 6: True Label: 1, Student Prediction Probability: 0.4448
Sample 7: True Label: 0, Student Prediction Probability: 0.2589
Sample 8: True Label: 0, Student Prediction Probability: 0.3109
Sample 9: True Label: 1, Student Prediction Probability: 0.5740
Sample 10: True Label: 1, Student Prediction Probability: 0.9059

Step-by-Step Distillation Student Model Predictions (First 10 Samples):
Sample 1: True Label: 1, Student Prediction Probability: 0.3457
Sample 2: True Label: 1, Student Prediction Probability: 0.4065
Sample 3: True Label: 0, Student Prediction Probability: 0.2526
Sample 4: True Label: 0, Student Prediction Probability: 0.3005
Sample 5: True Label: 0, Student Prediction Probability: 0.3160
Sample 6: True Label: 1, Student Prediction Probability: 0.3572
Sample 7: True Label: 0, Student Prediction Probability: 0.2987
Sample 8: True Label: 0, Student Prediction Probability: 0.3147
Sample 9: True Label: 1, Student Prediction Probability: 0.3898
Sample 10: True Label: 1, Student Prediction Probability: 0.5297

'''

########update###########

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import math


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
num_samples = 100000    # Number of samples
num_features = 10       # Number of features
batch_size = 512        # Mini-batch size
epochs_teacher = 5      # Teacher model training epochs
epochs_student = 10     # Student model training epochs
learning_rate = 0.01    # Learning rate for teacher and traditional student
student_lr = 0.005      # Learning rate for step-by-step student
T = 2.0                 # Temperature parameter (for smoothing probabilities)
alpha = 0.5             # Weight for hard label loss in distillation
beta_initial = 0.1      # Maximum beta value for rationale loss (reduced from 0.2)
consistency_lambda = 0.25  # Increased weight for consistency regularization
noise_std = 0.04         # Increased noise magnitude for consistency regularization
curriculum_epochs = 8    # Ramp-up period for rationale loss (extended to 8 epochs)
rationale_margin = 0.1   # Margin for cosine similarity loss

# Read data from CSV
df = pd.read_csv(csv_file)

# Extract features and labels
X_df = df.drop(columns=["label"]).values.astype(np.float32)
y_df = df["label"].values.astype(np.float32)

# Convert to torch tensors
X = torch.tensor(X_df)
y = torch.tensor(y_df)

# Construct DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# 2. Define a Simple Logistic Regression Model
# ---------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Return raw logits; apply sigmoid later for probabilities
        return self.linear(x)

# ---------------------------
# 3. Train Teacher Model
# ---------------------------
def train_teacher(model, dataloader, epochs, lr):
    model.train()
    criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid internally
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Teacher Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

# ---------------------------
# 4. Generate Teacher Outputs (Soft Labels)
# ---------------------------
def generate_teacher_outputs(model, dataloader, temperature):
    model.eval()
    teacher_logits_all = []
    teacher_probs_all = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            logits = model(inputs).squeeze(1)
            teacher_logits_all.append(logits)
            soft_probs = torch.sigmoid(logits / temperature)
            teacher_probs_all.append(soft_probs)
    teacher_logits = torch.cat(teacher_logits_all)
    teacher_probs = torch.cat(teacher_probs_all)
    return teacher_logits, teacher_probs

# ---------------------------
# 5. Traditional Distillation: Train Student Model
# ---------------------------
def train_student_traditional(teacher_probs_all, model_teacher, model_student, dataloader, epochs, lr, temperature, alpha):
    model_student.train()
    optimizer = optim.SGD(model_student.parameters(), lr=lr)
    hard_criterion = nn.BCEWithLogitsLoss()
    
    def kd_loss(student_logits, teacher_probs):
        student_probs = torch.sigmoid(student_logits / temperature)
        eps = 1e-7
        student_probs = torch.clamp(student_probs, eps, 1 - eps)
        teacher_probs = torch.clamp(teacher_probs, eps, 1 - eps)
        kl = teacher_probs * torch.log(teacher_probs / student_probs) + \
             (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
        return torch.mean(kl)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            student_logits = model_student(inputs).squeeze(1)
            loss_hard = hard_criterion(student_logits, targets)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + inputs.size(0)
            teacher_soft = teacher_probs_all[start_idx:end_idx].to(inputs.device)
            loss_kd = kd_loss(student_logits, teacher_soft)
            loss = alpha * loss_hard + (1 - alpha) * (temperature**2) * loss_kd
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Traditional Distillation: Epoch {epoch+1} Batch {batch_idx}, "
                      f"Loss_hard: {loss_hard.item():.4f}, Loss_KD: {loss_kd.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Traditional Distillation: Epoch [{epoch+1}/{epochs}], Total Loss: {avg_loss:.4f}")
    
    return model_student

# ---------------------------
# 6. Step-by-Step Distillation: Train Student Model 
#     (with Adaptive Rationale Weighting using Linear Ramp-Up, Margin-based Cosine Rationale Loss, 
#      Consistency Regularization, and Learning Rate Scheduling)
# ---------------------------
def train_student_step_by_step(teacher_logits_all, teacher_probs_all, model_teacher, model_student, dataloader, epochs, lr, temperature, alpha, beta_initial, consistency_lambda):
    model_student.train()
    optimizer = optim.Adam(model_student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    hard_criterion = nn.BCEWithLogitsLoss()
    
    def kd_loss(student_logits, teacher_probs):
        student_probs = torch.sigmoid(student_logits / temperature)
        eps = 1e-7
        student_probs = torch.clamp(student_probs, eps, 1 - eps)
        teacher_probs = torch.clamp(teacher_probs, eps, 1 - eps)
        kl = teacher_probs * torch.log(teacher_probs / student_probs) + \
             (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
        return torch.mean(kl)
    
    # Margin-based cosine similarity loss for rationale:
    cosine_loss = nn.CosineEmbeddingLoss(margin=rationale_margin)
    
    # Linear ramp-up for current_beta over curriculum_epochs
    def rampup(epoch, max_epochs, max_beta):
        return max_beta * min(1.0, epoch / max_epochs)
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        current_beta = rampup(epoch, curriculum_epochs, beta_initial)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            student_logits = model_student(inputs).squeeze(1)
            loss_hard = hard_criterion(student_logits, targets)
            
            start_idx = batch_idx * batch_size
            end_idx = start_idx + inputs.size(0)
            teacher_soft = teacher_probs_all[start_idx:end_idx].to(inputs.device)
            loss_kd = kd_loss(student_logits, teacher_soft)
            
            teacher_logits_batch = teacher_logits_all[start_idx:end_idx].to(inputs.device)
            # For cosine embedding loss, we need to reshape to (batch_size, 1) and target of 1
            cos_target = torch.ones(student_logits.size()).to(inputs.device)
            loss_rationale = cosine_loss(student_logits.unsqueeze(1), teacher_logits_batch.unsqueeze(1), cos_target)
            
            # Consistency Regularization: perturb inputs and enforce similar predictions
            noise = torch.randn_like(inputs) * noise_std
            student_logits_perturbed = model_student(inputs + noise).squeeze(1)
            # Here, we use KL divergence between the predictions (after sigmoid)
            preds = torch.sigmoid(student_logits)
            preds_perturbed = torch.sigmoid(student_logits_perturbed)
            eps = 1e-7
            preds = torch.clamp(preds, eps, 1 - eps)
            preds_perturbed = torch.clamp(preds_perturbed, eps, 1 - eps)
            loss_consistency = torch.mean(preds * torch.log(preds / preds_perturbed) +
                                            (1 - preds) * torch.log((1 - preds) / (1 - preds_perturbed)))
            
            loss_total = (alpha * loss_hard +
                          current_beta * loss_rationale +
                          (1 - alpha - current_beta) * (temperature**2) * loss_kd +
                          consistency_lambda * loss_consistency)
            
            loss_total.backward()
            optimizer.step()
            running_loss += loss_total.item()
            
            if batch_idx % 200 == 0:
                print(f"Step-by-Step Distillation: Epoch {epoch+1} Batch {batch_idx}, "
                      f"Loss_hard: {loss_hard.item():.4f}, Loss_rationale: {loss_rationale.item():.4f}, "
                      f"Loss_KD: {loss_kd.item():.4f}, Loss_consistency: {loss_consistency.item():.4f}, "
                      f"Current_beta: {current_beta:.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Step-by-Step Distillation: Epoch [{epoch+1}/{epochs}], Total Loss: {avg_loss:.4f}")
        
        # Step learning rate scheduler
        scheduler.step(avg_loss)
    
    return model_student

# ---------------------------
# 7. Main: Train Teacher, Generate Outputs, and Train Student Models
# ---------------------------
if __name__ == "__main__":
    # Train teacher model
    print("=== Training Teacher Model ===")
    teacher_model = LogisticRegressionModel(num_features)
    teacher_model = train_teacher(teacher_model, data_loader, epochs_teacher, learning_rate)
    
    # Generate teacher outputs (logits and soft labels) using temperature scaling
    teacher_logits_all, teacher_probs_all = generate_teacher_outputs(teacher_model, data_loader, T)
    print("Teacher model soft label samples:", teacher_probs_all[:5])
    
    # Ensure teacher outputs are on CPU for consistent indexing
    teacher_probs_all = teacher_probs_all.cpu()
    teacher_logits_all = teacher_logits_all.cpu()
    
    # Traditional distillation training for student model
    print("\n=== Training Student Model via Traditional Distillation ===")
    student_model_traditional = LogisticRegressionModel(num_features)
    student_model_traditional = train_student_traditional(teacher_probs_all, teacher_model, student_model_traditional,
                                                          data_loader, epochs_student, learning_rate, T, alpha)
    
    # Step-by-Step distillation training for student model 
    # (with adaptive rationale loss weight (linear ramp-up over 8 epochs), margin-based cosine rationale loss,
    #  consistency regularization, and learning rate scheduling)
    print("\n=== Training Student Model via Step-by-Step Distillation ===")
    student_model_step_by_step = LogisticRegressionModel(num_features)
    student_model_step_by_step = train_student_step_by_step(teacher_logits_all, teacher_probs_all, teacher_model,
                                                            student_model_step_by_step, data_loader, epochs_student, 
                                                            student_lr, T, alpha, beta_initial, consistency_lambda)
    
    # Testing: Display predictions for some samples from each student model
    student_model_traditional.eval()
    student_model_step_by_step.eval()
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(data_loader))
        logits_trad = student_model_traditional(sample_inputs).squeeze(1)
        preds_trad = torch.sigmoid(logits_trad)
        logits_step = student_model_step_by_step(sample_inputs).squeeze(1)
        preds_step = torch.sigmoid(logits_step)
        print("\nTraditional Distillation Student Predictions (first 10 samples):")
        for i in range(10):
            print(f"Sample {i+1}: True Label: {sample_targets[i].item():.0f}, Prediction: {preds_trad[i].item():.4f}")
        print("\nStep-by-Step Distillation Student Predictions (first 10 samples):")
        for i in range(10):
            print(f"Sample {i+1}: True Label: {sample_targets[i].item():.0f}, Prediction: {preds_step[i].item():.4f}")

'''
=== Training Teacher Model ===
Teacher Epoch [1/5], Loss: 0.6479
Teacher Epoch [2/5], Loss: 0.4656
Teacher Epoch [3/5], Loss: 0.3797
Teacher Epoch [4/5], Loss: 0.3307
Teacher Epoch [5/5], Loss: 0.2986
Teacher model soft label samples: tensor([0.7161, 0.5196, 0.5431, 0.3294, 0.7069])

=== Training Student Model via Traditional Distillation ===
Traditional Distillation: Epoch 1 Batch 0, Loss_hard: 0.7564, Loss_KD: 0.0817
Traditional Distillation: Epoch [1/10], Total Loss: 0.4871
Traditional Distillation: Epoch 2 Batch 0, Loss_hard: 0.5959, Loss_KD: 0.0743
Traditional Distillation: Epoch [2/10], Total Loss: 0.4253
Traditional Distillation: Epoch 3 Batch 0, Loss_hard: 0.5078, Loss_KD: 0.0750
Traditional Distillation: Epoch [3/10], Total Loss: 0.4000
Traditional Distillation: Epoch 4 Batch 0, Loss_hard: 0.4701, Loss_KD: 0.0767
Traditional Distillation: Epoch [4/10], Total Loss: 0.3901
Traditional Distillation: Epoch 5 Batch 0, Loss_hard: 0.4498, Loss_KD: 0.0786
Traditional Distillation: Epoch [5/10], Total Loss: 0.3850
Traditional Distillation: Epoch 6 Batch 0, Loss_hard: 0.4284, Loss_KD: 0.0803
Traditional Distillation: Epoch [6/10], Total Loss: 0.3832
Traditional Distillation: Epoch 7 Batch 0, Loss_hard: 0.4272, Loss_KD: 0.0866
Traditional Distillation: Epoch [7/10], Total Loss: 0.3812
Traditional Distillation: Epoch 8 Batch 0, Loss_hard: 0.4241, Loss_KD: 0.0796
Traditional Distillation: Epoch [8/10], Total Loss: 0.3817
Traditional Distillation: Epoch 9 Batch 0, Loss_hard: 0.4189, Loss_KD: 0.0800
Traditional Distillation: Epoch [9/10], Total Loss: 0.3815
Traditional Distillation: Epoch 10 Batch 0, Loss_hard: 0.4107, Loss_KD: 0.0850
Traditional Distillation: Epoch [10/10], Total Loss: 0.3806

=== Training Student Model via Step-by-Step Distillation ===
Step-by-Step Distillation: Epoch 1 Batch 0, Loss_hard: 0.7596, Loss_rationale: 1.2266, Loss_KD: 0.1113, Loss_consistency: 0.0001, Current_beta: 0.0000
C:\anaconda3\envs\graphrag-env\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
Step-by-Step Distillation: Epoch [1/10], Total Loss: 0.4424
Step-by-Step Distillation: Epoch 2 Batch 0, Loss_hard: 0.4219, Loss_rationale: 0.8125, Loss_KD: 0.0872, Loss_consistency: 0.0001, Current_beta: 0.0125
Step-by-Step Distillation: Epoch [2/10], Total Loss: 0.3904
Step-by-Step Distillation: Epoch 3 Batch 0, Loss_hard: 0.4134, Loss_rationale: 0.6992, Loss_KD: 0.0852, Loss_consistency: 0.0001, Current_beta: 0.0250
Step-by-Step Distillation: Epoch [3/10], Total Loss: 0.3926
Step-by-Step Distillation: Epoch 4 Batch 0, Loss_hard: 0.4184, Loss_rationale: 0.8281, Loss_KD: 0.0948, Loss_consistency: 0.0001, Current_beta: 0.0375
Step-by-Step Distillation: Epoch [4/10], Total Loss: 0.3978
Step-by-Step Distillation: Epoch 5 Batch 0, Loss_hard: 0.4350, Loss_rationale: 0.8164, Loss_KD: 0.0943, Loss_consistency: 0.0001, Current_beta: 0.0500
Step-by-Step Distillation: Epoch [5/10], Total Loss: 0.4067
Step-by-Step Distillation: Epoch 6 Batch 0, Loss_hard: 0.4096, Loss_rationale: 0.7031, Loss_KD: 0.0909, Loss_consistency: 0.0001, Current_beta: 0.0625
Step-by-Step Distillation: Epoch [6/10], Total Loss: 0.4082
Step-by-Step Distillation: Epoch 7 Batch 0, Loss_hard: 0.3953, Loss_rationale: 0.7617, Loss_KD: 0.0897, Loss_consistency: 0.0001, Current_beta: 0.0750
Step-by-Step Distillation: Epoch [7/10], Total Loss: 0.4123
Step-by-Step Distillation: Epoch 8 Batch 0, Loss_hard: 0.3960, Loss_rationale: 0.6992, Loss_KD: 0.0921, Loss_consistency: 0.0001, Current_beta: 0.0875
Step-by-Step Distillation: Epoch [8/10], Total Loss: 0.4172
Step-by-Step Distillation: Epoch 9 Batch 0, Loss_hard: 0.3669, Loss_rationale: 0.8125, Loss_KD: 0.0996, Loss_consistency: 0.0001, Current_beta: 0.1000
Step-by-Step Distillation: Epoch [9/10], Total Loss: 0.4235
Step-by-Step Distillation: Epoch 10 Batch 0, Loss_hard: 0.3784, Loss_rationale: 0.7227, Loss_KD: 0.0924, Loss_consistency: 0.0002, Current_beta: 0.1000
Step-by-Step Distillation: Epoch [10/10], Total Loss: 0.4225

Traditional Distillation Student Predictions (first 10 samples):
Sample 1: True Label: 1, Prediction: 0.4491
Sample 2: True Label: 1, Prediction: 0.4116
Sample 3: True Label: 1, Prediction: 0.4878
Sample 4: True Label: 0, Prediction: 0.2175
Sample 5: True Label: 0, Prediction: 0.3120
Sample 6: True Label: 0, Prediction: 0.3411
Sample 7: True Label: 0, Prediction: 0.3244
Sample 8: True Label: 0, Prediction: 0.2410
Sample 9: True Label: 0, Prediction: 0.2335
Sample 10: True Label: 0, Prediction: 0.1799

Step-by-Step Distillation Student Predictions (first 10 samples):
Sample 1: True Label: 1, Prediction: 0.4554
Sample 2: True Label: 1, Prediction: 0.4121
Sample 3: True Label: 1, Prediction: 0.4988
Sample 4: True Label: 0, Prediction: 0.2007
Sample 5: True Label: 0, Prediction: 0.3034
Sample 6: True Label: 0, Prediction: 0.3335
Sample 7: True Label: 0, Prediction: 0.3173
Sample 8: True Label: 0, Prediction: 0.2282
Sample 9: True Label: 0, Prediction: 0.2112
Sample 10: True Label: 0, Prediction: 0.1615

'''


1.4463*4000
