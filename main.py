import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import json   # <--- added for reading user input from file

# ------------------------
# Globals (Scaler & Encoders)
# ------------------------
scaler = StandardScaler()
encoders = {}

# ------------------------
# 1. Load Dataset
# ------------------------
def load_dataset(path="Sleep_health_and_lifestyle_dataset.csv"):
    return pd.read_csv(path)

# ------------------------
# 2. Preprocess
# ------------------------
def preprocess(df, fit=True):
    if fit:
        # Bin Quality.of.Sleep into 3 categories
        bins = [0, 4, 7, 10]
        labels = [0, 1, 2]  # Poor, Average, Good
        df = df.dropna(subset=["Quality of Sleep"])
        df["sleep_quality_cat"] = pd.cut(
            df["Quality of Sleep"], bins=bins,
            labels=labels, include_lowest=True
        ).astype(int)

    # Encode categorical features
    for col in ["Gender", "Sleep Disorder"]:
        if fit:
            enc = LabelEncoder()
            # Ensure "None" category is always recognized
            df[col] = df[col].fillna("None")
            enc.fit(list(df[col].astype(str).values) + ["None"])
            df[col] = enc.transform(df[col].astype(str))
            encoders[col] = enc
        else:
            df[col] = df[col].fillna("None")
            df[col] = encoders[col].transform(df[col].astype(str))

    feature_cols = [
        "Sleep Duration", "Physical Activity Level", "Stress Level",
        "Heart Rate", "Daily Steps", "Gender", "Age", "Sleep Disorder"
    ]
    X = df[feature_cols].astype(float)

    # Scale
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if fit:
        y = df["sleep_quality_cat"].values.astype(int)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X_scaled, y))
        return (
            torch.tensor(X_scaled[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long),
            torch.tensor(X_scaled[test_idx], dtype=torch.float32),
            torch.tensor(y[test_idx], dtype=torch.long),
            X.shape[1]
        )
    else:
        return torch.tensor(X_scaled, dtype=torch.float32)

# ------------------------
# 3. Dataset Wrapper
# ------------------------
class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ------------------------
# 4. Model
# ------------------------
def build_model(input_dim, num_classes=3):
    return nn.Sequential(
        nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, num_classes)
    )

# ------------------------
# 5. Training Loop
# ------------------------
def train_and_evaluate(model, train_loader, test_loader, device="cpu", epochs=15, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_acc = 0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_acc += loss.item() * yb.size(0)
            total += yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}  Loss: {loss_acc/total:.4f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

# ------------------------
# 6. Evaluation
# ------------------------
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

# ------------------------
# 7. Save & Load
# ------------------------
def save_model(model, path="sleep_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path, input_dim, num_classes=3):
    model = build_model(input_dim, num_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ------------------------
# 8. Predict New User
# ------------------------
def predict_new_user(model, user_dict):
    df = pd.DataFrame([user_dict])
    X_new = preprocess(df, fit=False)  # reuse scaler + encoders
    with torch.no_grad():
        logits = model(X_new)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = torch.argmax(logits, dim=1).item()
    mapping = {0: "Poor (Abnormal)", 1: "Average", 2: "Good"}
    return mapping[pred], probs

# ------------------------
# 9. Main
# ------------------------
def main():
    df = load_dataset("Sleep_health_and_lifestyle_dataset.csv")
    X_train, y_train, X_test, y_test, input_dim = preprocess(df, fit=True)

    train_loader = DataLoader(SleepDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(SleepDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = build_model(input_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_and_evaluate(model, train_loader, test_loader, device)

    # Save model
    save_model(model, "sleep_model.pth")
    print("Model saved to sleep_model.pth")

    # Load & predict for new user from text file
    model = load_model("sleep_model.pth", input_dim)
    with open("new_user.txt", "r") as f:
        new_user = json.load(f)

    result, probs = predict_new_user(model, new_user)
    print("New User Prediction:", result)
    print("Class Probabilities [Poor, Average, Good]:", probs)


if __name__ == "__main__":
    main()
