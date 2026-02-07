
import torch
from cust_transformer import *
from data_util import *
from speech_dataset import *
import wandb
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



train_data = load_datasets("train")
val_data = load_datasets("val")
test_data = load_datasets("test")
print(train_data[0].keys())
mapping = {}
for data in train_data:
    if data["Annotators"]['label'] not in mapping:
        mapping[data["Annotators"]['label']] = 0
    mapping[data["Annotators"]['label']] += 1

for key in mapping:
    print(key, mapping[key])

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your tokenizer
train_dataset = SpeechDataset(train_data, tokenizer)
val_dataset = SpeechDataset(val_data, tokenizer)
test_dataset = SpeechDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

test_loader = DataLoader(test_dataset, batch_size=16)


# %%
model = TransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    num_classes=3,  # normal, offensive, hatespeech
    d_model=256,  # Adjust if necessary
    num_layers=6
)

model.to("mps")

# %%
import torch.nn as nn
from transformers import AdamW
from torch.optim.lr_scheduler import OneCycleLR

batch_size = 4
seq_length = 10
d_model = 16  # Embedding dimension
dim_feedforward = 64
num_layers = 3  # Number of Transformer encoder layers
dropout = 0.1

encoder = TransformerEncoder(
    num_layers=num_layers,
    d_model=d_model,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)
input_data = torch.rand(batch_size, seq_length, d_model)
print("Input data shape:", input_data.shape)

output = encoder(input_data)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-4)
num_epochs = 50
scheduler = OneCycleLR(
    optimizer, 
    max_lr=0.01,            # Maximum learning rate
    steps_per_epoch=len(train_loader), 
    epochs=num_epochs,      # Total number of epochs
    pct_start=0.4          # Percentage of the cycle to increase learning rate
)
device = "mps"


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (output.argmax(1) == label).sum().item()
        
    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

def eval_model(model, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        label = batch["label"].to(device)
        with torch.no_grad():
            output = model(input_ids)
            loss = criterion(output, label)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == label).sum().item()
    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

# %%

wandb.init(project="CSC246-transformer-hate-speech-recognition", name="iteration_50", anonymous="allow")

wandb.log({'learning_rate': 5e-4, 'batch_size': batch_size, 'num_epochs': num_epochs})
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "best_model.pth")

patience = 10 
best_val_loss = float('inf')
patience_counter = 0


for epoch in range(num_epochs):
    # Training phase
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc})
    print(f"Epoch {epoch+1} training datasets: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
    
    # Validation phase (every 5 epochs or at the end)
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        val_loss, val_acc = eval_model(model, val_loader)
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch+1} evaluation datasets: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at epoch {epoch+1} with Val Loss = {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
        
        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print(f"Training complete. Best model saved at: {model_save_path}")
# %%
# Evaluate the model on the test set
test_dataset = SpeechDataset(test_dataset, tokenizer)
test_loss, test_acc = eval_model(model, test_loader)
print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})



