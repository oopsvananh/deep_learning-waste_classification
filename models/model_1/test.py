import sys
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.preprocessing import make_dataloaders, CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Thiết bị:', device)
_, _, test_loader = make_dataloaders()

filename = Path(__file__).resolve().parent / 'best_model1.pt'
model = torch.load(filename, map_location=device, weights_only=False)
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        y_true += labels.tolist()
        y_pred += outputs.argmax(1).cpu().tolist()

print('Accuracy :', accuracy_score(y_true, y_pred))
print('Precision:', precision_score(y_true, y_pred, average='weighted'))
print('Recall   :', recall_score(y_true, y_pred, average='weighted'))
print('F1-score :', f1_score(y_true, y_pred, average='weighted'))
print('Macro F1 :', f1_score(y_true, y_pred, average='macro'))
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))