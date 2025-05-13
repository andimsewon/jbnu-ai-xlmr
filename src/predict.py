import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import XLMRobertaClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict():
    test_df = pd.read_csv('../data/test.csv')
    test_texts = test_df['text'].values

    test_dataset = TextDataset(test_texts)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = XLMRobertaClassifier().to(device)
    model.load_state_dict(torch.load('../models/best_model.pt'))
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            ids, mask = batch['ids'].to(device), batch['mask'].to(device)
            outputs = model(ids, mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    submission = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    submission.to_csv('../output/submission.csv', index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    predict()
