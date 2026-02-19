import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Starting training process...")
for epoch in range(100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if (epoch+1) % 10 == 0:
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/100] -> Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), "emotion_transformer_1533.pth")
print("Success")