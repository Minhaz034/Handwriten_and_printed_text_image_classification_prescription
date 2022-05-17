
PATH = "./models/saved_model"
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device=device)

preds = test(model)
for i in range(test_X.size()[0]):
  plt.imshow(test_X[i],cmap='gray')
  plt.title(f"actual: {torch.argmax(test_y[i]).to(device)} predicted: {preds[i]}")
  plt.show()