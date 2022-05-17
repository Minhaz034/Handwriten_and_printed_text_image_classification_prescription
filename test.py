test_X.to(device)
test_y.to(device)

def test(net):
    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, H, W).to(device))[0]  # returns a list,
            predicted_class = torch.argmax(net_out)
            preds.append(predicted_class)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))
    return preds