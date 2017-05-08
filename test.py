from torch.utils.data import DataLoader

data_loader = DataLoader(dataset=dataset, num_workers=10, batch_size=128, shuffle=True)
data_iter = iter(data_loader)
