from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from utils import *


n_estimators = 10
max_samples = 0.8 
n_epochs = 1000
load=0

models = []
optimizers = []
lr_schedulers=[]


for i in range(n_estimators):
    model_realnum = Net2().to(device)
    if load:
        model_realnum=torch.load(r'model/physics_expert_{i}.pth'.format(i=i))
        print('model loaded')

    models.append(model_realnum)
    optimizer = torch.optim.Adam(model_realnum.parameters(), lr=0.001)
    optimizers.append(optimizer)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.005, patience=500, verbose=True)
    lr_schedulers.append(lr_scheduler)


for epoch in range(n_epochs):
    for i in range(n_estimators):
        indices = np.random.choice(len(train_dataset), int(len(train_dataset) * max_samples), replace=True)
        sampler = SubsetRandomSampler(indices)
        subset_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        all_loss = 0
        preds = []
        cs = []
        for j, (element,element_real,physic, c) in enumerate( subset_dataloader):
            model = models[i]
            lr_scheduler = lr_schedulers[i]
            optimizer = optimizers[i]
            model.train()
            optimizer.zero_grad()
            element_real = element_real.to(device)
            physic = physic.to(device)

            c = c.to(device)
            c=c.type(torch.float32)
            c=c.unsqueeze(1)
            output = model(physic)


            loss = criterion(output, c)
            loss.backward()
            optimizer.step()
            all_loss+=loss.item()
            preds += output.detach().cpu().numpy().tolist()
            cs += c.detach().cpu().numpy().tolist()
        train_r2 = r2_score(preds, cs)

        torch.save(model, 'model/physics_expert_{i}.pth'.format(i=i))
        print('estimators',i,'Epoch: {}, Loss: {:.6f}, R2 Score: {:.6f}'.format(epoch+1, all_loss/len(subset_dataloader), train_r2))


