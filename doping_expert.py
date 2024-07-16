import torchvision.models.resnet as resnet
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

n_estimators = 10 
max_samples = 0.8 
n_epochs =1000 
load=0
models = []
optimizers = []
lr_schedulers=[]

class num2onehot(nn.Module):
    def __init__(self,):
        super(num2onehot, self).__init__()
        self.int_embedding = nn.Linear(1, 10)
        self.dec_embedding = nn.Linear(10, 10)

    def forward(self, x):
        int_part = x // 1
        dec_part = x % 1
        first_dec = ((dec_part) * 10) // 1
        secend_dec = (((dec_part) * 100) // 1)%10
        third_dec = (((dec_part) * 1000) // 1)%10

        int_part = int_part.to(device)
        int_part = int_part.unsqueeze(2)

        first_dec = first_dec.to(device).type(torch.int64)
        secend_dec = secend_dec.to(device).type(torch.int64)
        third_dec = third_dec.to(device).type(torch.int64)

        first_dec = torch.nn.functional.one_hot(first_dec, num_classes=10)
        secend_dec = torch.nn.functional.one_hot(secend_dec, num_classes=10)
        third_dec = torch.nn.functional.one_hot(third_dec, num_classes=10)


        first_dec = first_dec.to(device).type(torch.float32)
        secend_dec = secend_dec.to(device).type(torch.float32)
        third_dec = third_dec.to(device).type(torch.float32)

        int_embed = self.int_embedding(int_part)


        int_embed=int_embed.unsqueeze(1)
        first_dec_embed=first_dec.unsqueeze(1)
        secend_dec_embed=secend_dec.unsqueeze(1)
        third_dec_embed=third_dec.unsqueeze(1)
        input = torch.cat([int_embed, first_dec_embed, secend_dec_embed,third_dec_embed], dim=1)
        return input

class Net(nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.resnet = resnet.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(1000, num_classes)
        self.num2onehot=num2onehot()
        

    def forward(self, x):
        x=self.num2onehot(x)
        x = self.resnet(x)
        x = self.fc(x)
        return x


for i in range(n_estimators):
    model_realnum = Net().to(device)

    if load:
        model_realnum=torch.load(r'model/doping_expert{i}.pth'.format(i=i))
        print('model load successfully')

    models.append(model_realnum)

    optimizer = torch.optim.Adam(model_realnum.parameters(), lr=0.0001,weight_decay=0.0001)
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

            output = model(element)


            loss = criterion(output, c)
            loss.backward()
            optimizer.step()

            all_loss+=loss.item()
            preds += output.detach().cpu().numpy().tolist()
            cs += c.detach().cpu().numpy().tolist()
        train_r2 = r2_score(preds, cs)
        torch.save(model, 'model/doping_expert_{i}.pth'.format(i=i))
        print('estimators',i,'Epoch: {}, Loss: {:.6f}, R2 Score: {:.6f}'.format(epoch+1, all_loss/len(subset_dataloader), train_r2))
 

