
from utils import *
from doping_expert import *
from element_expert import *
from physics_expert import *

class GatingNetwork(nn.Module):
    def __init__(self, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(201, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_experts)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=1)
    
class MoECNN(nn.Module):
    def __init__(self, expert1,expert2,expert3):
        super(MoECNN, self).__init__()
        self.noise_std=0.3
        self.num_active=8
        self.load_factor=0.7
        self.experts_1 = nn.ModuleList(expert1)
        self.experts_2 = nn.ModuleList(expert2)
        self.experts_3 = nn.ModuleList(expert3)
        for param in self.experts_1.parameters():
            param.requires_grad = False
        for param in self.experts_2.parameters():
            param.requires_grad = False
        for param in self.experts_3.parameters():
            param.requires_grad = False

        self.gating_network = GatingNetwork(30)

    def forward(self, element,element_real,physic):
        gate_logits = self.gating_network(torch.cat([element,physic], dim=1))
        expert_outputs_1 = [expert(element) for expert in self.experts_1]
        expert_outputs_2 = [expert(element_real) for expert in self.experts_2]
        expert_outputs_3 = [expert(physic) for expert in self.experts_3]

        expert_outputs =  expert_outputs_1+expert_outputs_2+expert_outputs_3
        expert_outputs = torch.stack(expert_outputs, dim=2)


        noise = torch.randn_like(gate_logits) * self.noise_std
        gate_logits = gate_logits + noise

        topk, indices = torch.topk(gate_logits, self.num_active, dim=1)
        mask = torch.zeros_like(gate_logits).scatter_(1, indices, 1)
        gate_logits = gate_logits * mask + (mask - 1) * 1e9
        gate_weights = F.softmax(gate_logits, dim=1) # 
 
        gating_scores = gate_weights.unsqueeze(1).expand_as(expert_outputs)
        output = torch.sum(gating_scores * expert_outputs, dim=2)

        return output,gate_weights
    
    def loss(self, element,element_real,physic, y):
        output, gate_weights = self.forward(element,element_real,physic)
        mse_loss = F.mse_loss(output, y) 
        load = torch.mean(gate_weights, dim=0) # (num_experts,)

        penalty = torch.max(load - self.load_factor, torch.zeros_like(load)).pow(2)
        penalty = torch.sum(penalty)
        total_loss = mse_loss + penalty
        return total_loss

noise_std = 0.1 
load_factor = 0.9 
n_estimators = 10
n_epochs = 2500 
load=0
models = []
optimizers = []
lr_schedulers=[]

model_realnums=[]
model_physics=[]
model_dopings=[]
for i in range(n_estimators):
    model_realnum=torch.load(r'model/element_expert_{i}.pth'.format(i=i))
    model_realnums.append(model_realnum)
    model_physic=torch.load(r'model/physic_expert_{i}.pth'.format(i=i))
    model_physics.append(model_physic)
    model_doping=torch.load(r'model/doping_expert_{i}.pth'.format(i=i))
    model_dopings.append(model_doping)


model= MoECNN(model_dopings,model_realnums,model_physics)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.005, patience=500, verbose=True)




all_loss=[]
best_val_r2 = float('inf')
best_model = None
epoch=1000

for i in range(epoch):
    running_loss = 0.0
    preds = []
    cs = []
    model.train()
    for j, (element,element_real,physic, c) in enumerate(train_loader):
        optimizer.zero_grad()
        element = element.to(device)
        element_real = element_real.to(device)
        physic = physic.to(device)

        c = c.to(device)
        c=c.type(torch.float32)
        c=c.unsqueeze(1)
        output,score = model(element,element_real,physic)
        loss = model.loss(element,element_real,physic, c)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        all_loss.append(loss.item()/len(train_loader))
        preds += output.detach().cpu().numpy().tolist()
        cs += c.detach().cpu().numpy().tolist()

    train_r2 = r2_score(preds, cs)
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    if i%1==0:
        for j, (element,element_real,physic, c) in enumerate(val_loader):
            optimizer.zero_grad()
            element = element.to(device)
            element_real = element_real.to(device)
            physic = physic.to(device)

            c = c.to(device)
            c=c.type(torch.float32)
            c=c.unsqueeze(1)
            output,score = model(element,element_real,physic)
            val_preds += output.detach().cpu().numpy().tolist()
            val_labels += c.detach().cpu().numpy().tolist()
            loss = criterion(output, c)
            val_loss += loss.item() * element.size(0)


        val_r2 = r2_score(val_preds,val_labels)
        val_loss = val_loss / len(val_loader.dataset)
        lr_scheduler.step(val_loss)

        print(f'Epoch: {epoch+1},  Val Loss: {val_loss:.4f}')

        if val_loss < best_val_r2:
            best_val_r2 = val_loss
            best_model = model.state_dict()
            print('best model saved',val_loss)

        model.load_state_dict(best_model)
        print('Epoch: {}, Loss: {:.6f}, R2 Score: {:.6f},Predict_Tc：{:.3f}，Real_Tc：{:.3f}'.format(i+1, running_loss/len(train_loader), val_r2,output[0].item(), c[0].item()))
        print(train_r2)
        if i%10==0:
            print(score[0])
        if log:
            wandb.log({"train_r2": train_r2})
            wandb.log({"val_r2": val_r2})
