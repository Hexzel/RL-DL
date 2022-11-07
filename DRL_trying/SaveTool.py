import torch.nn as nn
import torch

def StructToNetwork(struct):
    """Transform list of functions into nn.Module"""
    
    layers = []
    for layer in struct:
        layer_type = layer[0].lower()
        if layer_type == "linear":
            layers.append(nn.Linear(*layer[1]))
        elif layer_type == "relu":
            layers.append(nn.ReLU())
        elif layer_type == "dropout":
            layers.append(nn.Dropout(*layer[1]))
        elif layer_type == "logsoftmax":
            layers.append(nn.Dropout(*layer[1]))
        else:
            raise Exception("unknown function: " + layer[0])
    
    return nn.Sequential(*layers)

def SaveModel(network, filename = "temp.pth"):
    """Save Model(network, filename)"""
    
    checkpoint = {
        "struct": network.struct,
        "state_dict": network.state_dict()
    }
    torch.save(checkpoint, filename)
    

def main():
    class Pi(nn.Module):
        
        def __init__(self, in_dim = 1, out_dim = 1, struct = None):
            super(Pi, self).__init__()
            
            if (struct == None):
                self.struct = [
                    ("Linear", (in_dim, 128)),
                    ("ReLU",),
                    ("Dropout", (0.2,)),
                    ("Linear", (128, 64)),
                    ("ReLU",),
                    ("Dropout", (0.2,)),
                    ("Linear", (64, out_dim)),
                    ("LogSoftmax", (1, ))
                ]
            else:
                self.struct = struct
            
                    
            self.model = StructToNetwork(self.struct)
    
            
        def forward(self, x):
            pdparam = self.model(x)
            return pdparam
    
    
    
    pi = Pi(2, 4)
    SaveModel(pi, 'temp.pth')
    policy_info = torch.load('temp.pth')
    for key, _ in policy_info.items():
        print(key)
    pi2 = Pi(struct = policy_info['struct'])
    pi2.load_state_dict(policy_info['state_dict'])
    print(*pi.model.state_dict())
    print(*pi2.model.state_dict())
    print(pi.struct)

if __name__ == "__main__":
    main()
