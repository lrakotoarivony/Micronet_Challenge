import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def training(train_loader, valid_loader, model, criterion, optimizer,n_epochs,scheduler,filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, valid_losses, train_acc, valid_acc  = [], [], [], []
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity
    
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0, 0 # monitor losses
        class_correct_train ,class_total_train = 0, 0 
        class_correct_valid ,class_total_valid = 0, 0 
        

        # train the model
        model.train() # prep model for training
        for data, label in train_loader:
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            optimizer.zero_grad() # clear the gradients of all optimized variables
            output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(output, label) # calculate the loss
            loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step() # perform a single optimization step (parameter update)
            train_loss += loss.item() * data.size(0) # update running training loss

            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(label.data.view_as(pred)))
            for i in range(len(label)):
                digit = label.data[i]
                class_correct_train += correct[i].item()
                class_total_train += 1
            

        # validate the model
        model.eval()
        for data, label in valid_loader:
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            with torch.no_grad():
                output = model(data)
            loss = criterion(output,label)
            valid_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(label.data.view_as(pred)))
            for i in range(len(label)):
                digit = label.data[i]
                class_correct_valid += correct[i].item()
                class_total_valid += 1
        



        # calculate average loss over an epoch
        train_loss /= len(train_loader.sampler)
        valid_loss /= len(valid_loader.sampler)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)


        train_acc.append(class_correct_train/class_total_train)
        valid_acc.append(class_correct_valid/class_total_valid)


        print('epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), filename)
            valid_loss_min = valid_loss
            
        scheduler.step()
        print('lr : {} for epochs : {}'.format(optimizer.param_groups[0]['lr'],epoch))

    return train_losses, valid_losses,  train_acc, valid_acc

def evaluation(model, test_loader, criterion): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    for data, label in test_loader:
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, label)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(label.data.view_as(pred)))
        for i in range(len(label)):
            digit = label.data[i]
            class_correct[digit] += correct[i].item()
            class_total[digit] += 1

    test_loss = test_loss/len(test_loader.sampler)
    print('test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):

        if(np.sum(class_total[i])==0):
            print(class_names[i])
        else:
            print('test accuracy of %s: %2d%% (%2d/%2d)' % (class_names[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))

import torch.nn as nn
import numpy
from torch.autograd import Variable


class BC():
    def __init__(self, model):

        # First we need to 
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all 
        # parameters of the model 

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] # This will be used to save the full precision weights
        
        self.target_modules = [] # this will contain the list of modules to be modified

        self.model = model # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters

        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):     
            
        self.save_params()
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign()) # Le mec

    def BWN(self): # Binary Weight Network
        self.save_params()
        for index in range(self.num_of_params):
            E=self.target_modules[index].data.abs().mean()
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign() *E)
            

    def restore(self):

        ### restore the copy from self.saved_params into the model 

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
      
    def clip(self):

        ## To be completed 
        ## Clip all parameters to the range [-1,1] using Hard Tanh 
        ## you can use the nn.Hardtanh function
            
        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)  # Le mec 


#         for index in range(self.num_of_params):
#             hardtanh = nn.Hardtanh()
#             self.target_modules[index].data.copy_(hardtanh(self.target_modules[index].data)) # Nous


    def forward(self,x):

        ### This function is used so that the model can be used while training
        out = self.model(x)
        return out

def training_binary(n_epochs, train_loader, valid_loader, model, criterion, optimizer,method="BWN"):
    train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity



    for epoch in range(n_epochs):
        train_loss, valid_loss, test_loss = 0, 0, 0 # monitor losses
        class_correct_train ,class_total_train = 0, 0 
        class_correct_valid ,class_total_valid = 0, 0 
        class_correct_test ,class_total_test = 0, 0 


        # train the model
        model.model.train() # prep model for training
        for data, label in train_loader:
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            if(method == "BWN"):
                model.BWN()
            else:
                model.binarization()
            optimizer.zero_grad()

             # clear the gradients of all optimized variables


            output = model.forward(data) # forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(output, label) # calculate the loss

            loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
            model.restore()
            optimizer.step() # perform a single optimization step (parameter update)
            model.clip()

            train_loss += loss.item() * data.size(0) # update running training loss

            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(label.data.view_as(pred)))
            for i in range(len(label)):
                digit = label.data[i]
                class_correct_train += correct[i].item()
                class_total_train += 1
        

    # validate the model
        model.model.eval()
        if(method == "BWN"):
                model.BWN()
        else:
            model.binarization()
        for data, label in valid_loader:
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            with torch.no_grad():
                output = model.model(data)
            loss = criterion(output,label)
            valid_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(label.data.view_as(pred)))
            for i in range(len(label)):
                digit = label.data[i]
                class_correct_valid += correct[i].item()
                class_total_valid += 1
        model.restore()

        # calculate average loss over an epoch
        train_loss /= len(train_loader.sampler)
        valid_loss /= len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_acc.append(class_correct_train/class_total_train)
        valid_acc.append(class_correct_valid/class_total_valid)

        print('epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.model.state_dict(), 'model_binary.pt')
            valid_loss_min = valid_loss

    return train_losses, valid_losses, train_acc, valid_acc


def evaluation_binary(model, test_loader, criterion,method = "BWN"): 

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.model.eval()
  #model.binarization()
    if(method == "BWN"):
        model.BWN()
    else:
        model.binarization()
    for data, label in test_loader:
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        #with torch.no_grad():
        output = model.forward(data)
        #print(output)
        loss = criterion(output, label)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(label.data.view_as(pred)))
        for i in range(len(label)):
            digit = label.data[i]
            class_correct[digit] += correct[i].item()
            class_total[digit] += 1

    test_loss = test_loss/len(test_loader.sampler)
    print('test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):

        if(np.sum(class_total[i])==0):
            print(class_names[i])
        else:
            print('test accuracy of %s: %2d%% (%2d/%2d)' % (class_names[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
            print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, power=True):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        # xout = (xhard - x).detach() + x
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            if power:
                input_q = power_quant(input_abs, grids).mul(sign)  # project to Q^a(alpha, B)
            else:
                input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.power = power if w_bit>2 else False
        self.grids = build_power_value(self.w_bit, additive=True)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, power=self.power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight, mask = torch.zeros(1), mask_flag = False):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = 0
            std = 1
            wei = weight.data
            mean = wei[wei.nonzero(as_tuple=True)].mean()
            std = wei[wei.nonzero(as_tuple=True)].std()
            #print(x[x.nonzero(as_tuple=True)].mean())


            #mean = weight.data.mean()
            #std = weight.data.std()
            if(mask_flag):
                weight = weight.add(-mean).div(std) * mask      # weights normalization
            else:
                weight = weight.add(-mean).div(std)
            weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q


def act_quantization(b, grid, power=True):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)
            if power:
                input_q = power_quant(input_c, grid)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = 4
        self.weight_quant = weight_quantize_fn(w_bit=self.bit, power=True)
        self.act_grid = build_power_value(self.bit, additive=True)
        self.act_alq = act_quantization(self.bit, self.act_grid, power=True)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        if hasattr(self, 'weight_mask'):
            weight_q = self.weight_quant(self.weight, self.weight_mask, True)
        else :
            weight_q = self.weight_quant(self.weight)
        x = self.act_alq(x, self.act_alpha)
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)

def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/8 # FP4
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops
    
    # total params 
    

    #print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
    #print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    #print("ReLU: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    #print("AvgPool: S={}, F_in={}, P={}, params={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/2
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    #print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features,int(m.total_params.item()),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    inutile = True
    #print ("Sequential: No additional parameters  / op")

def remove_hook(m, x, y):
    m.total_ops = torch.zeros(1)

# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, custom_ops = {}):
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        
        else:
            print("Not implemented for ", m)

    def remove_hooks(m):
        if len(list(m.children())) > 0: return
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, (nn.AvgPool2d)) or isinstance(m, nn.Linear) or isinstance(m, nn.Sequential):
            m.register_forward_hook(remove_hook)

    model.apply(remove_hooks)
    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params

    return total_ops, total_params

def score(model , quantization = False):
    ref_params = 5586981
    ref_flops  = 834362880
    '''if(pruning):
        criterion = nn.CrossEntropyLoss()
        parameters_to_prune=[]
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
                parameters_to_prune.append((module,'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
        loaded_cpt=torch.load(filename)
        model.load_state_dict(loaded_cpt)
        evaluation(model, testloader, criterion)
    '''   
    flops, _ = profile(model, (1,3,32,32))
    flops = flops.item()
    
    params = 0
    if(quantization):
        for name, para in model.named_parameters():
            if "conv" in name and name.startswith("conv") == False:
                params += torch.sum(abs(para) >= 1e-20).item()/8
                #params+= para.nonzero().size(0)/8
            else:
                params += torch.sum(abs(para) >= 1e-20).item()/2
    else:
        for name, para in model.named_parameters():
            params += torch.sum(abs(para) >= 1e-20).item()/2
    return flops/ref_flops , params/ref_params