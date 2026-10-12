import torch
import copy

from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import gurobipy as grb


import nnverify.attack
from nnverify import util
from nnverify.domains import grb_utils

import time

device = 'cpu'


class PolyzTransformer:
    def __init__(self, prop, cof_constrain=None, bias_constrain=None, complete=False):
        """
        lcof: the coefficients for the lower bound transformation (w.r.t. the input variables)
        ucof: the coefficients for the upper bound transformation (w.r.t. the input variables)
        lcst: the constants for the lower bound transformation (w.r.t. the input variables)
        ucst: the constants for the upper bound transformation (w.r.t. the input variables)
        ilb: the lower bound for the input variables
        iub: the upper bound for the input variables
        During the verification, we will iteratively update the lcf, ucf, lcst, ucst
        while fixing the lb, and ub after normalization.
        """
        # self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.prop = prop
        self.size = prop.get_input_size()

        self.batch_size = self.size[0]

        cof = torch.eye(self.size[-1], device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        cst = torch.zeros(self.size[-1], device=self.device).unsqueeze(0).repeat(self.batch_size, 1)

        # cof = torch.eye(self.size[-1], device=self.device)
        # cst = torch.zeros(self.size[-1], device=self.device)

        if self.size[-1] == 784:
            self.shape = (1, 28, 28)
            self.size = 784
        elif self.size[-1] == 3072:
            self.shape = (3, 32, 32)
            self.size = 3072

        # if self.size[-1] == 784:
        #     self.shape = (1, 28, 28)
        # elif self.size[-1] == 3072:
        #     self.shape = (3, 32, 32)

        self.lcof = [cof, ]
        self.ucof = [cof, ]
        self.lcst = [cst, ]
        self.ucst = [cst, ]
        self.cur_lcof = None
        self.cur_lcst = None
        self.unstable_relus = []

        self.prop = prop
        self.cof_constrain = cof_constrain
        self.bias_constrain = bias_constrain
        self.optimize_lambda = False
        self.complete = complete


        




        # self.ilb = prop.input_lb
        # self.iub = prop.input_ub

        # self.map_for_noise_indices = {}

        # self.ilb = self.ilb.to(device)
        # self.iub = self.iub.to(device)

        # center = (self.ilb + self.iub) / 2
        # noise_ind = self.get_noise_indices()
        

        # cof = ((self.iub - self.ilb) / 2 * torch.eye(self.size))[noise_ind]


        # self.centers = []
        # self.cofs = []
        # self.relu_layer_cofs = []
        # self.set_zono(center, cof)
        # self.lp_model = None # This will be used for ReLU splits
        # self.map_relu_layer_idx_to_layer_idx = {}
        # self.masked = False # If the mask exist we use LP solver to find the bounds



        self.size = prop.get_input_size()
        self.prop = prop
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.complete = complete

        # Following fields are used for complete verification
        self.complete = complete
        self.map_for_noise_indices = {}

        self.batch_size = self.size[0]

        if self.size[-1] == 784:
            self.shape = (1, 28, 28)
            self.size = 784
        elif self.size[-1] == 3072:
            self.shape = (3, 32, 32)
            self.size = 3072

        
        self.ilb = self.ilb.to(device)
        self.iub = self.iub.to(device)


        center = (self.ilb + self.iub) / 2
        self.unstable_relus = []
        noise_ind = self.get_noise_indices()

        temp1 = ((self.iub - self.ilb) / 2).unsqueeze(-1).repeat(1, 1, self.size)
        temp2 = torch.eye(self.size).unsqueeze(0).repeat(self.batch_size, 1, 1)
        cof = (temp1 * temp2)

        self.centers = []
        self.cofs = []
        self.relu_layer_cofs = []
        self.set_zono(center, cof)
        self.lp_model = None # This will be used for ReLU splits
        self.map_relu_layer_idx_to_layer_idx = {}
        self.masked = False # If the mask exist we use LP solver to find the bounds






    def get_noise_indices(self):
        num_eps = 1e-7
        noise_ind = torch.where(self.iub > (self.ilb + num_eps))
        if noise_ind[0].size() == 0:
            # add one dummy index in case there is no perturbation
            noise_ind = torch.tensor([0]).to(device)
        return noise_ind


    def compute_lb(self):
        lhs = self.compute_lb_deeppoly()
        rhs = self.compute_lb_deepz()


        return torch.max(lhs, rhs)

    def get_zono_lb(self, adv_label, center, cof):
        cof = cof[:, adv_label]
        cof_abs = torch.sum(torch.abs(cof), dim=0)
        lb = center[adv_label] - cof_abs
        if self.prop.is_conjunctive():
            lb = torch.min(lb)
        else:
            lb = torch.max(lb)
        return lb
    
    def compute_lb_deepz(self):
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=-2)
        
        lb = center - cof_abs

        return lb

    def compute_lb_deeppoly(self, complete=False, adv_label=None):
        """
        return the lower bound for the variables of the current layer
        """
        lcof = self.lcof[-1]
        lcst = self.lcst[-1]

        
        
        lcof = lcof.to(self.device)
        lcst = lcst.to(self.device)

        start_time = (time.time())
        for i in range(2, len(self.lcof) + 1):
            lcof, lcst = absmul(self.lcof[-i], self.ucof[-i], self.lcst[-i], self.ucst[-i], lcof, lcst, down=True)
        
        self.cur_lcof = lcof.detach()
        self.cur_lcst = lcst.detach()

        pos_cof = F.relu(lcof)
        neg_cof = -F.relu(-lcof)

        pos_lb = (self.prop.input_lb.to(self.device).unsqueeze(1) @ pos_cof).squeeze(1)
        neg_lb = (self.prop.input_ub.to(self.device).unsqueeze(1) @ neg_cof).squeeze(1)

        lb = pos_lb + neg_lb + lcst


        if self.cof_constrain is not None:
            k = torch.zeros(1, lcof.shape[1]).requires_grad_()
            optimizer = optim.Adam([k], lr=1)
            scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

            for i in range(150):
                optimizer.zero_grad()
                k.data.clamp_(0)
                new_cof = lcof.detach() + k * self.cof_constrain
                new_pos_cof = F.relu(new_cof)
                new_neg_cof = -F.relu(-new_cof)

                new_pos_lb = self.prop.input_lb @ new_pos_cof
                new_neg_lb = self.prop.input_ub @ new_neg_cof
                new_lb = new_pos_lb + new_neg_lb + lcst.detach() + k * self.bias_constrain
                loss = -new_lb.mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
            new_lb = torch.maximum(lb, new_lb.detach())

            return new_lb.squeeze()

        if complete:
            lb = lb.squeeze()
            if self.prop.is_conjunctive():
                lb = torch.min(lb)
            else:
                lb = torch.max(lb)
            return lb, True, None
        
        return lb.squeeze()

    

    def compute_ub(self, test=True):
        return torch.min(self.compute_ub_deepz(test), self.compute_ub_deeppoly(test))

    def compute_ub_deepz(self, test=True):
        # This can be little faster by reusing the computation
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=-2)
        
        ub = center + cof_abs

        return ub

    def compute_ub_deeppoly(self, test=True):
        """
        return the upper bound for the variables of the current layer
        """
        ucof = self.ucof[-1]
        ucst = self.ucst[-1]

        for i in range(2, len(self.ucof) + 1):
            ucof, ucst = absmul(self.lcof[-i], self.ucof[-i], self.lcst[-i], self.ucst[-i], ucof, ucst, down=False)

        pos_cof = F.relu(ucof)
        neg_cof = -F.relu(-ucof)

        pos_ub = (self.prop.input_ub.to(self.device).unsqueeze(1) @ pos_cof).squeeze(1)
        neg_ub = (self.prop.input_lb.to(self.device).unsqueeze(1) @ neg_cof).squeeze(1)

        ub = pos_ub + neg_ub + ucst

        if self.cof_constrain is not None and test:

            k = torch.zeros(1, ucof.shape[1]).requires_grad_()
            optimizer = optim.Adam([k], lr=1)
            scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

            for i in range(150):
                optimizer.zero_grad()
                k.data.clamp_(0)
                new_cof = ucof.detach() - k * self.cof_constrain
                new_pos_cof = F.relu(new_cof)
                new_neg_cof = -F.relu(-new_cof)

                new_pos_ub = self.prop.input_ub @ new_pos_cof
                new_neg_ub = self.prop.input_lb @ new_neg_cof
                new_ub = new_pos_ub + new_neg_ub + ucst.detach() - k * self.bias_constrain

                loss = new_ub.mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            new_ub = torch.minimum(ub, new_ub.detach())

            return new_ub.squeeze()

        return ub.squeeze()

    def bound(self):
        return self.compute_lb(), self.compute_ub()
    
    def get_zono(self):
        return self.centers[-1], self.cofs[-1]

    def set_zono(self, center, cof):
        self.centers.append(center)
        self.cofs.append(cof)

    def get_cof_cst(self):
        return self.lcof, self.ucof, self.lcst, self.ucst

    def set_cof_cst(self, lcof, ucof, lcst, ucst):
        self.lcof.append(lcof.to(self.device))
        self.ucof.append(ucof.to(self.device))
        self.lcst.append(lcst.to(self.device))
        self.ucst.append(ucst.to(self.device))

    def get_all_bounds(self):
        lbs = []
        ubs = []

        ucof = copy.copy(self.ucof)
        ucst = copy.copy(self.ucst)
        lcof = copy.copy(self.lcof)
        lcst = copy.copy(self.lcst)

        for i in range(len(ucof)):
            self.ucof = ucof[:i + 1]
            self.ucst = ucst[:i + 1]
            self.lcof = lcof[:i + 1]
            self.lcst = lcst[:i + 1]

            lbs.append(self.compute_lb())
            ubs.append(self.compute_ub())

        return lbs, ubs

    

    def handle_linear(self, layer, last_layer=False):
        self.handle_linear_deepz(layer, last_layer)
        self.handle_linear_deeppoly(layer, last_layer)

        return self

    def handle_linear_deepz(self, layer, last_layer=False):
        """
        handle linear layer
        """
        weight = layer.weight.T
        bias = layer.bias

        # if last_layer:
        #     weight = weight @ self.prop.output_constr_mat()
        #     bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()

        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        prev_cent, prev_cof = self.get_zono()

        center = (prev_cent.unsqueeze(1) @ weight).squeeze(1) + bias
        cof = prev_cof @ weight

        self.set_zono(center, cof)

        return self

    def handle_linear_deeppoly(self, layer, last_layer=False):
        """
        handle linear layer
        """
        last_layer = False
        weight = layer.weight.T
        bias = layer.bias


        if last_layer:
            weight = weight @ self.prop.output_constr_mat()
            bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()

        if len(weight.shape) < 3:
            cof = weight.unsqueeze(0).repeat(self.batch_size, 1, 1)
        else:
            cof = weight
        cst = bias.unsqueeze(0).repeat(self.batch_size, 1)

        self.set_cof_cst(cof, cof, cst, cst)
        self.shape = (self.batch_size, weight.shape[1])
        self.size = weight.shape[1]

        
        return self

    
    def handle_conv2d(self, layer):
        self.handle_conv2d_deepz(layer)
        self.handle_conv2d_deeppoly(layer)

        return self
    
    def handle_conv2d_deepz(self, layer):
        """
        handle conv2d layer
        first transform it to linear matrix
        then use absmul func
        """
        weight = layer.weight
        bias = layer.bias
        num_kernel = weight.shape[0]

        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding

        shape = self.shape

        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        # self.shape = (num_kernel, output_h, output_w)
        # self.size = num_kernel * output_h * output_w

        prev_cent, prev_cof = self.get_zono()
        num_eps = prev_cof.shape[1]

        prev_cent = prev_cent.reshape(-1, shape[0], input_h, input_w)
        prev_cof = prev_cof.reshape(-1, shape[0], input_h, input_w)

        center = F.conv2d(prev_cent, weight, padding=layer.padding, stride=layer.stride, bias=bias).reshape(self.batch_size, -1)
        cof = F.conv2d(prev_cof, weight, padding=layer.padding, stride=layer.stride).reshape(self.batch_size, num_eps, -1)

        self.set_zono(center, cof)
        return self

    def handle_conv2d_deeppoly(self, layer):
        """
        handle conv2d layer
        first transform it to linear matrix
        then use absmul func
        """
        weight = layer.weight
        bias = layer.bias
        num_kernel = weight.shape[0]


        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding

        size = self.size
        shape = self.shape


        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        self.shape = (num_kernel, output_h, output_w)
        self.size = num_kernel * output_h * output_w

        ### pad cof, cst first ###

        cof = torch.eye(size).reshape(size, *(shape)).unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)

        pad2d = (p_w, p_w, p_h, p_h)
        cof = F.pad(cof, pad2d)

        ### change to the linear matrix form ###
        linear_cof = []
        for i in range(output_h):
            w_cof = []
            for j in range(output_w):
                h_start = i * s_h
                h_end = h_start + k_h
                w_start = j * s_w
                w_end = w_start + k_w

                w_cof.append(cof[:, :, :, h_start: h_end, w_start: w_end])

            linear_cof.append(torch.stack(w_cof, dim=2  ))

        linear_cof = torch.stack(linear_cof, dim=2).reshape(self.batch_size, size, output_h, output_w, -1)

        new_weight = weight.reshape(num_kernel, -1).T
        new_cof = linear_cof @ new_weight.unsqueeze(0)
        new_cof = new_cof.permute(0, 1, 4, 2, 3).reshape(self.batch_size, size, -1)
        new_cst = bias.view(-1, 1, 1).expand(num_kernel, output_h, output_w).reshape(1, -1)

        self.set_cof_cst(new_cof, new_cof, new_cst, new_cst)
        return self

    
    def handle_relu(self, layer, optimize=True, relu_mask=None):
        lb, ub = self.bound()
        self.handle_relu_deepz(layer, optimize, relu_mask, lb, ub)
        self.handle_relu_deeppoly(layer, optimize, relu_mask, lb, ub)

        return self

    def handle_relu_deeppoly(self, layer, optimize=True, relu_mask=None, lb=None, ub=None):
        """
        handle relu func
        abs(lb) > abs(ub) => k = 0, otherwise k = 1
        """
        size = self.size

        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0)))

        new_lcof = torch.zeros((self.batch_size, size)).to(self.device)
        new_ucof = torch.zeros((self.batch_size, size)).to(self.device)
        new_lcst = torch.zeros((self.batch_size, size)).to(self.device)
        new_ucst = torch.zeros((self.batch_size, size)).to(self.device)


        ### case 1 ub <= 0 ###
        ### will be cleared ###
        clear = ub <= 0

        ### case 2 lb >= 0 ###
        ### will be saved ###
        noclear = ~ clear
        save = noclear & (lb >= 0)

        ### case 3 lb<0 & ub>0 ###
        ### need to be approximated ###
        approximate = (noclear & (lb < 0)).to(self.device)

        if self.optimize_lambda:
            if hasattr(layer, 'opt_lambda'):
                pass
            else:
                mask = ~(approximate & (abs(lb) > abs(ub)))
                layer.opt_lambda = torch.ones(size) * mask.int()
                layer.opt_lambda.requires_grad_()
            opt_lambda = layer.opt_lambda
        else:
            mask = ~(approximate & (abs(lb) > abs(ub)))
            opt_lambda = torch.ones(size, device=self.device) * mask.int()

        ####! optimize  ####
        new_lcof[save] = 1

        act_lambda = opt_lambda * approximate.int()
        new_lcof += act_lambda

        ### upper bound ###
        ### saved part ###
        new_ucof[save] = 1

        ### change k ###
        denominator = ub - lb
        denominator[denominator == 0] = 1.
        tmp_ucof = ub / denominator
        tmp_ucst = - lb * ub / denominator
        new_ucof += tmp_ucof * approximate.int()
        new_ucst += tmp_ucst * approximate.int()

        self.set_cof_cst(torch.diag_embed(new_lcof), torch.diag_embed(new_ucof), new_lcst.reshape(self.batch_size, -1),
                                 new_ucst.reshape(self.batch_size, -1))
        
        return self
    


    def handle_relu_deepz(self, layer, optimize=True, relu_mask=None, lb=None, ub=None):
        """
        handle relu func
        """
        size = self.size

        prev_cent, prev_cof = self.get_zono()
        


        relu_layer_idx = len(self.unstable_relus)
        self.map_relu_layer_idx_to_layer_idx[relu_layer_idx] = len(self.cofs)
        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0))[0].tolist())

        num_eps = 1e-7
        lmbda = torch.div(ub, ub - lb + num_eps)
        mu = -(lb / 2) * lmbda

        active_relus = (lb > 0)
        passive_relus = (ub <= 0)
        ambiguous_relus = (~active_relus) & (~passive_relus)


        mult_fact = torch.ones(size, dtype=torch.bool)
        # mult_fact should have 1 at active relus, 0 at passive relus and lambda at ambiguous_relus
        mult_fact = mult_fact * (active_relus + ambiguous_relus * lmbda)

        new_noise_cofs = torch.diag_embed(mu * ambiguous_relus)
        non_empty_mask = new_noise_cofs.abs().sum(dim=0).sum(dim=1).bool()
        new_noise_cofs = new_noise_cofs[:, non_empty_mask, :]

        block_1 = mult_fact.unsqueeze(1) * prev_cof
        block_2 = new_noise_cofs

        cof = torch.cat([block_1, block_2], dim=1)
        center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus

        self.set_zono(center, cof)
        self.relu_layer_cofs.append(cof)


        return self
    
def absmul(lcof, ucof, lcst, ucst, weight, bias, down=True):
    """
    Absdomain multiplication
    """
    pos_wgt = F.relu(weight)
    neg_wgt = -F.relu(-weight)

    if down:
        new_lcof = lcof @ pos_wgt + ucof @ neg_wgt
        new_lcst = (lcst.unsqueeze(1) @ pos_wgt).squeeze(1) + (ucst.unsqueeze(1) @ neg_wgt).squeeze(1) + bias
        return new_lcof, new_lcst
    else:
        new_ucof = ucof @ pos_wgt + lcof @ neg_wgt
        new_ucst = (ucst.unsqueeze(1) @ pos_wgt).squeeze(1) + (lcst.unsqueeze(1) @ neg_wgt).squeeze(1) + bias
        return new_ucof, new_ucst

