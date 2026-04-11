# %% [markdown]
# 在选择了网络架构并设置好合适的超参数之后，我们需要正式开始模型的训练阶段  
# 找到使损失函数最小的参数值是我们的最终目标
# > 超参数：学习率、批次大小等人为预设参数    
# 
# 那我们应该怎么去管理这些参数呢，无论是查看、提取、还是在别的环境中复用

# %%
import torch 
from torch import nn 

net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
X = torch.rand(size=(4, 2))
net(X)


# %% [markdown]
# ## 参数访问
# * `net`表示整个网络
# * `net[2]`表述网络的第三层
# * `.state_dict()`导出状态字典，即网络的参数
# 
# > 可见，第二层有两个参数，分别是权重`weight`和偏置`bias`

# %%
net[2].state_dict()

# %% [markdown]
# ### 目标参数
# 我们前面所看到的并不是底层的参数本尊，而是每一个参数的实例（分身），如果我们需要对参数进行任何操作，必须访问到参数的底层数值   
# * 可见，参数实际上也是一个类的实例化，也就是一个对象，包含了多种信息

# %%
print(type(net[2].bias))
print('------')
print(net[2].bias)
print('------')
print(net[2].bias.data)
print('------')
print(net[2].weight.grad)

# %% [markdown]
# ### 一次性访问所有参数
# 

# %%
print([(name,param.shape) for name,param in net[2].named_parameters()])
print('------')
print('加上#')
print(*[(name,param.shape) for name,param in net[2].named_parameters()])#去掉列表的括号和逗号
print(*[(name, param.shape) for name, param in net.named_parameters()])

# %% [markdown]
# ### 从嵌套块收集参数

# %%
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):#循环4次，每次创建一个block1，并添加到net中
        net.add_module(f'block{i}',block1())#为每个block命名
    return net
rgnet = nn.Sequential(block2(),nn.Linear(4,1))
Y = torch.rand(size=(2,4))
rgnet(Y)

# %%
print(rgnet)

# %%
rgnet[0][1][0].bias.data#Sequential,block,layer都有各自的索引，依次访问到第一个block的第一个层的偏置参数数据

# %% [markdown]
# ## 参数初始化

# %% [markdown]
# ### 内置初始化
# 调用内置的初始化器来对参数进行初始化

# %%
#适用于全连接层的权重参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0],net[0].bias.data[0]#访问第一个block的第一个层的权重和偏置参数数据

# %% [markdown]
# ### 自定义初始化
# * 假如我们需要进行按照下面规则的初始化
# <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
#   <mtable displaystyle="true" columnalign="right" columnspacing="0em" rowspacing="3pt">
#     <mtr>
#       <mtd>
#         <mtable displaystyle="true" columnspacing="" rowspacing="3pt">
#           <mtr>
#             <mtd>
#               <mi>w</mi>
#               <mo>&#x223C;</mo>
#               <mrow data-mjx-texclass="INNER">
#                 <mo data-mjx-texclass="OPEN">{</mo>
#                 <mtable columnalign="left left" columnspacing="1em" rowspacing=".2em">
#                   <mtr>
#                     <mtd>
#                       <mi>U</mi>
#                       <mo stretchy="false">(</mo>
#                       <mn>5</mn>
#                       <mo>,</mo>
#                       <mn>10</mn>
#                       <mo stretchy="false">)</mo>
#                     </mtd>
#                     <mtd>
#                       <mtext>&#xA0;&#x53EF;&#x80FD;&#x6027;&#xA0;</mtext>
#                       <mfrac>
#                         <mn>1</mn>
#                         <mn>4</mn>
#                       </mfrac>
#                     </mtd>
#                   </mtr>
#                   <mtr>
#                     <mtd>
#                       <mn>0</mn>
#                     </mtd>
#                     <mtd>
#                       <mtext>&#xA0;&#x53EF;&#x80FD;&#x6027;&#xA0;</mtext>
#                       <mfrac>
#                         <mn>1</mn>
#                         <mn>2</mn>
#                       </mfrac>
#                     </mtd>
#                   </mtr>
#                   <mtr>
#                     <mtd>
#                       <mi>U</mi>
#                       <mo stretchy="false">(</mo>
#                       <mo>&#x2212;</mo>
#                       <mn>10</mn>
#                       <mo>,</mo>
#                       <mo>&#x2212;</mo>
#                       <mn>5</mn>
#                       <mo stretchy="false">)</mo>
#                     </mtd>
#                     <mtd>
#                       <mtext>&#xA0;&#x53EF;&#x80FD;&#x6027;&#xA0;</mtext>
#                       <mfrac>
#                         <mn>1</mn>
#                         <mn>4</mn>
#                       </mfrac>
#                     </mtd>
#                   </mtr>
#                 </mtable>
#                 <mo data-mjx-texclass="CLOSE" fence="true" stretchy="true" symmetric="true"></mo>
#               </mrow>
#             </mtd>
#           </mtr>
#         </mtable>
#       </mtd>
#     </mtr>
#   </mtable>
# </math>

# %%
def my_init(m):
    if type(m) == nn.Linear:
        print('Init',*[(name,parm.shape) for name,parm in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,-10)
        m.weight.data *= m.weight.data.abs() >= 5#将权重参数绝对值小于5的元素置零,将两项相乘后赋值
net.apply(my_init)
net[0].weight[:2]#访问第一个block的第一个层的前

# %%
#参数可以直接赋值
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

# %% [markdown]
# ## 参数绑定
# * 用于多个层共享参数
# 

# %%
shared = nn.Linear(8, 8 )
net = nn.Sequential(nn.Linear(2,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])#验证net[2]和net[4]是否共享参数
net[2].weight.data[0, 0] = 100
print(net[4].weight.data[0, 0])#验证net[4]的权重参数是否也被修改
print(net[2].weight.data[0, 0], net[4].weight.data[0, 0])#打印net[2]和net[4]的权重参数数据，验证它们是否相等

# %% [markdown]
# 第三个神经网络层`net[2]`,第五个神经网络层`net[4]`的参数是绑定的  
# ***当我们改变其中的一个值时，另一个也会跟着改变***

# %% [markdown]
# 


