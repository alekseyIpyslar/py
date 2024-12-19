

class ModelNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer=3, act_type=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act_type = act_type
        for n in range(1, n_layers + 1):
            self. layers.add_module(f'layer_{n}', nn.Linear(input_dim // n, input_dim // (n + 1)))

        self.layer_out = nn.Linear(input_dim // (n_layers + 1), output_dim)
        self.act_list = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lk_relu': nn.LeakyReLU(),
        })

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # x = nn.functional.tanh(x)
            if self.act_type and self.act_type in self.act_list:
                x = self.act_list[self.act_type](x)

        x = self.layer_out(x)
        return x

model = ModelNN(28 * 28, 10, act_type='lk_relu')
# print(len(list(model.parameters())))
print(model)
