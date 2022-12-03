import torch
import torch.nn as nn

optimal_layer_size = 224

class SoftmaxClassifier(nn.Module):
    def __init__(self, k_way = 2):
        super(SoftmaxClassifier, self).__init__()
        self.input_layer_bert = nn.Linear(k_way, optimal_layer_size)
        self.input_layer_resnet50 = nn.Linear(k_way, optimal_layer_size )
        self.hidden_layer = nn.Linear(optimal_layer_size, optimal_layer_size)
        self.output_layer = nn.Linear(optimal_layer_size, k_way)
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Softmax()

    def forward(self, bert_output, resnet50_output):
        bert_output = self.input_layer_bert(bert_output)
        bert_output == self.hidden_activation(bert_output)
        resnet50_output = self.input_layer_resnet50(resnet50_output)
        resnet50_output = self.hidden_activation(bert_output)
        X = torch.maximum(resnet50_output, bert_output)
        X = self.hidden_layer(X)
        X = self.hidden_activation(X)
        X = self.output_layer(X)
        X = self.output_activation(X)
        return X

model = SoftmaxClassifier(k_way=6)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

