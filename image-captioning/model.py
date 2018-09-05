import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return feature

class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.):

		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embeddings = nn.Embedding(vocab_size, embed_size)

		self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, 
			num_layers=num_layers, dropout=dropout, batch_first=True)

		self.dense = nn.Linear(hidden_size, vocab_size)

	def forward(self, features, captions):
		# remove stop word at the end of the caption, 
		# notice that will result in outputs.shape[1]==captions.shape[1]
		caption_embeddings = self.embeddings(captions[:, :-1])

		# add 2nd dimension
		features = features.view(len(features), 1, -1)
		inputs = torch.cat((features, caption_embeddings), 1)

		out, _ = self.lstm(inputs)

		out = self.dense(out)

		return out


	def sample(self, features, states=None, max_len=20):
		" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
		" inputs tensor contains the embedded input features corresponding to a single image. "
		hidden = states
		inputs = features
		sentence = []
		for _ in range(max_len):
			output, hidden = self.lstm(inputs, hidden)
			output = self.dense(output)

			prediction = torch.argmax(output, dim=2)
			#break when stop is reached
			if prediction == 1:
				break

			sentence.append(prediction.item())

			inputs = self.embeddings(prediction)

		return sentence