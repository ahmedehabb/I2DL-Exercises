import torch
import torch.nn as nn
import numpy as np
import copy

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################


        hidden_layer_sizes = hparams.get('n_hidden_layers', [256, 128])  # Default values
        self.latent_dim = self.hparams.get('latent_dim', 20)
        dropout_prob = self.hparams.get('dropout_prob', 0.2)
        
        # Create a list to hold the encoder layers
        encoder_layers = []
        
        # Input layer
        encoder_layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))

        # Hidden layers with batch normalization and dropout
        for i in range(len(hidden_layer_sizes) - 1):
            encoder_layers.append(nn.BatchNorm1d(hidden_layer_sizes[i]))
            encoder_layers.append(nn.ReLU())  # Activation function
            encoder_layers.append(nn.Dropout(p=dropout_prob))  # Dropout layer
            encoder_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))

        # Output layer (latent representation)
        encoder_layers.append(nn.BatchNorm1d(hidden_layer_sizes[-1]))
        encoder_layers.append(nn.ReLU())  # Activation function
        encoder_layers.append(nn.Linear(hidden_layer_sizes[-1], self.latent_dim))

        
        # Define the encoder as a sequence of layers
        self.encoder = nn.Sequential(*encoder_layers)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        # Extract the hidden layer sizes from the encoder's hparams
        hidden_layer_sizes = self.hparams.get('n_hidden_layers', [256, 128])  # Default values
        self.latent_dim = self.hparams.get('latent_dim', 20)
        dropout_prob = self.hparams.get('dropout_prob', 0.2)

        # Reverse the order of hidden_layer_sizes to mirror the encoder's architecture
        reversed_hidden_layer_sizes = copy.deepcopy(hidden_layer_sizes[::-1])

        # Latent dimension (input layer)
        decoder_layers = [nn.Linear(self.latent_dim, reversed_hidden_layer_sizes[0])]

        # Hidden layers with batch normalization and dropout
        for i in range(len(reversed_hidden_layer_sizes) - 1):
            decoder_layers.append(nn.BatchNorm1d(reversed_hidden_layer_sizes[i]))
            decoder_layers.append(nn.ReLU())  # Activation function
            decoder_layers.append(nn.Dropout(p=dropout_prob))  # Dropout layer
            decoder_layers.append(nn.Linear(reversed_hidden_layer_sizes[i], reversed_hidden_layer_sizes[i+1]))

        # Output layer (reconstructed input)
        decoder_layers.append(nn.BatchNorm1d(reversed_hidden_layer_sizes[-1]))
        decoder_layers.append(nn.ReLU())  # Activation function
        decoder_layers.append(nn.Linear(reversed_hidden_layer_sizes[-1], output_size))
        decoder_layers.append(nn.Sigmoid())  # Output activation function (e.g., for image data)

        # Define the decoder as a sequence of layers
        self.decoder = nn.Sequential(*decoder_layers)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        # Feed the input image to your encoder to generate the latent vector
        latent_vector = self.encoder(x)
        # Decode the latent vector to get the reconstruction of the input
        reconstruction = self.decoder(latent_vector)
        return reconstruction

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################


        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Reset gradients
        X = batch
        X = X.to(self.device)
        flattened_X = X.view(X.shape[0], -1)

        # Forward pass through the autoencoder
        output = self.forward(flattened_X)

        # Compute the loss
        loss = loss_func(output, flattened_X)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)

            # Forward pass through the autoencoder
            output = self.forward(flattened_X)

            # Compute the loss
            loss = loss_func(output, flattened_X)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################

        # Define the number of neurons for each hidden layer in the classifier
        hidden_layer_sizes = hparams['n_hidden_classifier']
        dropout_prob = hparams['dropout_prob']

        classifier_layers = []
        # Input layer
        classifier_layers.append(nn.Linear(encoder.latent_dim, hidden_layer_sizes[0]))
        classifier_layers.append(nn.BatchNorm1d(hidden_layer_sizes[0]))  # Batch normalization
        classifier_layers.append(nn.ReLU())  # Activation function

        # Hidden layers with batch normalization and dropout
        for i in range(len(hidden_layer_sizes) - 1):
            classifier_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            classifier_layers.append(nn.BatchNorm1d(hidden_layer_sizes[i + 1]))  # Batch normalization
            classifier_layers.append(nn.ReLU())  # Activation function

        # Output layer
        classifier_layers.append(nn.Linear(hidden_layer_sizes[-1], hparams['num_classes']))
        
        
        # Define the classifier as a sequence of layers
        self.model = nn.Sequential(*classifier_layers)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
