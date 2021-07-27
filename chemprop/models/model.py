from argparse import Namespace

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class EvaluationDropout(nn.Dropout):
    def forward(self, input):
        return nn.functional.dropout(input, p = self.p)


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, uncertainty: bool = False):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        :param uncertainty: Whether uncertainty values should be predicted.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        self.uncertainty = uncertainty

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.use_last_hidden = True

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)
        self.args = args

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        # When using dropout for uncertainty, use dropouts for evaluation in addition to training.
        if args.uncertainty == 'dropout':
            dropout = EvaluationDropout(args.dropout)
        else:
            dropout = nn.Dropout(args.dropout)

        activation = get_activation_function(args.activation)

        output_size = args.output_size

        if self.uncertainty:
            output_size *= 2

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 3):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.last_hidden_size),
            ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.last_hidden_size, output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        ffn = self.ffn if self.use_last_hidden else nn.Sequential(
            *list(self.ffn.children())[:-1])
        output = ffn(self.encoder(*input))

        if self.uncertainty:
            even_indices = torch.tensor(range(0, list(output.size())[1], 2))
            odd_indices = torch.tensor(range(1, list(output.size())[1], 2))

            if self.args.cuda:
                even_indices = even_indices.cuda()
                odd_indices = odd_indices.cuda()

            predicted_means = torch.index_select(output, 1, even_indices)
            predicted_uncertainties = torch.index_select(output, 1, odd_indices)
            capped_uncertainties = nn.functional.softplus(predicted_uncertainties)

            output = torch.stack((predicted_means, capped_uncertainties), dim = 2).view(output.size())

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training and self.use_last_hidden:
            output = self.sigmoid(output)

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size

    is_classifier = args.dataset_type == 'classification'
    if args.uncertainty == 'mve':
        model = MoleculeModel(classification=is_classifier, uncertainty=True)
    else:
        model = MoleculeModel(classification=is_classifier)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
