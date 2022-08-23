from string import punctuation
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(dataset):
    """
    Load given dataset from zip file and convert it to simple PyTorch format
    """

    import gzip
    import zipfile

    # unzip data
    zip_path = "data/{}.zip".format(dataset)
    data_path = "data/{}/".format(dataset)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)

    # file paths
    train_set = data_path + "train-images-idx3-ubyte.gz"
    train_labels = data_path + "train-labels-idx1-ubyte.gz"
    test_set = data_path + "t10k-images-idx3-ubyte.gz"
    test_labels = data_path + "t10k-labels-idx1-ubyte.gz"

    # load images
    images_as_tensors = []
    for file in [train_set, test_set]:
        with gzip.open(file, 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            images_as_tensors.append(torch.from_numpy(images).unsqueeze(1).float())

    # load labels
    labels_as_tensors = []
    for file in [train_labels, test_labels]:
        with gzip.open(file, 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            labels_as_tensors.append(torch.from_numpy(labels).long())

    return images_as_tensors[0].to(DEVICE), \
           labels_as_tensors[0].to(DEVICE), \
           images_as_tensors[1].to(DEVICE), \
           labels_as_tensors[1].to(DEVICE)


def reviews_preprocess(reviews, labels):
    """
    Remove punctuation from review data and replace multiple space with single space.
    Map labels from positive/negative to 1/0.

    Parameters
    ----------
    reviews: list of reviews
    labels: list of labels
    Returns
    -------
    all_reviews: reviews with punctuation removed
    all_words: list of all words occurring in the reviews
    labels: labels with 1 for positive and 0 for negative
    """
    all_reviews = list()
    for text in reviews:
        text = text.lower()
        text = "".join([ch for ch in text if ch not in punctuation])
        # replace multiple spaces with single space
        text = " ".join(text.split())
        all_reviews.append(text)
    all_text = " ".join(all_reviews)
    all_words = all_text.split()

    # map labels: "positive" = 1 and "negative" = 0
    labels = [1 if label.strip() == "positive" else 0 for label in labels]

    return all_reviews, all_words, labels


def reviews_create_word_ids(all_words):
    """
    Creates a dictionary mapping each word to an unique id.
    Parameters
    ----------
    all_words: list of all words occurring in the data
    Returns
    -------
    dictionary with word as key and corresponding id as value
    """
    count_words = Counter(all_words)
    total_words = len(all_words)
    sorted_words = count_words.most_common(total_words)
    word_ids = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}
    return word_ids


def reviews_encode(word_ids, reviews):
    """
    Replace each word in the review with its corresponding id specified in the
    dictionary word_ids

    Parameters
    ----------
    word_ids: dictionary with word as key and id as value
    reviews: review data

    Returns
    -------
    review data with each word replaced by its corresponding id

    """
    encoded_reviews = list()
    for review in reviews:
        encoded_review = list()
        for word in review.split():
            if word not in word_ids.keys():
                # if word is not available in word_ids put 0 in that place
                encoded_review.append(0)
            else:
                encoded_review.append(word_ids[word])
        encoded_reviews.append(encoded_review)

    return encoded_reviews


def reviews_pad(encoded_reviews, sequence_length):
    """
    Pad/truncate encoded reviews to the same sequence length, by adding zeros at the
    beginning or cutting the end.

    Parameters
    ----------
    encoded_reviews: review data with each word encoded with corresponding id
    sequence_length: length to pad/truncate the reviews to

    Returns
    -------
    Encoded reviews padded/truncated to sequence_length

    """
    padded_reviews = np.zeros((len(encoded_reviews), sequence_length), dtype=int)
    for i, review in enumerate(encoded_reviews):
        review_len = len(review)
        if review_len <= sequence_length:
            zeros = list(np.zeros(sequence_length - review_len, dtype=int))
            new = zeros + review
        else:
            new = review[:sequence_length]
        padded_reviews[i, :] = np.array(new)
    return padded_reviews


def reviews_split(padded_reviews, labels):
    """
    Splits data into train, validation and test set with a 80, 10, 10 split.

    Parameters
    ----------
    padded_reviews: reviews padded to the same sequence length
    labels: labels correspoding to reviews

    Returns
    -------
    train_x: Training data
    train_y: Training labels
    valid_x: Validation data
    valid_y: Validation labels
    test_x: Test data
    test_y: Test labels
    """
    train_x = padded_reviews[: int(0.8 * len(padded_reviews))]
    train_y = labels[: int(0.8 * len(padded_reviews))]
    valid_x = padded_reviews[
        int(0.8 * len(padded_reviews)) : int(0.9 * len(padded_reviews))
    ]
    valid_y = labels[int(0.8 * len(padded_reviews)) : int(0.9 * len(padded_reviews))]
    test_x = padded_reviews[int(0.9 * len(padded_reviews)) :]
    test_y = labels[int(0.9 * len(padded_reviews)) :]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def reviews_create_dataloaders(
    train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=50
):
    """
    Creates PyTorch data loaders

    Parameters
    ----------
    train_x: Training data
    train_y: Training labels
    valid_x: Validation data
    valid_y: Validation labels
    test_x: Test data
    test_y: Test labels
    batch_size: size of the batch

    Returns
    -------
    PyTorch data loaders for train, validation and test
    """
    # create Tensor Dataset
    train_data = TensorDataset(torch.LongTensor(train_x), torch.IntTensor(train_y))
    valid_data = TensorDataset(torch.LongTensor(valid_x), torch.IntTensor(valid_y))
    test_data = TensorDataset(torch.LongTensor(test_x), torch.IntTensor(test_y))

    # dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def reviews_train(
    net,
    train_loader,
    valid_loader,
    lr=0.01,
    lr_decay=0.0001,
    epochs=3,
    clip=5,
    print_every=5,
    criterion=nn.BCELoss(),
    device=DEVICE,
):
    """
    Train a network on the review data

    Parameters
    ----------
    net: Initialized model
    train_loader: Dataloader containing the training data
    valid_loader: Dataloader containing the validation data
    lr: learning rate
    epochs: number of epochs
    clip: clip gradients at this value
    print_every: print current train and validation loss every x steps
    criterion: Loss function
    device: device to train on - cpu or cuda
    """
    optimizer = torch.optim.Adagrad(
        [param for param in net.parameters() if param.requires_grad == True], lr=lr, lr_decay=lr_decay,
    )
    batch_processed_counter = 0

    net = net.to(device)
    criterion = criterion.to(device)

    net.train()

    # train for some number of epochs
    for e in range(epochs):
        valid_counter = 0
        epoch_val_losses = []
        losses = []
        print("Starting epoch", e + 1)

        # batch loop
        for inputs, labels in train_loader:
            batch_processed_counter += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            # output, h = net(inputs, h)
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            losses.append(loss.item())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs /
            # LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if batch_processed_counter % print_every == 0:
                valid_counter += 1
                # Get validation loss
                val_losses = []
                net.eval()
                num_correct = 0
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    output = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())
                    epoch_val_losses.append(val_loss.item())
                    
                    # convert output probabilities to predicted class (0 or 1)
                    pred = torch.round(output.squeeze())  # rounds to the nearest integer

                    # compare predictions to true label
                    correct_tensor = pred.eq(labels.float().view_as(pred)).cpu()
                    correct = np.squeeze(correct_tensor.numpy())
                    num_correct += np.sum(correct)
                    
                val_acc = num_correct / len(valid_loader.dataset)
                net.train()
                print(
                    "Epoch: {:2d}/{:2d}   ".format(e + 1, epochs),
                    "Batch: {:2d}  ".format(batch_processed_counter),
                    "Batch loss: {:.6f}   ".format(loss.item()),
                    "Val loss: {:.6f}".format(np.mean(val_losses)),
                    "Val acc: {:.6f}".format(val_acc),
                )
        print(len(epoch_val_losses))

        print(
            (
                "Finished epoch {}. Average batch loss: {}. "
                "Average validation loss: {}"
            ).format(e + 1, np.mean(losses), np.mean(epoch_val_losses))
        )


def reviews_test(net, test_loader, criterion=nn.BCELoss(), device=DEVICE):
    """
    Evaluate a trained network on the review data.

    Parameters
    ----------
    net: trained model
    test_loader: Dataloader containing the training data
    criterion: Loss function
    device: device to train on - cpu or cuda
    """
    test_losses = []  # track loss
    num_correct = 0

    net.eval()

    net = net.to(device)
    criterion = criterion.to(device)

    # iterate over test data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = net(inputs)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred)).cpu()
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))
