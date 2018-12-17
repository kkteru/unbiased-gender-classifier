# from __future__ import print_funtion
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import argparse
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import _pickle as pickle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from src.model import AutoEncoder


def normalize_images(images):
    return images.float().div_(255.0).mul_(2.0).add_(-1)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.relu3 = nn.LeakyReLU(0.2)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(8)
        self.relu4 = nn.LeakyReLU(0.2)

        self.fc = nn.Linear(in_features=2 * 2 * 8, out_features=num_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, input):
        output = self.conv1_bn(self.conv1(input))
        output = self.drop1(self.relu1(output))

        output = self.conv2_bn(self.conv2(output))
        output = self.relu2(output)

        output = self.pool(output)

        output = self.conv3_bn(self.conv3(output))
        output = self.drop3(self.relu3(output))

        output = self.conv4_bn(self.conv4(output))
        output = self.relu4(output)

        # print(output.shape)
        output = output.view(-1, 2 * 2 * 8)

        output = self.drop(self.fc(output))
        return output


class Exp():
    def __init__(self, args, X_train, X_test, X_val, y_train, y_test, y_val):
        self.args = args
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

    def get_predictions(self, data):
        model = torch.load(open(os.path.join(self.args.name, 'model.pth'), 'rb')).to(device)
        model.eval()

        X_test_ = data
        test_acc = 0.0
        if X_test_.shape[0] % args.batch_size == 0:
            num_iterations = X_test_.shape[0] // args.batch_size
        else:
            num_iterations = X_test_.shape[0] // args.batch_size + 1
        predictions = []
        for i in range(num_iterations):
            image = normalize_images(X_test_[i * args.batch_size:(i + 1) * args.batch_size, :, :, :])
            image = image.to(device)

            if args.remove_race:
                model_ae = torch.load(args.ae).to(device)
            else:
                model_ae = torch.load(args.ae_vanilla).to(device)
            model_ae.eval()

            image = model_ae.encode(image)[-1]

            # Predict classes using images from the test set
            outputs = model(image)

            # test_loss += loss.item() * image.size(0)
            _, prediction = torch.max(outputs.data, 1)
            predictions.extend(prediction.cpu().numpy().astype(int))
            # test_acc += float(torch.sum(prediction == label.data))/float(label.size(0))
        # print(" Accuracy: {} ".format(test_acc/float(num_iterations)))
        return np.array(predictions).astype(int)

    def evaluate_classwise(self, predictions, race_test, gender_test):
        a = []
        for c in range(5):
            c_idxs = np.where(race_test == c)[0]
            # print predictions
            # print c_idxs
            pred = predictions[c_idxs.astype(int)]
            y = gender_test[c_idxs]
            print('For class ' + str(c) + ' Accuracy = ' + str(accuracy_score(y, pred)))
            a.append(accuracy_score(y, pred))
        return a

    def evaluate(self, model, data, model_ae=None):
        model.eval()
        model_ae.eval()
        X_test_ = data[0]
        y_test_ = data[1]
        test_acc = 0.0
        if X_test_.shape[0] % args.batch_size == 0:
            num_iterations = X_test_.shape[0] // self.args.batch_size
        else:
            num_iterations = X_test_.shape[0] // self.args.batch_size + 1
        predictions = []
        for i in range(num_iterations):
            image = normalize_images(X_test_[i * self.args.batch_size:(i + 1) * self.args.batch_size, :, :, :])
            label = torch.LongTensor(y_test_[i * self.args.batch_size:(i + 1) * self.args.batch_size])
            image = image.to(device)
            label = label.to(device)

            image = model_ae.encode(image)[-1]
            # Predict classes using images from the test set
            outputs = model(image)

            # test_loss += loss.item() * image.size(0)
            _, prediction = torch.max(outputs.data, 1)
            predictions.extend(prediction.cpu().numpy().astype(int))
            # test_acc += float(torch.sum(prediction == label.data))/float(label.size(0))

        print(" Accuracy: {} ".format(accuracy_score(y_test_, predictions)))
        return accuracy_score(y_test_, predictions)

    def train(self, model, num_epochs, model_ae=None):
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        patience = 0
        lr = self.args.lr
        for epoch in tqdm(range(num_epochs)):
            model.train()
            model_ae.train()
            train_acc = 0.0
            train_loss = 0.0

            optimizer = Adam(model.parameters(), lr=lr)

            num_iterations = self.X_train.shape[0] // args.batch_size
            print('Epoch: ' + str(epoch + 1) + ' / ' + str(num_epochs))
            for i in tqdm(range(num_iterations)):
                image = normalize_images(self.X_train[i * self.args.batch_size:(i + 1) * self.args.batch_size, :, :, :])
                label = torch.LongTensor(self.y_train[i * self.args.batch_size:(i + 1) * self.args.batch_size])
                image = image.to(device)
                label = label.to(device)

                image = model_ae.encode(image)[-1]
                # Clear all accumulated gradients
                optimizer.zero_grad()
                # Predict classes using images from the test set
                outputs = model(image)
                # Compute the loss based on the predictions and actual labels
                loss = loss_fn(outputs, label)
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                optimizer.step()

                train_loss += loss.item() * image.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_acc += float(torch.sum(prediction == label.data)) / float(label.size(0))
                if i % args.log_interval == 0:
                    print("Iteration {}, Train Accuracy: {} , TrainLoss: {}".format(i + 1, train_acc / float(i + 1), train_loss / float(i + 1),))
            # # Call the learning rate adjustment function
            # adjust_learning_rate(epoch)

            # Compute the average acc and loss over all 50000 training images
            print('Validation')
            val_acc = self.evaluate(model, (self.X_val, self.y_val), model_ae)
            # Evaluate on the test set
            # test_acc = test()

            # Save the model if the test acc is greater than our current best
            if val_acc >= best_acc:
                torch.save(model, open(os.path.join(self.args.name, 'model.pth'), 'wb'))
                best_acc = val_acc
                print('Saved model')
            else:
                patience += 1
                if lr > 0.00001:
                    lr /= 1.5

            if patience > 5:
                break
            # Print the metrics

    def run(self, model, model_ae=None):
        self.train(model, self.args.num_epochs, model_ae)
        print('Test')
        model = torch.load(open(os.path.join(self.args.name, 'model.pth'), 'rb'))
        self.evaluate(model, (self.X_test, self.y_test), model_ae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Race Classifier')
    parser.add_argument('--name', type=str, default='default',
                        help='location of the data corpus')
    parser.add_argument('--data', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--num-runs', type=int, default='1',
                        help='number of runs')
    parser.add_argument('--seed', type=int, default='10',
                        help='random seed')
    parser.add_argument('--batch-size', type=int, default='30',
                        help='batch size')
    parser.add_argument('--num-classes', type=int, default='2',
                        help='batch size')
    parser.add_argument('--log-interval', type=int, default='100',
                        help='batch size')
    parser.add_argument('--num-epochs', type=int, default='50',
                        help='batch size')
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu id')
    parser.add_argument('--lr', type=float, default='0.001',
                        help='learning rate')
    parser.add_argument('--train-and-eval', action='store_true',
                        help='Load and store data')
    parser.add_argument('--eval', action='store_true',
                        help='Load and store data')
    parser.add_argument('--remove_race', action='store_true',
                        help='Encode image to remove race info before passing to the classifier')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA')
    parser.add_argument("--ae", type=str, default="./models/ae_race_invariant.pth",
                        help="Path to race invariant autoencoder model")
    parser.add_argument("--ae_vanilla", type=str, default="./models/ae_vanilla.pth",
                        help="Path to vanilla autoencoder model")
    # parser.add_argument('--out-dir', type=str, default='../data/wikitext-2/annotated',
    #                     help='location of the output directory')

    args = parser.parse_args()

    if not os.path.exists(args.name):
        os.makedirs(args.name)

    images = torch.load(os.path.join(args.data, 'images_256_256.pth'))
    print('Shape of images loaded : ' + str(images.shape))
    attributes = torch.load(os.path.join(args.data, 'attributes.pth'))
    gender_attributes = torch.LongTensor(attributes['Gender']).numpy()
    print('Shape of attributes loaded : ' + str(gender_attributes.shape))
    race_attributes = torch.LongTensor(attributes['Race']).numpy()
    cnt = Counter(race_attributes)
    print(cnt)

    per_race_items_test = int(race_attributes.shape[0] * .2 / 5)
    val_items = int(race_attributes.shape[0] * .1)
    np.random.seed(args.seed)

    all_indices = np.arange(race_attributes.shape[0])

    test_indices = []
    for c in range(5):
        c_idxs = np.where(race_attributes == c)[0]
        c_idxs = np.random.choice(c_idxs, per_race_items_test, replace=False)
        test_indices.extend(c_idxs)

    all_indices = np.arange(race_attributes.shape[0])
    train_indices = list(set(all_indices) - set(test_indices))

    val_indices = np.random.choice(train_indices, val_items, replace=False)
    train_indices = list(set(train_indices) - set(val_indices))

    X_train = images[train_indices]
    y_train = gender_attributes[train_indices]
    race_train = race_attributes[train_indices]

    X_val = images[val_indices]
    y_val = gender_attributes[val_indices]
    race_val = race_attributes[val_indices]

    X_test = images[test_indices]
    y_test = gender_attributes[test_indices]
    race_test = race_attributes[test_indices]

    cnt = Counter(race_test)
    print(cnt)

    if not os.path.isfile('image_test.pb'):
        pickle.dump(X_test, open('image_test.pb', 'wb'))
        pickle.dump(y_test, open('gender_test.pb', 'wb'))
        pickle.dump(race_test, open('race_test.pb', 'wb'))
    # X_train, X_test, y_train, y_test = train_test_split(images, gender_attributes, test_size=0.1, random_state=args.seed)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed)

    torch.manual_seed(args.seed)
    if args.cuda:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        else:
            print('CUDA not found')

    device = torch.device("cuda:" + str(args.gpu) if args.cuda and torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = SimpleCNN(args.num_classes)
    model.to(device)

    if args.remove_race:
        model_ae = torch.load(args.ae).to(device)
    else:
        model_ae = torch.load(args.ae_vanilla).to(device)

    # model_ae.eval()

    for params in model_ae.parameters():
        params.requires_grad = False

    e = Exp(args, X_train, X_test, X_val, y_train, y_test, y_val)

    if args.eval:
        predictions = e.get_predictions(X_test)
        print('Overall Accuracy = ' + str(accuracy_score(y_test, predictions)))
        a = e.evaluate_classwise(predictions, race_test, y_test)

    elif args.train_and_eval:
        all_perf = []
        for i in tqdm(range(args.num_runs)):
            print('*' * 89)
            print('Run ' + str(i + 1))

            model.apply(weights_init)
            e.run(model, model_ae)

            print('-' * 89)
            predictions = e.get_predictions(X_test)
            print('Overall Accuracy = ' + str(accuracy_score(y_test, predictions)))
            a = e.evaluate_classwise(predictions, race_test, y_test)
            all_perf.append(a)

        print('#' * 89)
        all_perf = np.array(all_perf)
        mean_perf = np.mean(all_perf, axis=0)
        std_perf = np.std(all_perf, axis=0)
        for c in range(5):
            print('For class ' + str(c) + ' Mean Accuracy = ' + str(mean_perf[c]) + '+/-' + str(std_perf[c]))
    else:
        model.apply(weights_init)
        e.run(model, model_ae)
