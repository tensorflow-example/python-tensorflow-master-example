import numpy as np

data = open('train.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# model hyper-parameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25    # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01   # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
bh = np.zeros((hidden_size, 1))                         # hidden bias
Why = np.random.randn(vocab_size, hidden_size) * 0.01   # hidden to output
by = np.zeros((vocab_size, 1))                          # output bias


def one_hot_vector(k, i):
    x = np.zeros(shape=(k, 1))
    x[i] = 1
    return x


def loss_fun(inputs, targets, hprev):
    """
    inputs, targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # forward pass: compute loss
    for t in range(len(inputs)):
        xs[t] = one_hot_vector(k=vocab_size, i=inputs[t])
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        loss += -np.log(ps[t][targets[t]][0])  # softmax (cross-entropy loss)

    # backward pass: compute gradients
    dWxh, dWhh, dWhy, dbh, dby = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext   # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    # clip to mitigate exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss/len(inputs), dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, sample_size):
    """
    sample a sequence of integers from the model
    - h is memory state
    - n is number of chars to sample
    """
    ix = np.random.randint(vocab_size)  # get random seed char ix
    x = one_hot_vector(k=vocab_size, i=ix)
    sample_ixs = []
    for t in range(sample_size):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(a=vocab_size, p=p.ravel())
        x = one_hot_vector(k=vocab_size, i=ix)
        sample_ixs.append(ix)
    return ''.join(ix_to_char[ix] for ix in sample_ixs)


n, p = 0, 0
smooth_loss = -np.log(1.0 / vocab_size)   # loss at iteration 0
hprev = np.zeros(shape=(hidden_size, 1))  # instantiate RNN memory
# memory variables for Adagrad
mWxh, mWhh, mWhy, mbh, mby = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)
while True:
    if p + seq_length + 1 >= data_size:
        p = 0                               # go from start of data
        hprev = np.zeros(shape=(hidden_size, 1))  # reset RNN memory

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # input `seq_length` characters into RNN and retrieve loss & gradients
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 1000 == 0:
        sample_chars = sample(hprev, sample_size=200)
        print('iter %d, loss: %f\n----\n %s \n----\n' % (n, smooth_loss, sample_chars))

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh,  Whh,  Why,  bh,  by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length
    n += 1
