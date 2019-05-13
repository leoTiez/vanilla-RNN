import numpy as np
import matplotlib.pyplot as plt


def char_to_ind(char, book_chars):
    """
    Function to convert a character to the corresponding index in the vector
    :param char: Character
    :param book_chars: Characters in the book
    :return: index
    """
    return np.where(book_chars == char)[0]


def ind_to_char(ind, book_chars):
    """
    Converts an index to a character
    :param ind: Index
    :param book_chars: Characters in the book
    :return: Character
    """
    return book_chars[ind]


def convert_text_data(text):
    """
    Converts text data to an array with all characters which are used in the text
    :param text: Text to convert
    :return: Array with all used characters
    """
    text_array = np.asarray([char for char in text])
    book_chars = np.unique(text_array)
    return book_chars


def ind_to_hot(ind, size):
    """
    Converts an index to a hot-one notation vector
    :param ind: index
    :param size: length of the vector
    :return: hot-one encoded value
    """
    hot_vector = np.zeros(size, dtype=RNN.PRECISION)
    hot_vector[ind] = 1
    return hot_vector


def read_text_data(file_name='./text-data/goblet_book.txt'):
    """
    Read the text data from a file
    :param file_name: Name of the file
    :return: Text data as a string
    """
    text = None
    file = None
    try:
        file = open(file_name, "r")
        text = file.read()
    except:
        raise EOFError("Could not read file")
    finally:
        file.close()

    return text


def convert_seq(seq, book_chars):
    """
    Convert a seqence of one-hot encoded characters to an actual character sequence
    :param seq: The sequence of one-hot encoded characters
    :param book_chars: Characters in the book
    :return: The converted written text sequence
    """
    written_seq = ''
    for char_hot in seq:
        ind = np.where(char_hot == 1)[0][0]
        written_seq += ind_to_char(ind, book_chars)

    return written_seq


def softmax(s):
    """
    Private static function. Computes the softmax
    :param s: data batch
    :return: softmax computed for every data value
    """
    return (np.exp(s - np.max(s, axis=0, keepdims=True)) /
            np.sum(np.exp(s - np.max(s, axis=0, keepdims=True)), axis=0)).T


def numerical_grad(rnn, input_data, hot_labels, h=1e-8):
    """
    Numerical computation of the gradient for comparison against the analytical calculation
    :param rnn: The RNN instance
    :param input_data: The data which is used for the gradient calculation
    :param hot_labels: Hot-one encoded labels
    :param h: Distance parameter for the gradient calculation
    :return: Numerical gradients
    """
    grad_recurrent_weight = np.zeros(rnn.recurrent_weights.shape, dtype=RNN.PRECISION)
    recurrent_weight_org = rnn.recurrent_weights.copy()
    for num in np.ndindex(rnn.recurrent_weights.shape):
        rnn.recurrent_weights = recurrent_weight_org[:, :]
        rnn.recurrent_weights[num] += h
        loss_plus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        rnn.recurrent_weights[num] -= 2 * h
        loss_minus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        grad_recurrent_weight[num] = (loss_plus - loss_minus) / float(2 * h)

    rnn.recurrent_weights = recurrent_weight_org[:, :]

    grad_input_weight = np.zeros(rnn.input_weights.shape, dtype=RNN.PRECISION)
    input_weight_org = rnn.input_weights.copy()
    for num in np.ndindex(rnn.input_weights.shape):
        rnn.input_weights = input_weight_org[:, :]
        rnn.input_weights[num] += h
        loss_plus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        rnn.input_weights[num] -= 2 * h
        loss_minus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        grad_input_weight[num] = (loss_plus - loss_minus) / float(2 * h)

    rnn.input_weights = input_weight_org[:, :]

    grad_output_weight = np.zeros(rnn.output_weights.shape, dtype=RNN.PRECISION)
    output_weight_org = rnn.output_weights.copy()
    for num in np.ndindex(rnn.output_weights.shape):
        rnn.output_weights = output_weight_org[:, :]
        rnn.output_weights[num] += h
        loss_plus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        rnn.output_weights[num] -= 2 * h
        loss_minus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        grad_output_weight[num] = (loss_plus - loss_minus) / float(2 * h)

    rnn.output_weights = output_weight_org[:, :]

    grad_output_bias = np.zeros(rnn.output_bias.shape, dtype=RNN.PRECISION)
    output_bias_org = rnn.output_bias.copy()
    for num in np.ndindex(rnn.output_bias.shape):
        rnn.output_bias = output_bias_org[:]
        rnn.output_bias[num] += h
        loss_plus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        rnn.output_bias[num] -= 2 * h
        loss_minus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        grad_output_bias[num] = (loss_plus - loss_minus) / float(2 * h)

    rnn.output_bias = output_bias_org[:]

    grad_bias = np.zeros(rnn.bias.shape, dtype=RNN.PRECISION)
    bias_org = rnn.bias.copy()
    for num in np.ndindex(rnn.bias.shape):
        rnn.bias = bias_org[:]
        rnn.bias[num] += h
        loss_plus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        rnn.bias[num] -= 2 * h
        loss_minus, _, _, _, _ = rnn.forward_loss(input_data, hot_labels)
        grad_bias[num] = (loss_plus - loss_minus) / float(2 * h)

    rnn.bias = bias_org[:]

    return grad_output_weight, grad_recurrent_weight, grad_input_weight, grad_output_bias, grad_bias


def relative_distance(grad, grad_num):
    """
    Relative distance between two gradients
    :param grad: Analytical gradient
    :param grad_num: Numerical gradient for the comparison
    :return: The relative distance
    """
    return np.divide(
        np.absolute(grad - grad_num),
        np.absolute(grad) + np.absolute(grad_num) + np.finfo(RNN.PRECISION).eps
    )


def gradient_clipping(gradient):
    """
    Gradient clipping to avoid exploding gradients
    :param gradient: The gradient
    :return: Clipped gradient
    """
    return np.maximum(np.minimum(gradient, 5), -5)


def test_case(
        character_data,
        learning_input,
        learning_output,
        number_of_hidden_states=5,
        threshold=1e-4
):
    """
    Test function to check whether accuracy of the analytical gradients is sufficient
    :param character_data: The characters used in the text
    :param learning_input: The learning data sequence
    :param learning_output: One-hot encoded labels
    :param number_of_hidden_states: Number of hidden states in the RNN network
    :param threshold: Relative distances smaller than this threshold are assumed to be sufficient
    :return: Boolean to determine whether the criterion has been met
    """
    rnn = RNN(character_data, number_hidden_states=number_of_hidden_states)
    _, probability, out, hidden, linear = rnn.forward_loss(learning_input,
                                                           learning_output)
    grad_out_weight, grad_rec_weight, grad_in_weight, grad_out_bias, grad_bias = \
        rnn.backward_pass(
            learning_input,
            linear,
            hidden,
            probability,
            learning_output,
            clip_grads=False
        )

    grad_out_weight_num, grad_rec_weight_num, grad_in_weight_num, grad_out_bias_num, grad_bias_num = \
        numerical_grad(
            rnn,
            learning_input,
            learning_output
        )

    relative_out = relative_distance(grad_out_weight, grad_out_weight_num)
    relative_recurrent = relative_distance(grad_rec_weight, grad_rec_weight_num)
    relative_in = relative_distance(grad_in_weight, grad_in_weight_num)
    relative_out_bias = relative_distance(grad_out_bias, grad_out_bias_num)
    relative_bias = relative_distance(grad_bias, grad_bias_num)

    criterion_out_weights = np.all(relative_out < threshold)
    print '\n Output weights: '
    print 'Criterion satisfied', criterion_out_weights
    print 'Error rate', relative_out[relative_out >= threshold].size / float(rnn.output_weights.size)
    print 'Max distance', relative_out.max()

    criterion_rec_weights = np.all(relative_recurrent < threshold)
    print '\n Recurrent weights'
    print 'Criterion satisfied', criterion_rec_weights
    print 'Error rate', relative_recurrent[relative_recurrent >= threshold].size / float(rnn.recurrent_weights.size)
    print 'Max distance', relative_recurrent.max()

    criterion_in_weights = np.all(relative_in < threshold)
    print '\n Input weights'
    print 'Criterion satisfied', criterion_in_weights
    print 'Error rate', relative_in[relative_in >= threshold].size / float(rnn.input_weights.size)
    print 'Max distance', relative_in.max()

    criterion_output_bias = np.all(relative_out_bias < threshold)
    print '\n Output bias'
    print 'Criterion satisfied', criterion_output_bias
    print 'Error rate', relative_out_bias[relative_out_bias >= threshold].size / float(rnn.output_bias.size)
    print 'Max distance', relative_out_bias.max()

    criterion_bias = np.all(relative_bias < threshold)
    print '\n Bias'
    print 'Criterion satisfied', criterion_bias
    print 'Error rate', relative_bias[relative_bias >= threshold].size / float(rnn.bias.size)
    print 'Max distance', relative_bias.max()

    return criterion_out_weights \
           and criterion_rec_weights \
           and criterion_in_weights \
           and criterion_output_bias \
           and criterion_bias


def create_training_data_seq(characters, book_text, seq_len, start_index=0):
    """
    Splits a data sequence into learning input data and the corresponding output data
    :param characters: Characters used in the book
    :param book_text: The book text
    :param seq_len: Length of the sequence
    :param start_index: Index where to start in the book text
    :return: Learning input data and learning output data
    """
    learning_data = np.asarray([ind_to_hot(char_to_ind(char, characters), characters.shape[0])
                                for char in book_text[start_index:start_index + seq_len+1]], dtype=RNN.PRECISION)
    learning_input = learning_data[:-1]
    learning_output = learning_data[1:]

    return learning_input, learning_output


class RNN:

    PRECISION = np.float128

    def __init__(
            self,
            character_data,
            number_hidden_states=100,
            sigma=0.01,
            seq_len=None
    ):
        """
        Constructor for the RNN class
        :param character_data: Characters used in the book
        :param number_hidden_states: Number of hidden states in the network
        :param sigma: Weight for the random initialization of the parameters.
        :param seq_len: Length of a learnt sequence
        """
        self.number_hidden_states = number_hidden_states
        self.seq_len = seq_len

        self.characters = character_data
        self.number_of_characters = self.characters.shape[0]

        # trainable set parameters
        # bias parameters
        self.bias = np.zeros(self.number_hidden_states, dtype=RNN.PRECISION)
        self.output_bias = np.zeros(self.number_of_characters, dtype=RNN.PRECISION)

        # weights
        self.recurrent_weights = np.random.randn(
            self.number_hidden_states,
            self.number_hidden_states
        ).astype(RNN.PRECISION) * sigma
        self.input_weights = np.random.randn(
            self.number_hidden_states,
            self.number_of_characters
        ).astype(RNN.PRECISION) * sigma
        self.output_weights = np.random.randn(
            self.number_of_characters,
            self.number_hidden_states
        ).astype(RNN.PRECISION) * sigma

    def process_input(
            self,
            input_vector,
            hidden_states
    ):
        """
        Transforms the input to an output for a single character according to the definition of the RNN
        :param input_vector: One-hot encoded
        :param hidden_states: Values of the hidden states
        :return: The output of the process
        """
        linear_transform = self.recurrent_weights.dot(hidden_states) + self.input_weights.dot(input_vector) + self.bias
        hidden = np.tanh(linear_transform).astype(RNN.PRECISION)
        output = self.output_weights.dot(hidden) + self.output_bias
        probability = softmax(output)

        return probability, output, hidden, linear_transform

    def generate_data(
            self,
            input_vector,
            hidden_states=None,
            seq_length=10
    ):
        """
        Generates a sequence given an input character in one-hot notation
        :param input_vector: Input character in one-hot notation
        :param hidden_states: Initial status of the hidden states. Default is zero
        :param seq_length: Length of the synthesised sequence
        :return: The synthesised sequence
        """
        if hidden_states is None:
            hidden_states = np.zeros(self.number_hidden_states, dtype=RNN.PRECISION)

        generated_seq = [input_vector]
        for _ in range(seq_length):
            probability, _, hidden_states, _ = self.process_input(input_vector, hidden_states)
            input_vector = ind_to_hot(
                np.random.choice(self.number_of_characters, 1, p=probability.astype(np.float64)),
                self.number_of_characters
            )
            generated_seq.append(input_vector)

        return generated_seq

    def forward_loss(
            self,
            input_data,
            hot_labels,
            hidden_states=None
    ):
        """
        Forward pass and calculation of the loss
        :param input_data: Input character sequence in one-hot notation
        :param hot_labels: Labels in one-hot notation
        :param hidden_states: Initial status of the hidden states
        :return: The loss / costs, array with the probability after the softmax calculation
        array with the output without the softmax transformation. array with the hidden states for every character,
        array with the linear transformation after using the recurrent information
        """
        if hidden_states is None:
            hidden_states = np.zeros(self.number_hidden_states, dtype=RNN.PRECISION)

        if len(input_data.shape) < 2:
            input_data = np.asarray([input_data], dtype=RNN.PRECISION)

        if len(hot_labels.shape) < 2:
            hot_labels = np.asarray([hot_labels], dtype=RNN.PRECISION)

        loss = 0
        linear_trans_vectors = []
        output_vectors = []
        hidden_state_vectors = [hidden_states]
        probability_vectors = []
        for input_vector, label in zip(input_data, hot_labels):
            probability, output, hidden_states, linear_trans = self.process_input(input_vector, hidden_states)
            linear_trans_vectors.append(linear_trans)
            hidden_state_vectors.append(hidden_states)
            output_vectors.append(output)
            probability_vectors.append(probability)
            loss -= np.log(label.dot(probability))

        return loss, \
               np.asarray(probability_vectors, dtype=RNN.PRECISION), \
               np.asarray(output_vectors, dtype=RNN.PRECISION), \
               np.asarray(hidden_state_vectors, dtype=RNN.PRECISION),\
               np.asarray(linear_trans_vectors, dtype=RNN.PRECISION)

    def backward_pass(
            self,
            input_vectors,
            linear_trans_vectors,
            hidden_state_vectors,
            probability_vectors,
            hot_labels,
            clip_grads=True
    ):
        """
        Backward pass of the back propagation
        :param input_vectors: Input data sequence in one-hot notation
        :param linear_trans_vectors: Array with vectors after the first linear transformation
        :param hidden_state_vectors: The hidden states which where used for computing the output
        :param probability_vectors: Array with values of the Softmax layer
        :param hot_labels: Labels in one-hot encoding
        :param clip_grads: Flag to determine whether or not to clip the gradients.
        It is recommended to only set it to zero when running the tests against the numerical calcualtion
        :return: gradient for the output weights, gradient for the recurrent weights, gradient for the input
        weights, gradient for the output bias, gradient for the input bias
        """
        grad_out = (-(hot_labels - probability_vectors)).astype(RNN.PRECISION)
        grad_output_weights = grad_out.T.dot(hidden_state_vectors[1:, :])
        grad_output_bias = np.sum(grad_out, axis=0)

        grad_a_wrt_h = 1 - np.tanh(linear_trans_vectors)**2

        left_term = grad_out[-1].dot(self.output_weights)

        # first gradient without a_t+1
        grad_a = left_term.dot(np.diag(grad_a_wrt_h[-1]))

        grad_recurrent_weights = np.zeros(self.recurrent_weights.shape, dtype=RNN.PRECISION)
        grad_input_weights = np.zeros(self.input_weights.shape, dtype=RNN.PRECISION)
        grad_bias = np.zeros(self.bias.shape, dtype=RNN.PRECISION)

        grad_recurrent_weights += np.outer(grad_a, hidden_state_vectors[-2])
        grad_input_weights += np.outer(grad_a, input_vectors[-1])
        grad_bias += grad_a

        for num, (grad_a_h, hidden_vector, input_vector) in enumerate(reversed(
                zip(grad_a_wrt_h[:-1, :], hidden_state_vectors[:-2, :], input_vectors[:-1, :])
        )):
            left_term = grad_out[-(num + 2)].dot(self.output_weights)
            grad_h = left_term + grad_a.dot(self.recurrent_weights)
            grad_a_h = np.diag(grad_a_h)
            grad_a = grad_h.dot(grad_a_h)

            grad_recurrent_weights += np.outer(grad_a, hidden_vector)
            grad_input_weights += np.outer(grad_a, input_vector)
            grad_bias += grad_a

        if clip_grads:
            grad_output_weights = gradient_clipping(grad_output_weights)
            grad_recurrent_weights = gradient_clipping(grad_recurrent_weights)
            grad_input_weights = gradient_clipping(grad_input_weights)
            grad_output_bias = gradient_clipping(grad_output_bias)
            grad_bias = gradient_clipping(grad_bias)

        return grad_output_weights, grad_recurrent_weights, grad_input_weights, grad_output_bias, grad_bias

    def adagrad_training(
            self,
            book_text,
            epochs=7,
            nabla=0.01,
            epsilon=1e-8,
            save_plot=True
    ):
        """
        AdaGrad learning procedure
        :param book_text: Text sequence
        :param epochs: Number of epochs for the training
        :param nabla: Learning rate
        :param epsilon: Small value to avoid dividing by zero used by the AdaGrad calculation
        :param save_plot: Flag to determine whether or not saving the plot
        :return: None
        """
        smooth_loss = 0
        loss_over_time = []
        for _ in range(epochs):
            initial_hidden = np.zeros(self.number_hidden_states, dtype=RNN.PRECISION)
            grad_sum_out = 0
            grad_sum_rec = 0
            grad_sum_in = 0
            grad_sum_out_bias = 0
            grad_sum_bias = 0

            for num, start_index in enumerate(range(0, len(book_text), self.seq_len)):
                input_training, output_training = create_training_data_seq(
                    self.characters,
                    book_text,
                    seq_len=self.seq_len,
                    start_index=start_index
                )

                loss, probability, output, hidden, linear_trans = self.forward_loss(
                    input_training,
                    output_training,
                    hidden_states=initial_hidden
                )

                grad_out, grad_rec, grad_in, grad_out_bias, grad_bias = self.backward_pass(
                    input_training,
                    linear_trans,
                    hidden,
                    probability,
                    output_training
                )

                grad_sum_out += grad_out**2
                grad_sum_rec += grad_rec**2
                grad_sum_in += grad_in**2
                grad_sum_out_bias += grad_out_bias**2
                grad_sum_bias += grad_bias**2

                self.output_weights -= (nabla / np.sqrt(grad_sum_out + epsilon)) * grad_out
                self.recurrent_weights -= (nabla / np.sqrt(grad_sum_rec + epsilon)) * grad_rec
                self.input_weights -= (nabla / np.sqrt(grad_sum_in + epsilon)) * grad_in
                self.output_bias -= (nabla / np.sqrt(grad_sum_out_bias + epsilon)) * grad_out_bias
                self.bias -= (nabla / np.sqrt(grad_sum_bias + epsilon)) * grad_bias

                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                loss_over_time.append(smooth_loss)
                if num % 100 == 0:
                    print 'Smooth Loss at step', num, ':', smooth_loss

                if num % 500 == 0:
                    print '\nSynthesised text\n'
                    print convert_seq(self.generate_data(input_training[0], hidden_states=initial_hidden, seq_length=200), self.characters)
                    print '\n'

                initial_hidden = hidden[-1]

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.plot(range(len(loss_over_time)), loss_over_time, 'g-', label='Smooth loss')
        ax.set_title('Smooth loss over update steps')
        plt.legend(loc='upper right')
        if save_plot:
            plt.savefig('img4/smooth-loss.png')
        else:
            plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    book_text_ = read_text_data()

    character_data_ = convert_text_data(book_text_)
    rnn_ = RNN(character_data_, number_hidden_states=100, sigma=.01, seq_len=25)

    seq_len_ = 25
    learning_input_, learning_output_ = create_training_data_seq(rnn_.characters, character_data_, seq_len_)

    if not test_case(character_data_, learning_input_, learning_output_):
        raise ValueError("The gradients do not match the set criterion")

    rnn_.adagrad_training(book_text_)
    print convert_seq(rnn_.generate_data(ind_to_hot(char_to_ind('H', rnn_.characters), rnn_.number_of_characters),
                                         seq_length=1000), rnn_.characters)

