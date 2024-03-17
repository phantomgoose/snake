use rand::prelude::*;
use reverse::Tape;

const INPUT_LAYER_WIDTH: usize = 10;
const MIDDLE_LAYER_WIDTH: usize = 64;
const HIDDEN_LAYER_WIDTH: usize = 64;
const OUTPUT_LAYER_WIDTH: usize = 4;
const MUTATION_RATE: f32 = 0.7;

#[derive(Clone)]
struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    activation: Activation,
}

#[derive(Copy, Clone)]
enum Activation {
    Sigmoid,
    ReLU,
}

impl Layer {
    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(self.biases.len());
        for (i, bias) in self.biases.iter().enumerate() {
            let mut output = inputs
                .iter()
                .zip(self.weights[i].iter())
                .map(|(input, weight)| input * weight)
                .sum::<f32>()
                + bias;
            output = match self.activation {
                Activation::Sigmoid => 1.0 / (1.0 + (-output).exp()),
                Activation::ReLU => {
                    if output < 0.0 {
                        0.0
                    } else {
                        output
                    }
                }
            };
            outputs.push(output);
        }
        outputs
    }

    /// Creates a child by randomly swapping weights & biases between the 2 given layers
    fn crossover(&self, other: &Layer) -> Layer {
        // TODO: fix all the cloning here (probably by switching to arrays for weights/biases)
        let mut cloned_self = self.clone();
        let mut cloned_other = other.clone();
        for i in 0..self.weights.len() {
            if random() {
                std::mem::swap(&mut cloned_self.weights[i], &mut cloned_other.weights[i]);
                std::mem::swap(&mut cloned_self.biases[i], &mut cloned_other.biases[i]);
            }
        }

        cloned_self
    }

    fn mutate(mut self) -> Layer {
        let mut rng = thread_rng();
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] *= 1. + rng.gen_range(-MUTATION_RATE..=MUTATION_RATE);
            }
            self.biases[i] *= 1. + rng.gen_range(-MUTATION_RATE..=MUTATION_RATE);
        }

        self
    }
}

#[derive(Clone)]
pub(crate) struct NeuralNetwork {
    layers: Vec<Layer>,
}

fn get_rand_vec_of_size(size: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    let mut vec: Vec<f32> = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(rng.gen_range(-1.0..=1.0));
    }
    vec
}

fn get_layer_of_size(input_size: usize, output_size: usize, activation: Activation) -> Layer {
    let mut weights = Vec::with_capacity(output_size);
    for _ in 0..output_size {
        weights.push(get_rand_vec_of_size(input_size));
    }
    let biases = get_rand_vec_of_size(output_size);
    Layer {
        weights,
        biases,
        activation,
    }
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNetwork {
    pub(crate) fn new() -> Self {
        let activation = Activation::ReLU;
        let input_layer = get_layer_of_size(INPUT_LAYER_WIDTH, MIDDLE_LAYER_WIDTH, activation);
        let hidden_layer = get_layer_of_size(MIDDLE_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, activation);
        let output_layer = get_layer_of_size(HIDDEN_LAYER_WIDTH, OUTPUT_LAYER_WIDTH, activation);
        Self {
            layers: vec![input_layer, hidden_layer, output_layer],
        }
    }

    fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        assert!(
            !self.layers.is_empty(),
            "Expected layers to exist in the neural network."
        );
        assert_eq!(
            self.layers[0].weights[0].len(),
            inputs.len(),
            "Input length must match input layer size"
        );

        self.layers
            .iter()
            .fold(inputs.to_vec(), |acc, layer| layer.forward(&acc))
    }

    fn probabilities(&self, inputs: &[f32]) -> Vec<f32> {
        let logits = self.predict(inputs);

        // TODO: make softmax a first class activation function by refactoring the layer forward func
        let tape = Tape::new();
        let params = tape.add_vars(
            logits
                .iter()
                .map(|val| *val as f64)
                .collect::<Vec<f64>>()
                .as_slice(),
        );
        talos::functions::softmax(&params)
            .iter()
            .map(|val| val.val as f32)
            .collect()
    }

    pub(crate) fn classify<T>(&self, inputs: &[f32], classes: &[T]) -> T
    where
        T: Copy,
    {
        let prediction = self.probabilities(inputs);
        assert_eq!(prediction.len(), classes.len());

        let max_index = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap_or_else(|| {
                panic!(
                    "Expected to find a max value in prediction {:?}, but it was None instead",
                    prediction
                )
            });

        classes[max_index]
    }

    pub(crate) fn mate(&self, other: &NeuralNetwork) -> NeuralNetwork {
        let mut new_network = NeuralNetwork { layers: vec![] };

        for (i, layer) in self.layers.iter().enumerate() {
            let new_layer = layer.crossover(&other.layers[i]).mutate();
            new_network.layers.push(new_layer);
        }

        new_network
    }
}
