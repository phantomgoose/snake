use rand::prelude::*;
use reverse::Tape;

const INPUT_LAYER_WIDTH: usize = 2;
const OUTPUT_LAYER_WIDTH: usize = 4;

struct Layer {
    weights: Vec<Vec<f32>>,
    // Now each layer has multiple neurons, each with its own set of weights
    biases: Vec<f32>,
    activation: Activation,
}

enum Activation {
    Sigmoid,
    ReLU,
    // Softmax,
}

impl Layer {
    fn new(weights: Vec<Vec<f32>>, biases: Vec<f32>, activation: Activation) -> Layer {
        assert_eq!(weights.len(), biases.len());
        Layer {
            weights,
            biases,
            activation,
        }
    }

    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut outputs = vec![];
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
}

pub(crate) struct NeuralNetwork {
    layers: Vec<Layer>,
}

fn get_rand_vec_of_size(size: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    let mut vec: Vec<f32> = Vec::with_capacity(size);
    for i in 0..size {
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

impl NeuralNetwork {
    pub(crate) fn new() -> Self {
        let input_layer =
            get_layer_of_size(INPUT_LAYER_WIDTH, OUTPUT_LAYER_WIDTH, Activation::ReLU);
        Self {
            layers: vec![input_layer],
        }
    }

    fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert!(!self.layers.is_empty());
        assert_eq!(self.layers[0].weights[0].len(), inputs.len());

        self.layers
            .iter()
            .fold(inputs, |acc, layer| layer.forward(&acc))
    }

    pub(crate) fn probabilities(&self, inputs: Vec<f32>) -> Vec<f32> {
        let logits = self.predict(inputs);
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
}
