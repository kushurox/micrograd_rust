use micrograd_rust::layer::{neuron::Layer, value::Operation, neuron::Neuron};


fn main() {
    let mut my_layer = Layer::new(2, 1, Operation::Tanh);
    // let cool_neuron = Neuron::new(2, Operation::Tanh);
    println!("{}", my_layer);
}
