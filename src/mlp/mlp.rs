use crate::layer::{neuron::Layer, self};

pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(inps: u32, hls: u32, ns: u32){
        // inps: number of inputs
        // ls: Number of hidden layers
        // ns: Number of neurons per layer
        let mut layers = Vec::new();
        let input_layer = Layer::new(inps, ns, crate::layer::value::Operation::Tanh);
        layers.push(input_layer);
        for _i in 0..hls {
            layers.push(Layer::new(ns, ns, crate::layer::value::Operation::Tanh))
        }
        todo!()
    }
}