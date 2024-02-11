use crate::layer::{neuron::Layer, value::Value};
use crate::layer::value::Operation;

type LossFn = fn(ypred: Value, ytrue:Value)->Value;
type Nparam=Vec<Value>;
type Lparam=Vec<Nparam>;
type Mparam=Vec<Lparam>;

pub struct MLP {
    layers: Vec<Layer>,
    loss: LossFn 
}

impl MLP {
    pub fn new(inps: u32, hls: u32, ns: u32, loss: LossFn) -> Self{
        // inps: number of inputs
        // ls: Number of hidden layers
        // ns: Number of neurons per layer
        let mut layers = Vec::new();
        let input_layer = Layer::new(inps, ns, Operation::Tanh);
        layers.push(input_layer);
        for _i in 0..hls {
            layers.push(Layer::new(ns, ns, Operation::Tanh))
        }
        let output_layer = Layer::new(ns, 1, Operation::Sigmoid);
        layers.push(output_layer);
        MLP {layers, loss}
    }

    pub fn parameters(&self) -> Mparam {
        let mut params = Vec::new();
            for l in &self.layers {
                params.push(l.parameters());
            }
        params
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        let mut curr_inp = inputs;
        for l in &self.layers {
            curr_inp = l.forward(curr_inp);
        }
        curr_inp
    }
}