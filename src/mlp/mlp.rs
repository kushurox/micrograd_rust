use crate::layer::utils::squared_diff;
use crate::layer::{neuron::Layer, value::Value};
use crate::layer::value::Operation;

type LossFn = fn(ypred: Value, ytrue:Value)->Value;
type Nparam=Vec<Value>;
type Lparam=Vec<Nparam>;
type Mparam=Vec<Lparam>;

// not implementing optimizers, sticking to simple gradient descent
pub struct MLP {
    pub layers: Vec<Layer>,
    loss: LossFn,
    learning_rate: f32
}

impl MLP {
    pub fn new(inps: u32, hls: u32, ns: u32) -> Self{
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
        MLP {layers, loss: squared_diff ,learning_rate: 0.05}
    }

    pub fn config(&mut self, learning_rate: f32, loss_fn: LossFn){
        self.learning_rate = learning_rate;
        self.loss = loss_fn
    }

    pub fn train(&mut self, x_train: Vec<Vec<Value>>, y_train: Vec<Value>) {
        let mut params = self.parameters();
        for (inputs, output) in x_train.into_iter().zip(y_train) {
            let res = self.forward(inputs);
            let mut cost = (self.loss)(res, output);
            println!("Cost: {}", cost);
            cost.backwards();
            for lparam in &mut params {
                for nparam in lparam {
                    for vparam in nparam {
                        let grad = vparam.ptr.borrow().grad;
                        let mut temp = vparam.ptr.borrow_mut();
                        temp.val = temp.val - (-self.learning_rate*grad); 
                    }
                }
            }
        }
    }

    pub fn parameters(&self) -> Mparam {
        let mut params = Vec::new();
            for l in &self.layers {
                params.push(l.parameters());
            }
        params
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Value {
        let mut curr_inp = inputs;
        for l in &self.layers {
            curr_inp = l.forward(curr_inp);
        }
        curr_inp[0].clone()
    }
}