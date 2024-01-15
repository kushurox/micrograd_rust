use std::fmt::Display;

use rand::random;

use super::value::Value;
use super::value::Operation;


#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>, // input-side edges.
    b: Value,
    activation: Operation
}

impl Neuron {
    pub fn new(n: u32, activation: Operation) -> Self{
        // n: Number of weights associated with the neuron
        let mut w = Vec::new();
        for _i in 0..n {
            w.push(Value::new(random()))
        }
        let b = Value::new(random());
        Neuron {w, b, activation}
    }

    pub fn forward(&self, x: Vec<Value>) -> Value {
        if x.len() != self.w.len() {
            panic!("Inputs Lengths not matching Weight's Length");
        }
        let mut res = self.w[0].clone() * x[0].clone();
        for i in 1..x.len() {
            res = res + (self.w[i].clone() * x[i].clone());
        }

        res = res + self.b.clone();

        match self.activation {
            Operation::Tanh => res.tanh(),
            _ => panic!("Activation not implemented")
        }
    }
}


#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(nin: u32, nn: u32, activation: Operation) -> Self {
        // nin: Number of inputs per neuron
        // nn: Number of neurons
        let mut neurons = Vec::<Neuron>::new();
        for _i in 0..nn {
            neurons.push(Neuron::new(nin, activation));
        }
        Layer {neurons}
    }

    pub fn forward(&self, inputs: Vec::<Value>) -> Vec<Value>{
        // inputs: outputs from previous layer
        let mut outs = Vec::<Value>::new();
        for n in &self.neurons {
            outs.push(n.forward(inputs.clone()));
        }
        outs
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neuron(ws=[ ")?;
        for ws in &self.w {
            write!(f, "{} ", ws)?;
        }
        write!(f, "], b={})", self.b)
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[\n")?;
        for ns in &self.neurons {
            write!(f, "\t{}\n", ns)?;
        }
        write!(f, "]")
    }
}