use std::fmt::Display;


use rand::random;

use super::value::Value;
use super::value::Operation;

type Nparam=Vec<Value>;
type Lparam=Vec<Nparam>;

#[derive(Debug, Clone)]
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
            panic!("Dimensions not matching!");
        }
        let mut res = Value::new(0.0);
        for (w, xi) in self.w.iter().zip(x) {
            res = res + (w.clone() * xi);
        }
        res = res + self.b.clone();
        match self.activation {
            Operation::Tanh => res.tanh(),
            Operation::Sigmoid => res.sigmoid(),
            _ => panic!("Not implemented")
        }
    }

    pub fn parameters(&self) -> Nparam {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
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
        let mut results = Vec::new();
        for n in &self.neurons {
            results.push(n.forward(inputs.clone()));
        }
        results
    }

    pub fn parameters(&self) -> Lparam {
        let mut params = Vec::new();
        for n in &self.neurons {
            params.push(n.parameters())
        }
        params
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