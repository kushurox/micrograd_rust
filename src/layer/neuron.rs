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

    pub fn forward(&mut self) -> Value {
        let mut res=self.w.pop().as_ref().unwrap().clone();
        for v in self.w.iter() {
            res = res + v.clone();
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

    pub fn forward(&mut self) -> Vec<Value>{
        let mut res = Vec::new();
        while let Some(mut n) = self.neurons.pop() {
            res.push(n.forward());
        }
        res
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for ws in &self.w {
            write!(f, "{} ", ws)?;
        }
        write!(f, "]")
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