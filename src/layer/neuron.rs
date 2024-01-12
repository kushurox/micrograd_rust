use std::fmt::Display;

use rand::random;

use super::value::Value;


#[derive(Debug)]
struct Neuron {
    w: Vec<Value>, // input-side edges.
    b: Value
}

impl Neuron {
    pub fn new(n: u32) -> Self{
        // n: Number of weights associated with the neuron
        let mut w = Vec::new();
        for _i in 0..n {
            w.push(Value::new(random()))
        }
        let b = Value::new(random());
        Neuron {w, b}
    }

    pub fn forward(&mut self) -> Value {
        let mut res=self.w.pop().as_ref().unwrap().clone();
        for v in self.w.iter() {
            res = res + v.clone();
        }
        res + self.b.clone()
    }
}


#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(nin: u32, nn: u32) -> Self {
        // nin: Number of inputs per neuron
        // nn: Number of neurons
        let mut neurons = Vec::<Neuron>::new();
        for _i in 0..nn {
            neurons.push(Neuron::new(nin));
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
        write!(f, "Neuron({})", self.w.len())
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer({})", self.neurons.len())
    }
}