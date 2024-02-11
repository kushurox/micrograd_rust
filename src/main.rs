use std::rc::Rc;
use std::cell::RefCell;

use micrograd_rust::layer::neuron::Layer;
use micrograd_rust::layer::neuron::Neuron;
use micrograd_rust::layer::value::Value;
use micrograd_rust::layer::value::Operation;
use micrograd_rust::mlp::mlp::MLP;

fn main() {
    let my_model = MLP::new(2, 1, 1, |v1, v2| {v1 + v2});
    // let inputs = vec![Value::new(1.0), Value::new(2.0)];
    // let res = my_model.forward(inputs);
    let params = my_model.parameters();
    for lparam in params {
        for nparam in lparam {
            for vparam in nparam {
                println!("{}", vparam)
            }
        }
        println!("--------------------------------------------")
    }
}