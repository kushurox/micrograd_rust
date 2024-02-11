use std::rc::Rc;
use std::cell::RefCell;

use micrograd_rust::layer::neuron::Layer;
use micrograd_rust::layer::neuron::Neuron;
use micrograd_rust::layer::utils::squared_diff;
use micrograd_rust::layer::value::Value;
use micrograd_rust::layer::value::Operation;
use micrograd_rust::mlp::mlp::MLP;

fn main() {
    let mut my_model = MLP::new(2, 1, 1);
    my_model.config(0.1, squared_diff);
    let x_train = vec![
        vec![Value::new(1.0), Value::new(2.0)],
        vec![Value::new(1.0), Value::new(2.0)],
        vec![Value::new(1.0), Value::new(2.0)]];

    let y_train = vec![Value::new(1.0), Value::new(1.0), Value::new(1.0)];
    let params = my_model.parameters();
    for lparam in params {
        for nparam in lparam {
            for vparam in nparam {
                println!("{}", vparam)
            }
        }
        println!("--------------------------------------------")
    }

    my_model.train(x_train, y_train);

    
}