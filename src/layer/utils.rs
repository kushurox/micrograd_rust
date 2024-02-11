use super::value::Value;

pub fn sigmoid(a: f32) -> f32 {
    let denom = 1.0 + (-a).exp();
    1.0/denom
}

pub fn d_sigmoid(a: f32) -> f32 {
    sigmoid(a)*(1.0-sigmoid(a))
}

pub fn d_tanh(a: f32) -> f32 {
    1.0 - a.tanh().powi(2)
}

pub fn squared_diff(ypred: Value, y: Value) -> Value{
    let temp = Value::new(2.0);
    ypred.pow(temp.clone()) - y.pow(temp)
}

#[macro_export]
macro_rules! layer_inputs {
    // Base case: empty vector
    () => {
        vec![]
    };

    // Recursive case: add first argument as a Value and recurse with remaining
    ($first:expr $(, $rest:expr)*) => {
        vec![Value::new($first)]
            .into_iter()
            .chain::<Vec<Value>>(layer_inputs!($($rest)*))
            .collect()
    };
}