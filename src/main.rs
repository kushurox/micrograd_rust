use std::time::Instant;

use value::Value;
mod value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let e = a * b;
    let d = e + c;
    let f = Value::new(-2.0);
    let mut L = d * f;
    L.backprop();
    println!("{:#?}", L);
}
