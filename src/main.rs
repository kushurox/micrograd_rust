use std::time::Instant;

use value::Value;
mod value;

fn main() {
    let t1 = Instant::now();
    let a = Value::new(3.0);
    let b = Value::new(4.0);
    let mut c = a*b;
    c.backprop();
    let t2 = Instant::now() - t1;
    println!("{}", t2.as_nanos());
}
