use std::{ops::{Add, Mul, Sub}, collections::VecDeque, rc::Rc, cell::RefCell, fmt::Display};

use super::utils::{sigmoid, d_sigmoid, d_tanh};

type Node = Rc<RefCell<Value>>;

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    None,
    Add,
    Sub,
    Mul,
    Tanh,
    Sigmoid
}


#[derive(Clone, Debug)]
pub struct Value {
    pub val: f32,
    pub prev: [Option<Node>; 2],
    pub operation: Operation,
    pub grad: f32
}



impl Value{
    pub fn new(val: f32) -> Self {
        Value {val, prev: [None, None], operation: Operation::None, grad: 1.0}
    }

    pub fn backprop(&mut self) {
        let temp = Rc::new(RefCell::new(self.clone()));
        let mut to_visit = VecDeque::from([temp]);
        loop {
            if to_visit.is_empty() {
                break;
            }
            let b1 = to_visit.pop_front().unwrap();
            let b2 = b1.borrow_mut();
            if let [Some(val1), Some(val2)] = &b2.prev {

                match b2.operation {
                    Operation::Add => {val1.borrow_mut().grad = b2.grad; val2.borrow_mut().grad = b2.grad},
                    Operation::Sub => {val1.borrow_mut().grad = b2.grad; val2.borrow_mut().grad = -b2.grad}, // v1 - v2
                    Operation::Mul => {val1.borrow_mut().grad = val2.borrow().val*b2.grad; val2.borrow_mut().grad = val1.borrow().val*b2.grad},
                    _ => {
                        panic!("Binary Operation Not implemented!");
                    }
                }
                to_visit.push_back(val1.clone());
                to_visit.push_back(val2.clone());
            } else if let [Some(val1), None] = &b2.prev {
                let mut bind = val1.borrow_mut(); // child node

                /*
                    if y = tanh(x);
                    y' = 1-tanh^2(x)
                 */

                match b2.operation {
                    Operation::Tanh => {
                        // derivative of tanh(x) w.r.t x = 1 - tanh^2(x)
                        bind.grad = d_tanh(bind.val)*b2.grad;
                    },
                    Operation::Sigmoid => {
                        // derivative of sigmoid(x) = sigmoid(x)[1-sigmoid(x)]
                        bind.grad = d_sigmoid(bind.val) * b2.grad;
                    }
                    _ => {
                        panic!("Function not implemented!");
                    }
                }
                to_visit.push_back(val1.clone());
            }

        }
    }
}


impl Add for Value{
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        Value {val: self.val + rhs.val, prev: [Some(Rc::new(RefCell::new(self))), Some(Rc::new(RefCell::new(rhs)))], operation: Operation::Add, grad:1.0}
    }
}

impl Sub for Value{
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        Value {val: self.val - rhs.val, prev: [Some(Rc::new(RefCell::new(self))), Some(Rc::new(RefCell::new(rhs)))], operation: Operation::Sub, grad:1.0}
    }
}

impl Mul for Value{
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        Value {val: self.val * rhs.val, prev: [Some(Rc::new(RefCell::new(self))), Some(Rc::new(RefCell::new(rhs)))], operation: Operation::Mul, grad:1.0}
    }
}

impl Mul<f32> for Value {
    type Output=Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Value {val: self.val * rhs, prev: [Some(Rc::new(RefCell::new(self))), Some(Rc::new(RefCell::new(Value::new(rhs))))], operation: Operation::Mul, grad:1.0}
        
    }
}

impl Value {
    pub fn tanh(self) -> Self{
        Value {val: self.val.tanh(), prev: [Some(Rc::new(RefCell::new(self))), None], operation: Operation::Tanh, grad:1.0}
    }
    pub fn sigmoid(self) -> Self {
        Value {val: sigmoid(self.val), prev:[Some(Rc::new(RefCell::new(self))), None], operation: Operation::Sigmoid, grad:1.0}
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.val)
    }
}
