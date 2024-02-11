use std::{cell::RefCell, fmt::Display, ops::{Add, Mul}, rc::Rc};

use super::utils::{d_sigmoid, d_tanh, sigmoid};


#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Tanh,
    Sigmoid
}


#[derive(Debug, Clone, PartialEq)]
pub struct ValueData {
    pub val: f32,
    operation: Option<Operation>,
    pub prev: (Option<Value>, Option<Value>),
    grad: f32
}

#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub ptr: Rc<RefCell<ValueData>>
}

impl Value {
    pub fn new(val: f32) -> Self {
        let inner = ValueData {val, operation: None, prev: (None, None), grad: 0.0};
        Self { ptr: Rc::new(RefCell::new(inner)) }
    }

    pub fn new_from(val: f32, operation: Option<Operation>, prev: (Option<Value>, Option<Value>)) -> Self {
        let inner = ValueData {val, operation, prev, grad: 0.0};
        Self { ptr: Rc::new(RefCell::new(inner)) }
    }

    pub fn backwards(&mut self) {

        let curr = self.clone();
        curr.ptr.borrow_mut().grad = 1.0;
        let (value1, value2) = curr.ptr.borrow().prev.clone();
        let mut next;

        match (value1, value2) {
            (Some(v1), None) => {
                next=vec![v1.clone()];
                if curr.ptr.borrow().operation.unwrap() == Operation::Tanh {
                    let dval = d_tanh(v1.ptr.borrow().val);
                    v1.ptr.borrow_mut().grad = dval * curr.ptr.borrow().grad;
                }
                else if curr.ptr.borrow().operation.unwrap() == Operation::Sigmoid {
                    let dval =  d_sigmoid(v1.ptr.borrow().val);
                    v1.ptr.borrow_mut().grad = dval * curr.ptr.borrow().grad;
                }
            },
            (Some(v1), Some(v2)) => {
                next=vec![v2.clone(), v1.clone()];
                if curr.ptr.borrow().operation.unwrap() == Operation::Add {
                    v1.ptr.borrow_mut().grad = curr.ptr.borrow().grad;
                    v2.ptr.borrow_mut().grad = curr.ptr.borrow().grad;
                }
                else if curr.ptr.borrow().operation.unwrap() == Operation::Mul {
                    v1.ptr.borrow_mut().grad = v2.ptr.borrow().val * curr.ptr.borrow().grad;
                    v2.ptr.borrow_mut().grad = v1.ptr.borrow().val * curr.ptr.borrow().grad;
                }
            },
            _ => return
        }

        while let Some(currv) = next.pop() {
            let (val1, val2) = currv.ptr.borrow().prev.clone();

            match (val1, val2) {
                (None, None) => continue,
                (None, Some(_)) => continue, // will never happen
                (Some(v1), None) => {
                    next.push(v1.clone());
                    if currv.ptr.borrow().operation.unwrap() == Operation::Tanh {
                        let dval = d_tanh(v1.ptr.borrow().val);
                        v1.ptr.borrow_mut().grad = dval * currv.ptr.borrow().grad;
                    }
                    else if currv.ptr.borrow().operation.unwrap() == Operation::Sigmoid {
                        let dval =  d_sigmoid(v1.ptr.borrow().val);
                        v1.ptr.borrow_mut().grad = dval * currv.ptr.borrow().grad;
                    }
                },  // single operand
                (Some(v1), Some(v2)) => { // binary operations
                    next.push(v1.clone());
                    next.push(v2.clone());

                    if currv.ptr.borrow().operation.unwrap() == Operation::Add {
                        v1.ptr.borrow_mut().grad = currv.ptr.borrow().grad;
                        v2.ptr.borrow_mut().grad = currv.ptr.borrow().grad;
                    }
                    else if currv.ptr.borrow().operation.unwrap() == Operation::Mul {
                        v1.ptr.borrow_mut().grad = v2.ptr.borrow().val * currv.ptr.borrow().grad;
                        v2.ptr.borrow_mut().grad = v1.ptr.borrow().val * currv.ptr.borrow().grad;
                    }
                },
            }

        }
    }

}



impl Add for Value {
    type Output=Self;

    fn add(self, rhs: Self) -> Self::Output {
        let prev = (Some(self.clone()), Some(rhs.clone()));
        Value::new_from(self.ptr.borrow().val + rhs.ptr.borrow().val, Some(Operation::Add), prev)
    }
}

impl Mul for Value {
    type Output=Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let prev = (Some(self.clone()), Some(rhs.clone()));
        Value::new_from(self.ptr.borrow().val * rhs.ptr.borrow().val, Some(Operation::Mul), prev)
    }
}

impl Value {
    pub fn tanh(self) -> Self{
        Value::new_from(self.ptr.borrow().val.tanh(), Some(Operation::Tanh), (Some(self.clone()), None))
    }
    pub fn sigmoid(self) -> Self {
        Value::new_from(sigmoid(self.ptr.borrow().val), Some(Operation::Sigmoid), (Some(self.clone()), None))
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({}, grad={})", self.ptr.borrow().val, self.ptr.borrow().grad)
    }
}