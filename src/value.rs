use std::{ops::{Add, Mul, Sub}, collections::VecDeque, rc::Rc, cell::RefCell};

type Node = Rc<RefCell<Value>>;

#[derive(Clone, Copy, Debug)]
enum Operation {
    None,
    Add,
    Sub,
    Mul,
}


#[derive(Clone, Debug)]
pub struct Value {
    pub val: f32,
    pub prev: Option<[Node; 2]>,
    operation: Operation,
    grad: f32
}



impl Value{
    pub fn new(val: f32) -> Self {
        Value {val, prev: None, operation: Operation::None, grad: 1.0}
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
            if let Some(vals) = &b2.prev {
                let [val1, val2] = vals;
                to_visit.push_back(val1.clone());
                to_visit.push_back(val2.clone());

                match b2.operation {
                    Operation::Add => {val1.borrow_mut().grad = b2.grad; val2.borrow_mut().grad = b2.grad},
                    Operation::Sub => {val1.borrow_mut().grad = b2.grad; val2.borrow_mut().grad = -b2.grad}, // v1 - v2
                    Operation::Mul => {val1.borrow_mut().grad = val2.borrow().val*b2.grad; val2.borrow_mut().grad = val1.borrow().val*b2.grad},
                    _ => {
                        panic!("Not implemented!");
                    }
                }
            }

        }
    }
}



impl Add for Value{
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        Value {val: self.val + rhs.val, prev: Some([Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))]), operation: Operation::Add, grad:1.0}
    }
}

impl Sub for Value{
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        Value {val: self.val - rhs.val, prev: Some([Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))]), operation: Operation::Sub, grad:1.0}
    }
}

impl Mul for Value{
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        Value {val: self.val * rhs.val, prev: Some([Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))]), operation: Operation::Mul, grad:1.0}
    }
}