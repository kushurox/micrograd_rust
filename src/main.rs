use micrograd_rust::layer::neuron::Layer;


fn main() {
    let mut my_layer = Layer::new(2, 3);
    println!("{}", my_layer.forward().len());
}
