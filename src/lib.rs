pub mod layer;
pub mod mlp;

#[cfg(test)]
mod tests {
    use crate::layer::utils::sigmoid;

    #[test]
    fn sig_test() {
        assert_eq!(sigmoid(5.0), 0.9933071490757268);
    }
}

