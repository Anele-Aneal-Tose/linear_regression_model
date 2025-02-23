use burn::tensor::{Tensor};
use burn::tensor::ops::Add;  // Only Add operation is needed here
use burn::nn::{Module, Trainable};

pub struct LinearRegression {
    weights: Tensor<f32>,
    bias: Tensor<f32>,
}

impl LinearRegression {
    pub fn new() -> Self {
        // Initialize weights and bias with random values
        let weights = Tensor::randn(&[1], burn::device::Cpu);
        let bias = Tensor::randn(&[1], burn::device::Cpu);

        LinearRegression { weights, bias }
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        // Use direct operations on Tensor for forward pass
        &x * &self.weights + &self.bias
    }
}
