use burn::tensor::{Tensor};
use burn::optim::{Adam, Optimizer};  // Import the correct optimizer trait
use burn::nn::{Module, Trainable};   // Module and Trainable are needed for parameters
use burn::tensor::ops::{Add, Mul, Sub};  // These operations are directly part of the Tensor API now

use crate::model::LinearRegression;
use crate::data::{generate_data};

pub fn train_model(model: &mut LinearRegression, num_epochs: usize, learning_rate: f32) {
    let (x_train, y_train) = generate_data(100);

    let x_tensor = Tensor::from(x_train);
    let y_tensor = Tensor::from(y_train);

    // Create an optimizer with the correct syntax
    let mut optimizer = Adam::default(); // This is the correct way to initialize Adam

    // Define the learning rate schedule (for example, this will be a fixed learning rate for now)
    optimizer.set_learning_rate(learning_rate);

    for epoch in 0..num_epochs {
        // Forward pass
        let predictions = model.forward(&x_tensor);

        // Compute loss (Mean Squared Error)
        let loss = (predictions - &y_tensor).powi(2).mean().unwrap();

        // Backpropagate
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Print loss every 10 epochs
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss: {:.4}", epoch, loss.item::<f32>());
        }
    }
}
