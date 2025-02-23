mod train;
mod model;
mod data;

//use burn::tensor::TensorExt;
use burn::tensor::{Tensor};
use crate::model::LinearRegression;
use crate::train::train_model;
use crate::data::generate_data;
use textplots::{Chart, Plot};

fn main() {
    // Create a new LinearRegression model
    let mut model = LinearRegression::new();

    // Train the model with 1000 epochs and a learning rate of 0.01
    train_model(&mut model, 1000, 0.01);

    // Test the model
    let (x_test, y_test) = generate_data(20);
    let x_tensor = Tensor::from(x_test);
    let predictions = model.forward(&x_tensor);

    // Print predictions and true values
    for (pred, true_val) in predictions.iter().zip(y_test.iter()) {
        println!("Pred: {:.4}, True: {:.4}", pred.item::<f32>(), true_val);
    }

    // Visualize predictions vs true values using textplots
    let mut chart = Chart::new(80, 20);
    chart.lineplot(&x_test, &y_test).label("True Values");
    chart.lineplot(&x_test, &predictions.iter().map(|x| x.item::<f32>()).collect::<Vec<f32>>()).label("Predictions");
    chart.display();
}
