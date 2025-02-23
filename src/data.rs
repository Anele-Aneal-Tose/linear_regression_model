use rand::Rng;

pub fn generate_data(num_samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_values = Vec::with_capacity(num_samples);
    let mut y_values = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let x = rng.gen_range(-10.0..10.0);
        let noise = rng.gen_range(-0.5..0.5);
        let y = 2.0 * x + 1.0 + noise;

        x_values.push(x);
        y_values.push(y);
    }

    (x_values, y_values)
}
